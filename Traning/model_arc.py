# !pip install torchtext
# importing required libraries
import mediapipe as mp
# from mediapipe.framework.formats import landmark_pb2
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe import solutions
import os
# import pandas as pd
import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
import pandas as pd
import numpy as np
# import seaborn as sns
# import torchtext
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
print(torch.__version__)

class Embedding_in(nn.Module):
    def __init__(self, embed_dim,path_pose_model='/home/Rupali/ISL-to-text/INCLUDE50/INCLUDE 50/MP_Models/pose_landmarker_heavy.task', path_hand_model='/home/Rupali/ISL-to-text/INCLUDE50/INCLUDE 50/MP_Models/hand_landmarker.task'):
        super(Embedding_in,self).__init__()
        
        self.dim = embed_dim
        self.pose_model_path = path_pose_model
        self.path_model_hand = path_hand_model
        
        # model_path = '/kaggle/input/pose-and-hand/pose_landmarker_heavy.task'
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarker.use_gpu = True
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.options_pose = self.PoseLandmarkerOptions(base_options = self.BaseOptions(model_asset_path = self.pose_model_path),running_mode=self.VisionRunningMode.IMAGE)
        
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarker.use_gpu = True
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        self.options_hand = self.HandLandmarkerOptions(
            base_options= self.BaseOptions(model_asset_path=self.path_model_hand),num_hands=2)
        
        self.hand_embd = nn.Sequential(
                    nn.Linear(42,42),
                    nn.Tanh(),
                    nn.Linear(42,126),
                    nn.Tanh(),
                    nn.Linear(126,126),
                    nn.Tanh()
        )
        
        self.pose_embd = nn.Sequential(
                    nn.Linear(44,44),
                    nn.Tanh(),
                    nn.Linear(44,126),
                    nn.Tanh(),
                    nn.Linear(126,126),
                    nn.Tanh(),
                    nn.Linear(126,126),
                    nn.Tanh()
        )
    def forward(self,frame):
        X_left = [0 for i in range(21)]
        X_right = [0 for i in range(21)]
        Y_left = [0 for i in range(21)]
        Y_right = [0 for i in range(21)]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        h,w,_  = frame.shape
        with self.PoseLandmarker.create_from_options(self.options_pose) as landmarker:
            results = landmarker.detect(mp_image)
        # frame,_,_ = draw_landmarks_pose(frame,results)
        if(len(results.pose_landmarks)!=0):
            X_pose = [results.pose_landmarks[0][i].x for i in range(len(results.pose_landmarks[0]))]
            Y_pose = [results.pose_landmarks[0][i].y for i in range(len(results.pose_landmarks[0]))]
            landmarks = results.pose_landmarks[0]
            min_x = min(X_pose)*w -130
            max_x = max(X_pose)*w +130
            min_y = min(Y_pose)*h -130
            max_y = max(Y_pose)*h +130
            img = frame[int(min_y):int(max_y),int(min_x):int(max_x)]
            #cv2.imwrite('/kaggle/working/crop.png',img)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
            with self.HandLandmarker.create_from_options(self.options_hand) as landmarker:
                landmarks = landmarker.detect(image)
            # an_img,x,y = draw_landmarks_hand(img,landmarks)
            if(len(landmarks.hand_landmarks)!=0):
                l = 0
                r = 0
                for j in range(len(landmarks.hand_landmarks)):
                    if landmarks.handedness[j][0].display_name == 'Right':
                        X_right = [landmarks.hand_landmarks[j][i].x for i in range(len(landmarks.hand_landmarks[j]))]
                        Y_right = [landmarks.hand_landmarks[j][i].y for i in range(len(landmarks.hand_landmarks[j]))]
                        r = 1
                    if landmarks.handedness[j][0].display_name == 'Left':
                        X_left = [landmarks.hand_landmarks[j][i].x for i in range(len(landmarks.hand_landmarks[j]))]
                        Y_left = [landmarks.hand_landmarks[j][i].y for i in range(len(landmarks.hand_landmarks[j]))]
                        l = 1
                
        
        right = torch.tensor(np.array([*X_right,*Y_right],dtype=np.float32),requires_grad=True)
        # right = torch.tensor(Y_right,requires_grad=True)
        left = torch.tensor(np.array([*X_left,*Y_left],dtype=np.float32),requires_grad=True)
        # left = torch.tensor(Y_left,requires_grad=True)
        pose = torch.tensor(np.array([*X_pose[:22],*Y_pose[:22]],dtype=np.float32),requires_grad=True)
        # ypose = torch.tensor(*Y_pose[:22],requires_grad=True)
        # hand = torch.stack((left,right))
        l_emb = self.hand_embd(left)
        r_emb = self.hand_embd(right)
        pose_emb = self.pose_embd(pose)
        
        return torch.stack((l_emb,r_emb,pose_emb))
                # dt = dt = [*X_pose,*Y_pose,*X_left,*Y_left,*X_right,*Y_right,word,l,r]
                # dta = pd.concat([dta,pd.DataFrame(dt).T])
                #print(pd.DataFrame(dt).T)
                # data = pd.concat([data,pd.DataFrame(dt).T])
                #cv2.imwrite('/kaggle/working/lmed'+file+'.png',an_img)
            
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=126, n_heads=1):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #126
        self.n_heads = n_heads   #2
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #63
       
        #key,query and value matrixes  
        self.query_matrix = nn.Linear(self.embed_dim , self.single_head_dim ,bias=False)
        self.key_matrix = nn.Linear(self.embed_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.embed_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim*3 ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        # batch_size = key.size(0)
        # seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        # seq_length_query = query.size(1)
        
        # 32x10x512
        # key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        # query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        # value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (batch, single_head_dim)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        # q = torch.transpose(q,0,1)  
        k = torch.transpose(k,0,1)  
        # v = torch.transpose(v,0,1)  
       
        # computes attention
        # adjust key for matrix multiplication
        # k_adjusted = torch.transpose(k,0,1)
        product = torch.matmul(q, k)  #[w11,w12,w13.....] #(batch, batch)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        scores = scores.view(1,-1)
        print("Scores: ",scores.shape)
        # concat = scores.view()
        
        output =  self.out(scores)
        print("Output: ",output.shape)
       
        return output   

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=126, expansion_factor=4, n_heads=1):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        # self.dropout2 = nn.Dropout(0.2)

        self.final = nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim,embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim,embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim,embed_dim)
        )
        
        self.valw = nn.Linear(378,126)
    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        
        attention_out= self.attention(key,query,value)  #32x10x512
        value_weight = self.valw(value.view(1,-1))
        attention_residual_out = attention_out + value_weight  #32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x10x512
        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.final(feed_fwd_residual_out)

        return norm2_out



class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=1, expansion_factor=4, n_heads=1):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = Embedding_in(embed_dim) #Change to Embedding_in
        # self.scale = nn.Linear(num_layers*)
        # self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    def forward(self, x):
        out = self.embedding_layer(x)
        for layer in self.layers:
            out = layer(out,out,out)
        return out  #32x10x512

class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length,num_layers=1, expansion_factor=4, n_heads=1):
        super(Transformer, self).__init__()
        
        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        
        # self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        # self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.convert = nn.Sequential(
                    nn.Linear(126,114),
                    nn.Tanh(),
                    nn.Linear(114,114),
                    nn.Tanh(),
                    nn.Linear(114,114),
                    nn.Softmax()
        )
    
    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask    

    def decode(self,src,trg):
        """
        for inference
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size,seq_len = src.shape[0],src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len): #10
            out = self.decoder(out,enc_out,trg_mask) #bs x seq_len x vocab_dim
            # taking the last token
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self, src):
        """
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        # trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        outputs = self.convert(enc_out)
   
        # outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs