import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# This model failed
sign_pred = tf.keras.models.load_model('C:/Projects/ISL-to-text/0-9/Models/42in_88acc.keras')
MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image,x_coordinates,y_coordinates

cap = cv2.VideoCapture(0)
i = 0
while cap.isOpened():
    Data = []
    suc, image = cap.read()
    image = image[0:1920,0:1080]
    if not suc:
        continue
    image = cv2.flip(image, 1)
    # cv2.imshow('image',image)
    # print(type(image))
    base_options = python.BaseOptions(model_asset_path= "C:/Projects/ISL-to-text/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options= base_options,num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=image)
    result = detector.detect(mp_image)
    if len(result.hand_landmarks) >0:
        img,X,Y = draw_landmarks_on_image(mp_image.numpy_view(),result)
        for i in range(len(X)):
            Data.append((X[i])*1000)
            Data.append((Y[i])*1000)
        inp = pd.DataFrame(Data)
        print(inp.shape)
        res = sign_pred.predict(inp.T)
        print(res)
    else:
        img = image.copy()
    cv2.imshow('image',img)
    if cv2.waitKey(5) & 0xFF == 27:
        break