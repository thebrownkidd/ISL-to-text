import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

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
# image = "0-9/1/asl1.jpg"
# img = cv2.imread(image)[40:1040,460:1460]
# cv2.imshow("image",img)
# cv2.waitKey(0)
# cv2.imwrite("Cropped_ISL1.jpg",img)
dir = "C:/Projects/ISL Research/0-9/"

for x in os.listdir(dir):
    image_path = dir+x+"/"+os.listdir(dir+x)[0]
    print(image_path)
    ima = os.listdir(dir+x)[0]
    img = cv2.imread(image_path)[40:1040,460:1460]
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.imwrite(image_path+"_crop.jpg",img)
    base_options = python.BaseOptions(model_asset_path= "C:/Projects/ISL Research/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options= base_options,num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    img_loc = image_path+"_crop.jpg"
    image = image = mp.Image.create_from_file(img_loc)
    # image  = image[40:1040,460:1460]
    detection_result = detector.detect(image)

    annotated_image,X,Y = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow("image",annotated_image)
    cv2.waitKey(0)
    cv2.imwrite(image_path + "_crop_marked.jpg",annotated_image)
    
    pos = []
    for i in range(len(X)):
      X[i] = X[i]*1000
      Y[i] = Y[i]*1000
      pos.append(X[i] + Y[i]*1000)
    f = open(str(image_path+"XY.txt"),'a')
    f.write(str(X))
    f.write(str(Y))
    f.close()
    f = open(str(image_path+"posencoded.txt"),'a')
    f.write(str(pos))
    f.close()