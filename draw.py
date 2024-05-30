from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
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

# dir = "C:/Projects/ASl Research/0-9/"
# images = os.listdir(dir)
# x,y = 0.0,0.0
# for x in images:
#     img_loc = dir+"/"+x
#     img = cv2.imread(img_loc)
#     # img = cv2.cvtColor(img,cv2.COLOR_RGBA)
#     # cv2.imshow("image",img)
#     # cv2.waitKey(0)

#     base_options = python.BaseOptions(model_asset_path= "C:/Projects/ASl Research/hand_landmarker.task")
#     options = vision.HandLandmarkerOptions(base_options= base_options,num_hands=2)
#     detector = vision.HandLandmarker.create_from_options(options)

#     image = mp.Image.create_from_file(img_loc)

#     detection_result = detector.detect(image)

#     annotated_image,X,Y = draw_landmarks_on_image(image.numpy_view(), detection_result)
#     # cv2.imshow("landmarked",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#     # cv2.waitKey(0)
#     f = open(str(x+".txt"),'a')
#     f.write(str(X))
#     f.write(str(Y))
#     f.close()
#     f = open(str('landmarked'+x),'a')
#     landmarkedimg = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
#     f.write(str(landmarkedimg))
#     f.close()