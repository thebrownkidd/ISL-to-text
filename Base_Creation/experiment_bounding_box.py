import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# # Initialize MediaPipe Hands and Drawing modules
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # Path to the input image
# image_path = 'C:/Projects/ISL-to-text/0-9/5/asl5.jpg'
# output_path = 'C:/Projects/ISL-to-text/0-9/5/asl5.2.jpg'

# # Read the image from file
# image = cv2.imread(image_path)

# # Convert the BGR image to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Initialize MediaPipe Hands
# with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands:
#     # Process the image and detect hands
#     results = hands.process(image_rgb)

#     # Draw hand annotations and bounding boxes on the image
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Calculate the bounding box coordinates
#             x_min = min([landmark.x for landmark in hand_landmarks.landmark])
#             x_max = max([landmark.x for landmark in hand_landmarks.landmark])
#             y_min = min([landmark.y for landmark in hand_landmarks.landmark])
#             y_max = max([landmark.y for landmark in hand_landmarks.landmark])

#             # Convert normalized coordinates to pixel values
#             height, width, _ = image.shape
#             x_min = int(x_min * width)
#             x_max = int(x_max * width)
#             y_min = int(y_min * height)
#             y_max = int(y_max * height)

#             # Draw the bounding box
#             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#             # Optionally, draw hand landmarks and connections
#             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # Save the processed image to a file
#     cv2.imwrite(output_path, image)
#     print(f'Processed image saved to {output_path}')

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
dir = "C:/Projects/ISL-to-text/0-9/"

for x in os.listdir(dir):
    image_path = dir+x+"/"+os.listdir(dir+x)[0]
    print(image_path)
    ima = os.listdir(dir+x)[0]
    img = cv2.imread(image_path)[40:1040,460:1460]
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    cv2.imwrite(image_path+"_crop.jpg",img)
    base_options = python.BaseOptions(model_asset_path= "C:/Projects/ISL-to-text/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options= base_options,num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    img_loc = image_path+"_crop.jpg"
    image = image = mp.Image.create_from_file(img_loc)
    # image  = image[40:1040,460:1460]
    detection_result = detector.detect(image)

    annotated_image,X,Y = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow("image",annotated_image)
    cv2.waitKey(0)
    # cv2.imwrite(image_path + "_crop_marked.jpg",annotated_image)
    print(detection_result)
    # pos = []
    # for i in range(len(X)):
    #   X[i] = X[i]*1000
    #   Y[i] = Y[i]*1000
    #   pos.append(X[i] + Y[i]*1000)
    # f = open(str(image_path+"XY.txt"),'w')
    # f.write(str(X))
    # f.write(str(Y))
    # f.close()
    # f = open(str(image_path+"posencoded.txt"),'w')
    # f.write(str(pos))
    # f.close()