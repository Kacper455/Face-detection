import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

source = cv2.VideoCapture(0)

source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
source.set(cv2.CAP_PROP_FPS, 60)
source.set(cv2.CAP_PROP_FRAME_HEIGHT, 1050)
source.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

if not source.isOpened():
    print("Error: Could not open video file.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

while True:
    ret, frame = source.read()
    if (ret == True):

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 4)

        for (x, y, w, h) in faces:
             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             ROI_img = gray_img[y:y+w, x:x+w]
             ROI_color = frame[y:y+w, x:x+h]
             eyes = eye_cascade.detectMultiScale(ROI_img,1.1,6)

             for (ex, ey, ew, eh) in eyes:
                 cv2.circle(ROI_color, (ex + 30, ey + 30), int((ey/2)), (0, 255, 0), 2)

        cv2.imshow('0', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

source.release()
