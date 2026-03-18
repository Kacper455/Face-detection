import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('kolonko.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

while True:
    ret, frame = cap.read()
    if (ret == True):

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)

        for (x, y, w, h) in faces:
             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             ROI_img = gray_img[y:y+w, x:x+w]
             ROI_color = frame[y:y+w, x:x+h]
             eyes = eye_cascade.detectMultiScale(ROI_img,1.05,5)

             for (ex, ey, ew, eh) in eyes:
                 cv2.circle(ROI_color, (ex + 30, ey + 30), int((ey/2)), (0, 255, 0), 2)
        if (h):
            print("active")
            print(ret)

        else:
            print("inactive")
            print(ret)

        cv2.imshow('1', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    #cv2.destroyAllWindows()
# plt.imshow(RGB_img)
# plt.waitforbuttonpress()
# plt.close('all')