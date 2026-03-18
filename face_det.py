import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bs
import pandas as pd
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('360_F_530010960_WkmW6q8FIZ42LkEQoSRCLTNTw0WgJqKY.jpg')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

faces = face_cascade.detectMultiScale(RGB_img, 1.5, 6)

for (x, y, w, h) in faces:
     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
     ROI_img = RGB_img[y:y+w, x:x+w]
     ROI_color = img[y:y+w, x:x+h]
     eyes = eye_cascade.detectMultiScale(ROI_img,1.5,6)

     for (ex, ey, ew, eh) in eyes:
         eye_center = (x+ ex + ew//2, y + ey + eh//2)
         radius = int(round((ew+eh)*0.25))
         cv2.circle(ROI_color, (ex + 30,  ey + 30), int((ey/3)), (0, 255, 0), 2)

cv2.imshow("1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(RGB_img)
# plt.waitforbuttonpress()
# plt.close('all')
# urls = []
# numeric = 11
# def urls_to_scrap():
#     for i in range(numeric):
#         url = f"{address}?pn={i}"
#         urls.append(url)
#     return urls
#
#
# links = []
# def scrape():
#     for idx, url in enumerate(urls_to_scrap(), start=1):
#         response = requests.get(url).content
#         soup = bs(response, 'html.parser')
#
#         for x in soup.find_all('li'):
#             label = x.find('span', id ="data-test")
#             for y in label:
#                 text = y.get_text(strip=True)
#                 status.append(text)
#                 print(status)
#     return status
#
#
# def create_df():
#     numer = []
#     for i in range(numeric):
#         i=i+1
#         numer.append(i)
#     data ={'Numeracja':numer,
#            'Adresy': urls_to_scrap()}
#
#     df = pd.DataFrame(data)
#     print(df)
#     return df
#
# # if __name__ == '__main__':
#     # create_df()