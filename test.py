from sklearn.neighbors import KNeighborsClassifier


import pickle
import cv2 as cv
import numpy as np
import os
video = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open("data/names.pkl",'rb') as f:
    LABLES = pickle.load(f)
with open("data/faces_data.pkl",'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABLES)

while True:
    ret,frame = video.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h,x:x+w, : ]
        resized_img = cv.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)
        cv.putText(frame,str(output[0]),(x,y-15),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv.rectangle(frame,(x,y),(x + w,y + h),(50,50,255),1)

    cv.imshow("frame",frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv.destroyAllWindows()