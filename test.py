


from sklearn.neighbors import KNeighborsClassifier
import pickle
import cv2 as cv
import numpy as np
import os
import csv
import time
from datetime import datetime

from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.speak(str1)

video = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open("data/names.pkl",'rb') as f:
    LABLES = pickle.load(f)
with open("data/faces_data.pkl",'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABLES)
COL_NAMES = ['NAME','TIME']


while True:
    ret,frame = video.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h,x:x+w, : ]
        resized_img = cv.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendence/Attendence_" + date + ".csv")
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)

        cv.putText(frame,str(output[0]),(x,y-15),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv.rectangle(frame,(x,y),(x + w,y + h),(50,50,255),1)
        attendence=[str(output[0]),str(timestamp)]
    cv.imshow("frame",frame)
    k = cv.waitKey(1)
    if k == ord("o"):
        speak("Attendance taken")
        time.sleep(5)
        if exist:
            with open("Attendence/Attendence_" + date + ".csv","+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendence)
            csvfile.close()
        else:
            with open("Attendence/Attendence_" + date + ".csv","+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendence)
            csvfile.close()
    if k == ord('q'):
        break
video.release()
cv.destroyAllWindows()


