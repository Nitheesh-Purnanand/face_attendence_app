import cv2 as cv
video = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
faces_data = []
i = 0
while True:
    ret,frame = video.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h,x:x+w, : ]
        resized_img = cv.resize(crop_img,(50,50))
        if len(faces_data) <= 100 and i%10 == 0:
            faces_data.append(resized_img)
        i = i+1
        cv.putText(frame,str(len(faces_data)),(50,50),cv.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        cv.rectangle(frame,(x,y),(x + w,y + h),(50,50,255),1)

    cv.imshow("frame",frame)
    k = cv.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break
video.release()
cv.destroyAllWindows()