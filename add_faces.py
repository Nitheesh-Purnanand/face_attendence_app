import cv2 as cv
video = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
while True:
    ret,frame = video.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x + w,y + h),(50,50,255),1)

    cv.imshow("frame",frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv.destroyAllWindows()