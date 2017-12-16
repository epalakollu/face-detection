import numpy as np
import cv2

cap = cv2.VideoCapture(0)

eye_path = "haarcascade_eye.xml"


while(True):
  ret, img = cap.read()

  img = cv2.resize(img, (0,0), fx=0.5,fy=0.5)

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  path = "haarcascade_frontalface_default.xml"

  face_cascade = cv2.CascadeClassifier(path)

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.30, minNeighbors=5, minSize=(30,30))
  print(faces)

  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    eye_cascade = cv2.CascadeClassifier(eye_path)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.02,minNeighbors=20,minSize=(10,10))
    print(len(eyes))

    for (x, y, w, h) in eyes:
      xc = (x + x+w)/2
      yc = (y + y+h)/2
      radius = w/2
      cv2.circle(img, (int(xc),int(yc)), int(radius), (255,0,0), 2)

    cv2.imshow("Image",img)


 # cv2.circle(img, point, radius, color, line_width)
  #cv2.imshow("Frame",img)

  ch = cv2.waitKey(30)
  if ch & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()