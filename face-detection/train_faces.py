import numpy as np
import cv2
import glob

counter = 0

for filename in glob.glob('images/data/*.jpeg'):

  print(filename)
  img = cv2.imread(filename,1)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  path = "haarcascade_frontalface_default.xml"

  face_cascade = cv2.CascadeClassifier(path)

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
  print(len(faces))

  for (x, y, w, h) in faces:
    crop_img = img[y:y+h, x:x+w]
    counter += 1
    print(filename)
    cv2.imwrite(filename,crop_img)
 