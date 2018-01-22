import numpy as np
import cv2
import glob

counter = 0

for filename in glob.glob('/Users/ekumar/EKProject/projects/facedetection/pi/face-detection/face-detection/images/face_recognition/train/*.jpg'):

  print('name',filename)
  img = cv2.imread(filename,1)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  path = "../data/haarcascade_frontalface_default.xml"

  face_cascade = cv2.CascadeClassifier(path)

  cv2.imshow('image',gray)

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.30, minNeighbors=5, minSize=(40,40))
  print(len(faces))

  for (x, y, w, h) in faces:
    crop_img = img[y:y+h, x:x+w]
    counter += 1
    print('uma_'+str(counter)+'.jpg')
    cv2.imwrite('uma_'+str(counter)+'.jpg',crop_img)
 