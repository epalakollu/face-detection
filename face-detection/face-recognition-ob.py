import numpy as np
import cv2
from traindata import traindata
from datetime import datetime
import os
import utils as utils




#train data if flag enabled or else load file from trained file
if utils.getEnvironmentValueByKey('TRAIN_GENDER_CLASSFICATION_DATA')=='TRUE':
  traindata.trainDataForFacialRecognition()

radius=1
neighbors=8
grid_x=8;
grid_y=8;



#handling multiple versions of OpenCV
if cv2.__version__ > "3.1.0":
  faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
  faceRecognizer.read("data/face_recognition_OB.yml")
else:
  faceRecognizer = cv2.face.createLBPHFaceRecognizer(radius,neighbors,grid_x,grid_y,130)
  faceRecognizer.load("data/face_recognition_OB.yml")



#Enable and capture video feed
cap = cv2.VideoCapture(0)

while(True):
  ret, img = cap.read()

  #img = cv2.resize(img, (0,0), fx=0.5,fy=0.5)
  img = cv2.resize(img, (640,360))
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  #use opencv default haar cascde file for face detection
  path = "data/haarcascade_frontalface_default.xml"

  face_cascade = cv2.CascadeClassifier(path)  
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.20, minNeighbors=5, minSize=(20,20))

  for (x, y, w, h) in faces:

    crop_img = img[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (50,50))
    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    if cv2.__version__ > "3.1.0":
      result = cv2.face.StandardCollector_create()
      faceRecognizer.predict_collect(crop_img,result)
      predictedLabel = result.getMinLabel()
      conf = result.getMinDist()
    else:
      result = cv2.face.MinDistancePredictCollector()
      faceRecognizer.predict(crop_img,result,200)
      predictedLabel = result.getLabel()
      conf = result.getDist()

      if conf>160:
        predictedLabel = -1

      print(conf)
      print(predictedLabel)


    cv2.putText(img, str(predictedLabel), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
    cv2.imshow('cropeed',crop_img)
    cv2.imshow("Image",img)




  cv2.imshow("Image",img)

  ch = cv2.waitKey(30)
 
  if ch & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()


