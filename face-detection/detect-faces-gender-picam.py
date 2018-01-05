import numpy as np
import cv2
from traindata import traindata
from datetime import datetime
import os
import utils as utils
from picamera.array import PiRGBArray
from picamera import PiCamera



#Uncomment to enable lights on PI
if utils.getEnvironmentValueByKey('ENVIRONMENT_TYPE')=='RASPBERRYPI':
  import showLightsOnPI as showPILights


#train data if flag enabled or else load file from trained file
if utils.getEnvironmentValueByKey('TRAIN_GENDER_CLASSFICATION_DATA')=='TRUE':
  traindata.trainGenderClassficationData('images/male/*.*',1,'images/female/*.*',2,50,50)


#handling multiple versions of OpenCV
if cv2.__version__ > "3.1.0":
  faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
  faceRecognizer.read("data/face_recognizer_gender.yml")
else:
  faceRecognizer = cv2.face.createLBPHFaceRecognizer()
  faceRecognizer.load("data/face_recognizer_gender.yml")
  

#initialize variables
gender_classifier = {1:'Male',2:'Female'}
startTime = datetime.now()
total = datetime.now()-startTime
runningFaceTime = datetime.now()-startTime

counter = 0
facesFound = 'false'
faceTime = 0
minute = 0
faceCounter = 0
totalMaleFaces = 0
totalFemaleFaces = 0
prevMinute = datetime.now().minute


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.2)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  # grab the raw NumPy array representing the image, then initialize the timestamp
  # and occupied/unoccupied text
  img = frame.array

  #img = cv2.resize(img, (0,0), fx=0.5,fy=0.5)
  img = cv2.resize(img, (640,360))
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  #use opencv default haar cascde file for face detection
  path = "data/haarcascade_frontalface_default.xml"

  face_cascade = cv2.CascadeClassifier(path)  
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.30, minNeighbors=5, minSize=(20,20))


  maleFacesCount = 0
  femaleFacesCount = 0
  maleFacesFoundinFrame = 0
  femaleFacesFoundinFrame = 0

  if faceTime!=0 and (datetime.now().minute>prevMinute or (datetime.now().minute==0 and prevMinute==59)):
   utils.recordStatistics(faceTime, round(runningFaceTime.total_seconds()), totalMaleFaces,totalFemaleFaces)
   prevMinute = datetime.now().minute
   runningFaceTime = datetime.now()-datetime.now()



  for (x, y, w, h) in faces:


    #calculate face time
    if facesFound == 'false':
      faceTime = datetime.now()
      minute = faceTime.minute
      print('setting ',faceTime)
      facesFound = 'true'
      faceCounter = len(faces)   
    elif datetime.now().minute>minute:
      minuteFaceTime =  (datetime.now() - faceTime) 
      print('Face time spent in last 1 minute: ', runningFaceTime)
      minute = datetime.now().minute
      faceCounterNow = len(faces)
      
      #ensure that face counter only updated if more faces come into frame than before
      if faceCounterNow > faceCounter:
        faceCounter = faceCounterNow
      #set total male faces
      if maleFacesCount >  totalMaleFaces:
        totalMaleFaces = maleFacesCount
      #set total female faces
      if femaleFacesCount > totalFemaleFaces:
        totalFemaleFaces = femaleFacesCount

      #calculate and record face time in analytics file
      minuteFaceTime =  round(runningFaceTime.total_seconds())
      #utils.recordStatistics(faceTime, minuteFaceTime, totalMaleFaces,totalFemaleFaces)
      runningFaceTime = datetime.now()-datetime.now()



    crop_img = img[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (50,50))
    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    #predict gender from faces found in frame
    #compatibility with opencv version
    if cv2.__version__ > "3.1.0":
      result = cv2.face.StandardCollector_create()
      faceRecognizer.predict_collect(crop_img,result)
      predictedLabel = result.getMinLabel()
      conf = result.getMinDist()
    else:
      result = cv2.face.MinDistancePredictCollector()
      faceRecognizer.predict(crop_img,result,0)
      predictedLabel = result.getLabel()
      conf = result.getDist()

    #increment
    if predictedLabel==1:
      maleFacesCount +=1
      maleFacesFoundinFrame = 1
    elif predictedLabel==2:
      femaleFacesCount +=1
      femaleFacesFoundinFrame = 1

    #print('Predicted Label: {} Confidence: {}', gender_classifier.get(predictedLabel),conf)
   # print('Confidence: ',conf)

    cv2.putText(img, gender_classifier.get(predictedLabel), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
    cv2.imshow('cropeed',crop_img)
    cv2.imshow("Image",img)

  #enable LED lights on PI
  if utils.getEnvironmentValueByKey('ENVIRONMENT_TYPE')=='RASPBERRYPI':
    showPILights.showGenderLights(maleFacesCount,femaleFacesCount)

  #stream data to display sources
  utils.streamFacesData(maleFacesFoundinFrame,femaleFacesFoundinFrame)

  #if len(faces) < 1 and facesFound=='true' and round((datetime.now()-runningFaceTime).total_seconds())>0:
  if len(faces) < 1 and facesFound=='true':
    facesFound = 'false'
    newTime = datetime.now()
    total =  total + (newTime - faceTime)
    runningFaceTime = runningFaceTime + (newTime - faceTime)

    #if datetime.now().minute>minute:
      #utils.recordStatistics(faceTime, round(runningFaceTime.total_seconds()), totalMaleFaces,totalFemaleFaces)

    print('running counter',total)
  elif len(faces)>0:
    if maleFacesCount >  totalMaleFaces:
      totalMaleFaces = maleFacesCount

    if femaleFacesCount > totalFemaleFaces:
      totalFemaleFaces = femaleFacesCount



  cv2.imshow("Image",img)

  ch = cv2.waitKey(30)
 
  if ch & 0xFF == ord('q'):
 
    endTime = datetime.now()   
    #if len(faces) > 0 and facesFound=='true':
    if faceTime==0:
      faceTime = datetime.now()

    total =  total + (endTime - faceTime)
    runningFaceTime = runningFaceTime + (endTime - faceTime)
    minuteFaceTime =  runningFaceTime.total_seconds()
    utils.recordStatistics(faceTime, round(runningFaceTime.total_seconds()), totalMaleFaces,totalFemaleFaces)    
    print('Total face time on camera: ',total)
    print('Total activity time: ', (endTime-startTime))
    break

cap.release()
cv2.destroyAllWindows()


