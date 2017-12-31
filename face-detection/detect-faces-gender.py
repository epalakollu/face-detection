import numpy as np
import cv2
from traindata import traindata
from datetime import datetime


def recordStatistics(facetime, total, maleFacesCount,femaleFacesCount):
  statsFile = open("analytics.txt", "a")
  string = str(facetime.date())+' '+str(facetime.hour)+':'+str(facetime.minute)+','+str(total)+','+str(maleFacesCount)+','+str(femaleFacesCount)

  statsFile.write(string+'\n')
  statsFile.close()




cap = cv2.VideoCapture(0)


#recordStatistics(datetime.now(),0,5,3)

#traindata.trainGenderClassficationData('images/male/*.*',1,'images/female/*.*',2,50,50)

faceRecognizer = cv2.face.createLBPHFaceRecognizer()
faceRecognizer.load("face_recognizer_gender.yml")

gender_classifier = {1:'Male',2:'Female'}

counter = 0

startTime = datetime.now()
total = datetime.now()-startTime
runningFaceTime = datetime.now()-startTime

facesFound = 'false'
faceTime = 0
minute = 0
faceCounter = 0
totalMaleFaces = 0
totalFemaleFaces = 0



while(True):
  ret, img = cap.read()

  img = cv2.resize(img, (0,0), fx=0.5,fy=0.5)

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  path = "haarcascade_frontalface_default.xml"

  face_cascade = cv2.CascadeClassifier(path)

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.30, minNeighbors=5, minSize=(40,40))
  #print('faces',faces)

  maleFacesCount = 0
  femaleFacesCount = 0


  for (x, y, w, h) in faces:

    if facesFound == 'false':
      faceTime = datetime.now()
      minute = faceTime.minute
      print('setting ',faceTime)
      facesFound = 'true'
      faceCounter = len(faces)   
    elif datetime.now().minute>minute:
      minuteFaceTime =  (datetime.now() - faceTime) 
      print('Face time spent in last 1 minute: ', minuteFaceTime)
      minute = datetime.now().minute
      faceCounterNow = len(faces)
      
      if faceCounterNow > faceCounter:
        faceCounter = faceCounterNow
        
        if maleFacesCount >  totalMaleFaces:
          totalMaleFaces = maleFacesCount

        if femaleFacesCount > totalFemaleFaces:
          totalFemaleFaces = femaleFacesCount

      minuteFaceTime =  round(runningFaceTime.total_seconds())
      recordStatistics(faceTime, minuteFaceTime, totalMaleFaces,totalFemaleFaces)
      runningFaceTime = datetime.now()-datetime.now()



    crop_img = img[y:y+h, x:x+w]

    #counter += 1
    #cv2.imwrite("testdata/face"+str(counter)+'.jpeg',crop_img)

    crop_img = cv2.resize(crop_img, (50,50))
    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    result = cv2.face.MinDistancePredictCollector()
    faceRecognizer.predict(crop_img,result,0)
    predictedLabel = result.getLabel()
    conf = result.getDist()

    if predictedLabel==1:
      maleFacesCount +=1
    elif predictedLabel==2:
      femaleFacesCount +=1

    #print('Predicted Label: {} Confidence: {}', gender_classifier.get(predictedLabel),conf)
   # print('Confidence: ',conf)

    cv2.putText(img, gender_classifier.get(predictedLabel), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
    cv2.imshow('cropeed',crop_img)

  #if len(faces) < 1 and facesFound=='true' and round((datetime.now()-runningFaceTime).total_seconds())>0:
  if len(faces) < 1 and facesFound=='true':
    facesFound = 'false'
    newTime = datetime.now()
    total =  total + (newTime - faceTime)
    runningFaceTime = runningFaceTime + (newTime - faceTime)
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
    total =  total + (endTime - faceTime)
    runningFaceTime = runningFaceTime + (endTime - faceTime)
    minuteFaceTime =  runningFaceTime.total_seconds()
    recordStatistics(faceTime, round(runningFaceTime.total_seconds()), totalMaleFaces,totalFemaleFaces)    
    print('Total face time on camera: ',total)
    print('Total activity time: ', (endTime-startTime))


    break

cap.release()
cv2.destroyAllWindows()


