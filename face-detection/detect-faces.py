import numpy as np
import cv2
from PIL import Image
import glob

cap = cv2.VideoCapture(0)

eye_path = "haarcascade_eye.xml"

image_list = []
label_list = []
count = 0


for filename in glob.glob('images/female/*.png')+glob.glob('images/female/*.jpg')+glob.glob('images/female/*.jpeg'):
    im=Image.open(filename).convert('L')

    imageNp=np.array(im,'uint8')

    if(len(imageNp.shape)>=3):  
        imageNp = cv2.cvtColor(imageNp,cv2.COLOR_BGR2GRAY)
    
    imageNp = cv2.resize(imageNp, (50,50))
    print('imageNp: ',str(len(imageNp.shape)))
    image_list.append(imageNp)
    class_label = 1
    print(class_label)
    label_list.append(class_label)


for filename in glob.glob('images/male/*.png')+glob.glob('images/male/*.jpg')+glob.glob('images/male/*.jpeg'):
    im=Image.open(filename).convert('L')

    imageNp=np.array(im,'uint8')

    print(filename)
    if(len(imageNp.shape)>=3):
       imageNp = cv2.cvtColor(imageNp,cv2.COLOR_BGR2GRAY)
    imageNp = cv2.resize(imageNp, (50,50))

    image_list.append(imageNp)
    class_label = 2
    print(class_label)
    label_list.append(class_label)

    
labels = np.array(label_list)

print(labels)
#big_image=np.concatenate(image_list[0:len(image_list)-1],axis=1)

#cv2.imshow('image',big_image)

fisher_faces = cv2.face.createLBPHFaceRecognizer()
#fisher_faces.train(image_list,labels)
#fisher_faces.save("trainedmodels.yml")

fisher_faces.load('face_recognizer_gender.yml')

while(True):
  ret, img = cap.read()

  img = cv2.resize(img, (0,0), fx=0.5,fy=0.5)

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  path = "haarcascade_frontalface_default.xml"

  face_cascade = cv2.CascadeClassifier(path)

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=3, minSize=(40,40))
  print('faces',faces)

  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    crop_img = img[y:y+h, x:x+w]

    crop_img = cv2.resize(crop_img, (50,50))
    print('cropped image shape: ',crop_img.shape)
    predictedLabel = -1
    confidence = 0

    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

    predictedLabel = fisher_faces.predict(crop_img)

    print('Predicted Label: ',predictedLabel)
    print('Confidence: ',confidence)

    if predictedLabel==2:
      strPredicted = 'Male'
    elif predictedLabel==1:
      strPredicted = 'Female'
    else:
      strPredicted = 'Not Found'

    cv2.putText(img, strPredicted, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
    

    ''' eye_cascade = cv2.CascadeClassifier(eye_path)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.02,minNeighbors=20,minSize=(10,10))
    print(len(eyes))

    for (x, y, w, h) in eyes:
      xc = (x + x+w)/2
      yc = (y + y+h)/2
      radius = w/2
      cv2.circle(img, (int(xc),int(yc)), int(radius), (255,0,0), 2)
    '''
    cv2.imshow('cropeed',crop_img)

  cv2.imshow("Image",img)
    


 # cv2.circle(img, point, radius, color, line_width)
  #cv2.imshow("Frame",img)

  ch = cv2.waitKey(30)
  if ch & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()