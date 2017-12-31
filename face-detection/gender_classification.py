import numpy as np
import cv2
from PIL import Image
import glob


eye_path = "haarcascade_eye.xml"

'''
image_list = []
label_list = []
count = 0

for filename in glob.glob('images/female/*.png')+glob.glob('images/female/*.jpg')+glob.glob('images/female/*.jpeg'):
    im=Image.open(filename)

    imageNp=np.array(im,'uint8')

    if(len(imageNp.shape)>=3):  
        imageNp = cv2.cvtColor(imageNp,cv2.COLOR_BGR2GRAY)
    
    imageNp = cv2.resize(imageNp, (40,40))
    print('imageNp: ',str(len(imageNp.shape)))
    image_list.append(imageNp)
    class_label = 2
    print(class_label)
    label_list.append(class_label)


for filename in glob.glob('images/male/*.png')+glob.glob('images/male/*.jpg')+glob.glob('images/male/*.jpeg'):
    im=Image.open(filename)

    imageNp=np.array(im,'uint8')

    print(filename)
    if(len(imageNp.shape)>=3):
       imageNp = cv2.cvtColor(imageNp,cv2.COLOR_BGR2GRAY)
    imageNp = cv2.resize(imageNp, (40,40))

    image_list.append(imageNp)
    class_label = 1
    print(class_label)
    label_list.append(class_label)

    
labels = np.array(label_list)


fisher_faces = cv2.face.createLBPHFaceRecognizer()
fisher_faces.train(image_list,labels)
fisher_faces.save("trainedmodels.yml")
'''

fisher_faces = cv2.face.createLBPHFaceRecognizer()
fisher_faces.load('face_recognizer_gender.yml')


img = cv2.imread("images/cmto.png",1)

path = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(path)

faces = face_cascade.detectMultiScale(img, scaleFactor=1.10, minNeighbors=5, minSize=(20,20))
print('faces',faces)
counter = 0
for (x, y, w, h) in faces:


    counter += 1

    crop_img = img[y:y+h, x:x+w]

    crop_img = cv2.resize(crop_img, (40,40))
    print('cropped image shape: ',crop_img.shape)
    predictedLabel = -1
    confidence = 0

    cv2.imwrite("face"+str(counter)+'.jpeg',crop_img)

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

    result = cv2.face.MinDistancePredictCollector()
    fisher_faces.predict(crop_img,result,0)
    predictedLabel = result.getLabel()
    conf = result.getDist()


    if predictedLabel==1:
      strPredicted = 'Male'
    elif predictedLabel==2:
      strPredicted = 'Female'
    else:
      strPredicted = 'Not Found'

    print("{} identified with confidence {}".format(strPredicted, conf))

    cv2.putText(img, strPredicted, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))

    print('Predicted Label: ',predictedLabel)
    print('Confidence: ',confidence)
    

cv2.imshow("Image",img)

ch = cv2.waitKey(0)

cv2.destroyAllWindows()