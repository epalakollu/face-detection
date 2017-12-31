import numpy as np
import cv2
from PIL import Image
import glob
from traindata import traindata


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

#traindata.trainGenderClassficationData('images/male/*.*',1,'images/female/*.*',2,50,50)

fisher_faces = cv2.face.createLBPHFaceRecognizer()
fisher_faces.load('face_recognizer_gender.yml')

image_formats = {'jpeg','jpg','png','JPG','JPEG','PNG'}
path = '/Users/ekumar/Downloads/imdb_crop/94/*.*'

for filename in glob.glob(path):

    print(filename)

    if filename.__contains__('.') and image_formats.__contains__(filename.split('.')[1]):


        img = cv2.imread(filename,1)

        print(img.shape)

        if img.shape[0]>600:
            img = cv2.resize(img,(600,600))

        path = "haarcascade_frontalface_default.xml"

        face_cascade = cv2.CascadeClassifier(path)

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.30, minNeighbors=5, minSize=(40,40))
        print('faces',faces)
        counter = 0
        for (x, y, w, h) in faces:


            counter += 1

            crop_img = img[y:y+h, x:x+w]

            
            print('cropped image shape: ',crop_img.shape)
            predictedLabel = -1
            confidence = 0

            

     
            
            crop_img_bgr = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            crop_img_bgr = cv2.resize(crop_img_bgr, (50,50))

            result = cv2.face.MinDistancePredictCollector()
            fisher_faces.predict(crop_img_bgr,result,0)
            predictedLabel = result.getLabel()
            conf = result.getDist()



            strList = filename.split('/')
            strFileName = strList[len(strList)-1]
            strFileNameList = strFileName.split('.')
            strFileName = strFileNameList[0]+'_'+str(counter)+'.'+strFileNameList[1]
            print(strFileName)

            if predictedLabel==1:
              strPredicted = 'Male'
              cv2.imwrite("male/"+strFileName,crop_img)
            elif predictedLabel==2:
              strPredicted = 'Female'
              cv2.imwrite("female/"+strFileName,crop_img)          
            else:
              strPredicted = 'Not Found'

            print("{} identified with confidence {}".format(strPredicted, conf))

            cv2.putText(img, strPredicted, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))

            print('Predicted Label: ',predictedLabel)
            print('Confidence: ',confidence)
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.imshow("Image",img)

#ch = cv2.waitKey(0)

#cv2.destroyAllWindows()