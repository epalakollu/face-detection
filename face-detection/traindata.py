import numpy as np
import cv2
from PIL import Image
import glob

image_formats = {'jpeg','jpg','png'}


class traindata:

    def get_data_labels(path,gender,height,width):

        images = []
        label_list = []

        print(image_formats)

        for filename in glob.glob(path):

            if filename.__contains__('.') and image_formats.__contains__(filename.split('.')[1]):

                print(filename)
               
                image=Image.open(filename).convert('L')
                imageNp=np.array(image,'uint8')            
                imageNp = cv2.resize(imageNp, (height,width))            
                class_label = gender            

                images.append(imageNp)
                label_list.append(class_label)

        #labels = np.array(label_list)
        return images,label_list

    def trainGenderClassficationData(malePath,maleLabel,femalePath,femaleLabel,height,width):

        images = []
        labels = []

        imagesMale, labelsMale = traindata.get_data_labels(malePath,maleLabel,height,width)

        imagesFemale, labelsFemale = traindata.get_data_labels(femalePath,femaleLabel,height,width)

        images = imagesMale + imagesFemale
        labels = labelsMale + labelsFemale

        print (labels)

        faceRecognizer = cv2.face.createLBPHFaceRecognizer()
        faceRecognizer.train(images,np.array(labels))
        faceRecognizer.save("data/face_recognizer_gender.yml")

    def trainDataForFacialRecognition():
        imageNames = []
        images = []
        labels = []

        imageNames.append('/Users/ekumar/EKProject/projects/facedetection/pi/face-detection/face-detection/images/face_recognition/img_1.jpg')
        labels.append(2)

        imageNames.append('/Users/ekumar/EKProject/projects/facedetection/pi/face-detection/face-detection/images/face_recognition/img_2.jpg')
        labels.append(2)

        imageNames.append('/Users/ekumar/EKProject/projects/facedetection/pi/face-detection/face-detection/images/face_recognition/img_3.jpg')
        labels.append(2)

        imageNames.append('/Users/ekumar/EKProject/projects/facedetection/pi/face-detection/face-detection/images/face_recognition/img_4.jpg')
        labels.append(2)

        imageNames.append('/Users/ekumar/EKProject/projects/facedetection/pi/face-detection/face-detection/images/face_recognition/nick.jpg')
        labels.append(1)

        imageNames.append('/Users/ekumar/EKProject/projects/facedetection/pi/face-detection/face-detection/images/face_recognition/uma_2.jpg')
        labels.append(3)

        for filename in imageNames:

            if filename.__contains__('.') and image_formats.__contains__(filename.split('.')[1]):

                print(filename)
               
                image=Image.open(filename).convert('L')
                imageNp=np.array(image,'uint8')            
                imageNp = cv2.resize(imageNp, (50,50))                  

                images.append(imageNp)

        faceRecognizer = cv2.face.createLBPHFaceRecognizer()
        faceRecognizer.train(images,np.array(labels))
        faceRecognizer.save("data/face_recognition_OB.yml")


