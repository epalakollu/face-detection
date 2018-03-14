import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import cv2
import numpy as np

KEY = 'f464bbf19594453392415e3e288dacfb'  # Replace with a valid subscription key (keeping the quotes in place).
CF.Key.set(KEY)
# If you need to, you can change your base API url with:
#CF.BaseUrl.set('https://westcentralus.api.cognitive.microsoft.com/face/v1.0/')

BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)


def detectFaces(img):

    img_str = cv2.imencode('.jpg', img)[1].tostring()

    faces = CF.face.detect(BytesIO(img_str))
    return faces

def recognizeFaces(personId1, personId2):

    result = CF.face.verify(personId1,personId2)
    return result

def createFacelist(listName):
    response = CF.face_list.create(listName)
    print(response)

def addFaceToList(listName,face):
    img_str = cv2.imencode('.jpg', np.array(face))[1].tostring()
    response = CF.face_list.add_face(img_str,listName)
    print(response) 

def getFaceListDetails(listName):
    response = CF.face_list.get(listName)
    print(response)





