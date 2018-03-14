import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import cv2
import kairos_face_utils as kairos
from traindata import traindata
import base64

#traindata.enrollTrainingDataWithKairos()

imageStream= open('./images/face_recognition/img_1.jpg','rb')
image_read = imageStream.read()
image_64_encode = base64.b64encode(image_read).decode('ascii')


kairos.recognizeFace(image_64_encode,"face_recognition_OB")

