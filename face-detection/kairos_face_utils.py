import requests
from datetime import datetime
import json
import os
import urllib3

# put your keys in the header
headers = {
    "app_id": "replace",
    "app_key": "replace with actual"
}




def enrollPhotoGallery(image,title,gallery_name):

  data={}
  data["image"]= image
  data["gallery_name"] = gallery_name
  data["subject_id"] = title

  payload = json.dumps(data)


  url = "http://api.kairos.com/enroll"

  print(datetime.now())
  
  r = requests.post(url, data=payload, headers=headers)

  print(datetime.now())
  print(r.content)


def verifyImage(image,title,gallery_name):
  data={}
  data["image"]= image
  data["gallery_name"] = gallery_name
  data["subject_id"] = title

  payload = json.dumps(data)



  url = "http://api.kairos.com/verify"

  print(datetime.now())
  
  r = requests.post(url, data=payload, headers=headers)

  print(datetime.now())
  print(r.content)

def recognizeFace(image,gallery_name):
  data={}
  data["image"]= image
  data["gallery_name"] = gallery_name
  data["Content-Type"] = 'application/json'

  os.environ['NO_PROXY'] = 'api.kairos.com'

  proxies = {
    "http": None
  }

  payload = json.dumps(data)

  print(payload)


  url = "http://api.kairos.com/recognize"

  print(datetime.now())

  #http = urllib3.PoolManager()

  #r = http.request('POST', url,
                # headers=headers, body= payload)
  
  r = requests.post(url, data=payload, headers=headers, proxies=proxies, stream=False)

  print(datetime.now())
  print(r.content)
  #print(r.headers)

