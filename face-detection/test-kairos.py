import requests
from datetime import datetime

# put your keys in the header
headers = {
    "app_id": "8e907e08",
    "app_key": "05516e9a5bcc8ec91e696c11c56aff64"
}

payload = '{"image":"https://media.kairos.com/liz.jpg",  "gallery_name":"MyGallery","subject_id":"Elizabeth"}'

url = "http://api.kairos.com/verify"

# make request

print(datetime.now())

r = requests.post(url, data=payload, headers=headers)
print(datetime.now())
print(r.content)