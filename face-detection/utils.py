from os.path import join, dirname
from dotenv import load_dotenv
import os
from socketIO_client import SocketIO, LoggingNamespace
import json


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


def getEnvironmentValueByKey(KEY):
  return os.getenv(KEY)

def recordStatistics(facetime, total, maleFacesCount,femaleFacesCount):
  statsFile = open("analytics.txt", "a")
  data = {}
  data['date'] = str(facetime.date())+' '+str(facetime.hour)+':'+str(facetime.minute)
  data['total'] = str(total)
  data['maleFacesCount'] = str(maleFacesCount)
  data['femaleFacesCount'] = str(femaleFacesCount)
  jsonData = json.dumps(data)



  string = str(facetime.date())+' '+str(facetime.hour)+':'+str(facetime.minute)+','+str(total)+','+str(maleFacesCount)+','+str(femaleFacesCount)
  
  statsFile.write(string+'\n')
  statsFile.close()
  
  with SocketIO('127.0.0.1', 8000, LoggingNamespace) as socketIO:
    socketIO.emit('statsdata',jsonData)
    print(jsonData)


