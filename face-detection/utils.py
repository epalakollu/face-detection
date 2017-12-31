from os.path import join, dirname
from dotenv import load_dotenv
import os



dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

def getEnvironmentValueByKey(KEY):
  return os.getenv(KEY)

def recordStatistics(facetime, total, maleFacesCount,femaleFacesCount):
  statsFile = open("analytics.txt", "a")
  string = str(facetime.date())+' '+str(facetime.hour)+':'+str(facetime.minute)+','+str(total)+','+str(maleFacesCount)+','+str(femaleFacesCount)

  statsFile.write(string+'\n')
  statsFile.close()