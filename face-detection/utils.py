from os.path import join, dirname
from dotenv import load_dotenv
import os
from socketIO_client import SocketIO, LoggingNamespace
import json


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


def getEnvironmentValueByKey(KEY):
  return os.getenv(KEY)


