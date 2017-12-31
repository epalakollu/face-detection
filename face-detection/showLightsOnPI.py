import RPi.GPIO as GPIO
import time

def showGenderLights(maleFlag,femaleFlag):
  
  GPIO.setmode(GPIO.BCM) 
  GPIO.setwarnings(False)
  GPIO.setup(18,GPIO.OUT)

  if maleFlag==1:
    print("LED on")
    GPIO.output(18,GPIO.HIGH)
  elif femaleFlag==1:
    GPIO.output(18,GPIO.LOW)

