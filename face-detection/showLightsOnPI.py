import RPi.GPIO as GPIO
import time

def showGenderLights(maleFlag,femaleFlag):
  
  GPIO.setmode(GPIO.BCM) 
  GPIO.setwarnings(False)
  GPIO.setup(18,GPIO.OUT)
  GPIO.setup(23,GPIO.OUT)
  #light male LED
  if maleFlag>=1:
    print("LED on")
    GPIO.output(18,GPIO.HIGH)
  else:
    GPIO.output(18,GPIO.LOW)

  #light female LED
  if femaleFlag>=1:
    GPIO.output(23,GPIO.HIGH)
  else:
    GPIO.output(23,GPIO.LOW)

  if maleFlag==0 and femaleFlag==0:
    GPIO.output(18,GPIO.LOW)
    GPIO.output(23,GPIO.LOW)
