import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37, GPIO.IN)         #Read output from PIR motion sensor
GPIO.setup(12, GPIO.OUT)         #LED output pin

time.sleep(5)

'''while  GPIO.input(37)==1:
	time.sleep(5)
print("ready")
'''
while True:
       i=GPIO.input(37)
       if i==0:                 #When output from motion sensor is LOW
             print("No intruders",i)
             GPIO.output(12, GPIO.LOW)  #Turn OFF LED
             time.sleep(2)
       elif i==1:               #When output from motion sensor is HIGH
             print("Intruder detected",i)
             GPIO.output(12, GPIO.HIGH)  #Turn ON LED
             time.sleep(2)

