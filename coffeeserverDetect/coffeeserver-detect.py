import cv2
import numpy as np

coffeeserver_cascade = cv2.CascadeClassifier('coffeeserver-cascade-12stages.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coffeeserver = coffeeserver_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1, minSize=(150, 150))

    for (x,y,w,h) in coffeeserver:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0.255,255), 2)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
