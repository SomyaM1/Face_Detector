# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
a = cv2.CascadeClassifier(r"file:///C:/Users/b/Downloads/haarcascade_frontalface_default%20(1).xml")
b = cv2.VideoCapture(0)
while True:
    c_rec,d_image = b.read()
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    f = a.detectMultiScale(e, 1.3, 6)
    for (x1,y1,h1,w1) in f:
        cv2.rectangle(d_image, (x1,y1), (x1+w1, y1+h1), (255,0,0))
    cv2.imshow("img",d_image)
    h = cv2.waitKey(40) & 0xff
    if h == 40:
        break
b.release()
cv2.destroyAllWindows()
