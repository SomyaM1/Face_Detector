# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 19:36:46 2025

@author: admin
"""

import cv2
import sys

# Load the Haar cascade
a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if a.empty():
    print("Error: Haar cascade file not loaded. Check the file path.")
    sys.exit()

# Initialize the webcam
b = cv2.VideoCapture(0)
if not b.isOpened():
    print("Error: Camera not accessible.")
    sys.exit()

while True:
    # Read frame from the webcam
    c_rec, d_image = b.read()
    if not c_rec:
        print("Error: Frame not read from camera.")
        break

    # Convert to grayscale
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = a.detectMultiScale(e, scaleFactor=1.3, minNeighbors=6)
    
    # Draw rectangles around detected faces
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Face Detection", d_image)

    # Break the loop on 'Esc' key press
    h = cv2.waitKey(40) & 0xFF
    if h == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all windows
b.release()
cv2.destroyAllWindows()
