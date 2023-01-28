import cv2

import numpy as np

from AgeGenderDeepLearningModel import predictAgeGender

# Load the cascade classifier for face detection

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the image and convert it to grayscale

img = cv2.imread("image.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Iterate over the detected faces

for (x, y, w, h) in faces:

    # Draw a rectangle around the face

    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Crop the face from the image

    face_img = img[y:y+h, x:x+w]

    # Predict the age and mood of the person in the face

    age, gender, mood = predictAgeGender(face_img)

    # Add the age and mood labels to the image

    cv2.putText(img, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(img, f"Mood: {mood}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show the image with the detected faces and labels

cv2.imshow("Faces", img)

cv2.waitKey(0)

cv2.destroyAllWindows()

