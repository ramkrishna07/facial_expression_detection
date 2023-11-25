# Importing Libraries:These lines import necessary libraries for handling images, machine learning models, computer vision tasks, and numerical operations.
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
# User Input for File Paths:Prompts the user to input paths for the cascade classifier file and the trained model file.
cascadepath=input("Enter the cascade path=")
modelpath=input("Enter the model path=")
# Loading Cascade Classifier and Keras Model:Loads the Cascade Classifier and the trained Keras model for facial expression recognition using the provided paths.
face_classifier = cv2.CascadeClassifier(cascadepath)
classifier =load_model(modelpath)
# Emotion Labels:Defines a list of emotions corresponding to the output classes of the model.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
# User Input for Image Path: Takes user input for the image file path and reads the image using OpenCV 
image_path=input("Enter the image path=")
frame = cv2.imread(image_path)


# Face Detection and Emotion Recognition:Converts the loaded image to grayscale and uses the Cascade Classifier to detect faces within the image.
labels = []
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray,1.3,5)

# Processing Detected Faces:Loops through each detected face, extracts the region of interest (ROI), preprocesses it for the model, makes a prediction for the emotion, and annotates the image with the predicted emotion label.

for (x,y,w,h) in faces:
        # Extracts face ROI
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        # Processes the face ROI for prediction
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            # Predicts the emotion
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

# Display and Save Result:Displays the annotated image with detected emotions, saves the resulting image with the label as its filename, and closes the OpenCV windows.
cv2.imshow('Emotion Detector',frame)
cv2.imwrite(f"{label}.jpg",frame)


cv2.destroyAllWindows()