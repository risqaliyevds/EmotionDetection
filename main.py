import pickle
import os
from func import *
import cv2

# Load model
classifier = pickle.load(open('emotion_classifier_model.pkl', 'rb'))

# Define categories of val set
category_path_list = os.listdir('eINTERFACE_2021_Image/val')
category_classes = {}
for i in range(len(category_path_list)):
    category_classes[category_path_list[i]] = i

# Open the video stream
cap = cv2.VideoCapture(0)

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = image_preprocessor(frame)

    prediction = predict(classifier, image, category_classes)

    # Display the predictions on the original frame
    cv2.putText(frame, "Prediction: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

