import cv2
import numpy as np

def image_preprocessor(img):

    resize_image= cv2.resize(img, (128, 128))
    expand_input = np.expand_dims(resize_image, axis = 0)
    input_data  = np.array(expand_input)
    input_data = input_data/255

    return input_data


def classification_class(dict_classes, prediction_result):
    for key in dict_classes:
        if dict_classes[key] == prediction_result:
            return key


def predict(model, input_data, classes):
    pred = model.predict(input_data)
    result = pred.argmax()
    return classification_class(classes, result)