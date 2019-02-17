"""
Functions for classifying an uploaded image and predicting a dog breed.
"""

import json
import numpy as np
import cv2
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D


def classify(img_path):
    """
    Classify an image.

    Input parameters:
        - img_path: path to the image file

    Return value: a dict with the following keys and values
        - contains_human_face: True if image contains a human face, False otherwise.
        - contains_dog: True if image contains a dog, False otherwise.
        - dog_breed: the dog breed most similar to the image.
    """
    return {'contains_human_face': contains_human_face(img_path),
            'contains_dog': contains_dog(img_path),
            'dog_breed': most_similar_dogbreed(img_path)}

face_cascade = cv2.CascadeClassifier('./dogbreedapp/haarcascade_frontalface_alt.xml')

def contains_human_face(img_path):
    """
    Detect a human face in an image.

    Input parameters:
        - img_path: path to the image file.
    
    Return value:
        - True if a human face was detected, False otherwise.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def load_ResNet50_model():
    """
    Load the ResNet50 model for dog detection.
    Setting a global variable named Resnet50_model.
    There is an issue with keras inside flask: https://github.com/keras-team/keras/issues/6462#issuecomment-460747448
    Therefore the call of _make_predict_function() on the model is done here.
    """
    global ResNet50_model
    ResNet50_model = resnet50.ResNet50(weights='imagenet')
    ResNet50_model._make_predict_function()

load_ResNet50_model()

def contains_dog(img_path):
    """
    Detect a dog in an image.

    Input parameters:
        - img_path: path to the image file.
    
    Return value:
        - True if a dog was detected, False otherwise.
    """
    def ResNet50_predict_labels(img_path):
        "use the Resnet50_model to determine the index label of the most probable dog breed."
        input_tensor = load_image_tensor_from_image_file(img_path)
        img = resnet50.preprocess_input(input_tensor)
        prediction = ResNet50_model.predict(img)
        return np.argmax(prediction)

    def load_image_tensor_from_image_file(img_path):
        "loads an image from an image path and transforms it into a tensor"
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    prediction = ResNet50_predict_labels(img_path)

    return ((prediction <= 268) & (prediction >= 151))


def load_xception_model():
    """
    Load the Xception model for dog breed detection.
    Setting a global variable named xception_model.
    There is an issue with keras inside flask: https://github.com/keras-team/keras/issues/6462#issuecomment-460747448
    Therefore the call of _make_predict_function() on the model is done here.
    """
    global xception_model
    xception_model = xception.Xception(weights='imagenet', include_top=False)
    xception_model._make_predict_function()
load_xception_model()

def create_dogbreed_model():
    """
    Creating the model used on top of the xception model.
    Setting a global variable named dogbreed_model.
    There is an issue with keras inside flask: https://github.com/keras-team/keras/issues/6462#issuecomment-460747448
    Therefore the call of _make_predict_function() on the model is done here.
    """
    global dogbreed_model
    dogbreed_model = Sequential()
    dogbreed_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    dogbreed_model.add(Dense(133, activation='softmax'))
    dogbreed_model.load_weights('./dogbreedapp/weights.best.Xception.hdf5')
    dogbreed_model._make_predict_function()
create_dogbreed_model()

def load_json(filepath):
    "utility for loading json file"
    with open(filepath, 'r') as fh:  
        return json.load(fh)

global dogbreed_names
dogbreed_names = load_json('./dogbreedapp/dogbreed_names.json')

def most_similar_dogbreed(img_path):
    """
    Determines the dog breed most resembling to an image.

    Input parameters:
        - img_path: path to an image file

    Return value:
        - name of the dog breed most resembling the image.
    """
    def path_to_tensor(img_path):
        "loads an rgb image an converts it to a tensor."
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def extract_bottleneck_feature(img_path):
        "extracts the bottleneck feature from an image file"
        tensor = path_to_tensor(img_path)
        preprocessed_input = xception.preprocess_input(tensor)
        bottleneck_feature = xception_model.predict(preprocessed_input)
        return bottleneck_feature

    bottleneck_feature = extract_bottleneck_feature(img_path)
    predicted_vector = dogbreed_model.predict(bottleneck_feature)

    return dogbreed_names[np.argmax(predicted_vector)]



