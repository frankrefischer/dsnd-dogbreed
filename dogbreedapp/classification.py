import json
import numpy as np
import cv2
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D


def classify(img_path):
    return {'contains_human_face': contains_human_face(img_path),
            'contains_dog': contains_dog(img_path),
            'dog_breed': most_similar_dogbreed(img_path)}

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('./dogbreedapp/haarcascade_frontalface_alt.xml')

def contains_human_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# https://github.com/keras-team/keras/issues/6462#issuecomment-460747448
def load_ResNet50_model():
    global ResNet50_model
    ResNet50_model = resnet50.ResNet50(weights='imagenet')
    ResNet50_model._make_predict_function()

load_ResNet50_model()

def contains_dog(img_path):

    def ResNet50_predict_labels(img_path):
        # returns prediction vector for image located at img_path
        input_tensor = load_image_tensor_from_image_file(img_path)
        img = resnet50.preprocess_input(input_tensor)
        prediction = ResNet50_model.predict(img)
        return np.argmax(prediction)

    def load_image_tensor_from_image_file(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    prediction = ResNet50_predict_labels(img_path)

    return ((prediction <= 268) & (prediction >= 151))


def load_xception_model():
    global xception_model
    xception_model = xception.Xception(weights='imagenet', include_top=False)
    xception_model._make_predict_function()
load_xception_model()

def create_dogbreed_model():
    global dogbreed_model
    dogbreed_model = Sequential()
    dogbreed_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    dogbreed_model.add(Dense(133, activation='softmax'))
    dogbreed_model.load_weights('./dogbreedapp/weights.best.Xception.hdf5')
    dogbreed_model._make_predict_function()
create_dogbreed_model()

def load_json(filepath):
    with open(filepath, 'r') as fh:  
        return json.load(fh)

global dogbreed_names
dogbreed_names = load_json('./dogbreedapp/dogbreed_names.json')

def most_similar_dogbreed(img_path):
    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def extract_bottleneck_feature(img_path):
        tensor = path_to_tensor(img_path)
        preprocessed_input = xception.preprocess_input(tensor)
        bottleneck_feature = xception_model.predict(preprocessed_input)
        return bottleneck_feature

    # obtain predicted vector
    bottleneck_feature = extract_bottleneck_feature(img_path)
    predicted_vector = dogbreed_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    global dogbreed_names
    return dogbreed_names[np.argmax(predicted_vector)]



