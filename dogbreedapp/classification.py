import numpy as np
import cv2
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image                  
from keras.applications.resnet50 import ResNet50
 
def classify(filepath):
    return {'contains_human_face': contains_human_face(filepath),
            'contains_dog': contains_dog(filepath),
            'dog_breed': 'mambo jimbo'}

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
    ResNet50_model = ResNet50(weights='imagenet')
    ResNet50_model._make_predict_function()

load_ResNet50_model()

def contains_dog(img_path):

    def ResNet50_predict_labels(img_path):
        # returns prediction vector for image located at img_path
        input_tensor = load_image_tensor_from_image_file(img_path)
        img = preprocess_input(input_tensor)
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



