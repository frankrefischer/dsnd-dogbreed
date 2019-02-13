import cv2

def classify(filepath):
    return {'contains_human_face': contains_human_face(filepath),
            'contains_dog': True,
            'dog_breed': 'mambo jimbo'}

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def contains_human_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


