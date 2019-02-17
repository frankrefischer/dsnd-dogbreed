# Which dog breed is most similar to a given image?

## Contents

* [Project definition](https://github.com/frankrefischer/dsnd-dogbreed/#project-definition)

## Project definition

This is a work for my udacity data science nanodegree.

It is a web app, that allows you to upload an image of a dog or a human and tells you which
dog breed is most resembling one.

That is realized by using a pretrained convolutional network with transfer learning.

Additionally for every image will be detected if it contains a dog or a human face.

## Analysis

The models used in this app were developed in a udacity workspace environment with a gpu.

The training, validation and testing data was provided by udacity.
It contained 8351 images of dogs categorized into 133 dog breeds.

These images were splitted into:
* 6680 images for training
* 835 images for validation
* 836 images for testing 

For the model detecting a human face a collection of 13233 images of human faces was used.

### Detecting a human face

For detecting human faces open CV's implementation of Haar feature based cascade classifiers was used.

To test the quality of the face detection method, we chose 100 random images from the of human face images and 100 random images from the set of dog images.

From the subset with 100 human faces the haar cascade classifier detected in every image a human face. Thats what we expected.

From the subset with 100 dog images the haar cascade classifier detected in 11 images a human face. Here we would have wished, that none was detected.

### Detecting a dog

For detecting dogs we used the ResNet50 model with weights trained on the imagenet data set.

To test the quality of the dog detection method, we used the same procedure as for the human face detection.

From the subset of with 100 human faces, in no image a dog was detected.

From the subset with 100 dogs, in every image a dog was detected.

Thats perfect.

### Determining the most resembling dog breed

### using a CNN from scratch

```python
model = Sequential()

model.add(Conv2D(16, (2,2), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(133, activation='softmax'))
```



## Conclusion

## Libraries used

## How to run the application locally

## The files

