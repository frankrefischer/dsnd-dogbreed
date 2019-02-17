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

For the prediction of the dog breed we tried several ways.

### using a CNN from scratch

The first try was to build a CNN from scratch.

The model architecture was as following:

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
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

Training this model for 10 epochs with a batch size of 20 resulted in a test accuracy of 4.1%.
Thats pretty bad and not much better than he dumbest method possible: choosing one of the 133 dog breed categories randomly.

### using pretrained CNNs with transfer learning

The second try was to compare 4 pretrained models with transfer learning.
On top of the pretrained model, we added a global average pooling layer and a dense layer with 133 units.
```
model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=self.train.shape[1:]))
        model.add(Dense(133, activation='softmax'))```
```

The results were:
* VGG16: test accuracy of 40.55%
* VGG19: test accuracy of 45.45%
* InceptionV3: test accuracy of 80.02%
* Xception: test accuracy of 84.45%

## Conclusion

Using pretrained CNNs resulted in much better test accuracy than training a CNN from scratch.

VGG16 and VGG19 were much better than a CNN from scratch.

But InceptionV3 and Xception topped them with test accuracies >80%.

Xception got the best test accuracy.

For all models evaluated we did not invest much effort in tuning the model.

To get further improvements we could:
    - do data augmentation on the image sets
    - add dropout
    - do grid search of model parameters
    
## Libraries used

## How to run the application locally

## The files

