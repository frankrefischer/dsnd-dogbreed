# Which dog breed is most similar to a given image?

## Contents

* [Project definition](https://github.com/frankrefischer/dsnd-dogbreed/#project-definition)
* [Analysis](https://github.com/frankrefischer/dsnd-dogbreed/#analysis)
* [Detecting a human face](https://github.com/frankrefischer/dsnd-dogbreed/#detecting-a-human-face)
* [Detecting a dog](https://github.com/frankrefischer/dsnd-dogbreed/#detecting-a-dog)
* [Determining the most resembling dog breed](https://github.com/frankrefischer/dsnd-dogbreed/#determining-the-most-resembling-dog-breed)
* [Conclusion](https://github.com/frankrefischer/dsnd-dogbreed/#conclusion)
* [Libraries used](https://github.com/frankrefischer/dsnd-dogbreed/#libraries-used)
* [How to run the application locally](https://github.com/frankrefischer/dsnd-dogbreed/#how-to-run-the-application-locally)

## Project definition

This is a work for my udacity data science nanodegree.

It is a web app, that allows you to upload an image of a dog or a human and tells you which
dog breed is most resembling one.

That is realized by using a pretrained convolutional network with transfer learning.

Additionally for every image will be detected if it contains a dog or a human face.

### Metric for face detection

To test the quality of the face detection method, we chose 100 random images from the of human face images and 100 random images from the set of dog images.

From the subset with 100 human faces classifier detected in every image a human face. Thats what we expected.

From the subset with 100 dog images the haar cascade classifier detected in 11 images a human face. Here we would have wished, that none was detected, but a rate of 11% is sufficient for our purposes.

### Metric for dog detection

To test the quality of the dog detection method, we used the same procedure as for the human face detection.

From the subset of with 100 human faces, in no image a dog was detected.

From the subset with 100 dogs, in every image a dog was detected.

Thats perfect.

### Metric for breed prediction

We used accuracy for performance measurement of of the dog breed prediction model.
It fits multiclass classification and its interpretation is quite intuitive: percentage of total images classified correctly.

## Analysis

The training, validation and testing data was provided by udacity.

It contained 8351 images of dogs categorized into 133 dog breeds.

These images were splitted into:
* 6680 images for training
* 835 images for validation
* 836 images for testing 

The distribution of images to dog breed categories was roughly similar in all three sets:

![<Category Distribution](<category_distribution.png>)

The x axis simply lists the indices of the 133 dog breed categories, ranging from 0 to 132. On the y axis you can see, how many percents of the image set belong to a dog breed category with a given index. As you can see the curves for all three sets (training, validation, tests) run roughly parallel. So there are no unbalanced distributions of categories between the three images sets.

## Methodology

The models used in this app were developed in a udacity workspace environment with a gpu.

### Preprocessing steps

The Keras CNN with Tensorflow backend expects a tensor with the shape:

>(n_images, n_rows, n_columns, n_channels)

Where:
* n_images is the number of images
* n_rows is the number of pixel rows in each image
* n_columns is the number of pixel columns in each image
* n_channels is the number of channels in each image

So each image has to be:
* loaded from disk
* scaled to a size of 224x224 pixels
* be converted to a tensor of the required shape

This was done by the following code:

```python
from keras.preprocessing import image
import numpy as np

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
```

### Detecting a human face

For detecting human faces open CV's implementation of Haar feature based cascade classifiers was used.


### Detecting a dog

For detecting dogs we used the ResNet50 model with weights trained on the imagenet data set.


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

The second try was to compare 5 pretrained models with transfer learning.
On top of the pretrained model, we added a global average pooling layer and a two dense layers with 133 units in each one.

```
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=self.train.shape[1:]))
        model.add(Dense(133, activation='relu'))
        model.add(Dense(133, activation='softmax'))
```        

The results were:
* VGG16: test accuracy of 69.02%
* VGG19: test accuracy of 69.02% (really!)
* ResNet50: test accuracy of 75.84%
* InceptionV3: test accuracy of 77.75%
* Xception: test accuracy of 80.26%

## Refinement

### Overfitting

Below we compare the history of training and validation accuracy over 20 epochs for all five models.

On the left there is a run without a dropout layer.
On the right there is a run with a 50% dropout layer added.

For all models, overfitting starts early on with the second epoch.

Introducing the dropout layer reduces overfitting, but the validation accuracy does not increase.

![<>](<VGG16_without_dropout.png>)
![<>](<VGG16_with_dropout.png>)

![<>](<VGG19_without_dropout.png>)
![<>](<VGG19_with_dropout.png>)

![<>](<ResNet50_without_dropout.png>)
![<>](<ResNet50_with_dropout.png>)

![<>](<InceptionV3_without_dropout.png>)
![<>](<InceptionV3_with_dropout.png>)

![<>](<Xception_without_dropout.png>)
![<>](<Xception_with_dropout.png>)


## Conclusion

Using pretrained CNNs resulted in much better test accuracy than training a CNN from scratch.

VGG16 and VGG19 were much better than a CNN from scratch.

InceptionV3 and Xception topped them with the best test accuracies.

Xception got the best test accuracy: thats why we used it for the implementation.

For all models evaluated we did not invest much effort in tuning the model.
Adding dropout to reduce overfitting did not work out.

To get further improvements we could:
    * do data augmentation on the image sets
    * do grid search of model parameters

### Some Reflections

With pretrained CNNs its quite easy -- even for a newbie like me -- to get acceptable performance for a toy task like classifying dog breeds.

I was surprised, that adding dropout did not help out to improve performance. The overfitting was reduced, but no performance gain was made.

When in a hurry to get quick results I would start next time with Xception immediately and invest more time in tuning.

Data augmentation is the next step I will take to increase performance.

## Libraries used

The following libraries were used:
* flask for the implementation of the web app
* keras and tensorflow for the usage of the prediction models
* cv2 for the usage of the haar cascade classifier
* and: numpy, json, werkzeug

## How to run the application locally

```
cd dogbreedapp
mkdir uploads
python dogbreedapp.py
```

Then open in browser: http://0.0.0.0:3001


