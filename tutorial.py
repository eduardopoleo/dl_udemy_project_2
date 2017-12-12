#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:07:10 2017

@author: eduardopoleo
"""

# Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D # images are 2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten # conver the feature map into the vector to feed the NN
from keras.layers import Dense

# Initializing the CNN

classifier = Sequential()


# Convolution
#args: 
# number of feature detectors ( the more the better usually )
# number of columns per feature detector
# number of rows per feature detector
# input_shape(dim1, dim2, number_of_channels)
classifier.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu')) # tensorflow backend order

# Pooling
# pool_size: size of the subtable that slides over the feature table to choose the max feature
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Train Acc: 0.80, Test Acc: 0.77
# Adding an additional layer to improve Acc results:

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images taken from (https://keras.io/preprocessing/image/)

from keras.preprocessing.image import ImageDataGenerator

# pre proccessing the image to prevent overfitting
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# just need to scale the test set not process it cuz it's used on predictions
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(32, 32), 
        batch_size=32,
        class_mode='binary')

# the dimensions to which all images found will be resized. 
# Default: (256, 256). Do not know if you can go higher
# In general higher better cuz we have more information about the pixels but it's more
# computational intensive.
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

history = classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32, # number of batches to be considered on the training set
        epochs=1) # number of batches to bbe considered on the test set.

# Part 3, Making new prediction

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(32, 32))
test_image = image.img_to_array(test_image) # add the color dimention => 3
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_generator(test_image, steps = 2000/32)



training_set.class_indices
