# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:22:25 2017

@author: Doug_Pedersen
"""

import os
import csv
import random
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
#import keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D


samples = []
#with open('./driving_log.csv') as csvfile:
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # Don't store header:
        if line[0] != "center":
            samples.append(line)
with open('driving_log_bridge.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # Don't store header:
        if line[0] != "center":
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=25):  
    num_samples = len(samples)
    print("num_samples = %d " % num_samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../IMG/'+batch_sample[0].split('/')[-1]
                # print(batch_sample[3])
                if os.path.isfile(name):
                    center_image = cv2.imread(name)
                else:
                    print(name)

                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=25)
validation_generator = generator(validation_samples, batch_size=25)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# and don't model sky and hood:
model.add(Cropping2D(cropping=((75, 25), (0, 0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x / 127.5 - 1.))

# Model from https://github.com/commaai/research/blob/master/train_steering_model.py
# Recommended from project resources.
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#model = load_model("model.h5")

# train the model
model.fit_generator(train_generator, 
                    samples_per_epoch = len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), 
                    nb_epoch= 3 )  # 3

model.save("model.h5")



