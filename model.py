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
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.normalization import BatchNormalization

# amount to correct desired steering angle for the side camera images. >0 steers right
angle_corr = 0.09 
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

def read_image(file_loc):
    if os.path.isfile(file_loc):
        img = cv2.imread(file_loc)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    else:
        # flag something if image not found
        print(file_loc)
        img = []
        
    return img
    
""" Data generator - use to load training data as needed to reduce RAM usage """
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
                # print(batch_sample[3])
                center_image = read_image('../IMG/'+batch_sample[0].split('/')[-1])
                left_image = read_image('../IMG/'+batch_sample[1].split('/')[-1])
                right_image = read_image('../IMG/'+batch_sample[2].split('/')[-1])

                center_angle = float(batch_sample[3])
                
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, 
                               center_angle + angle_corr, 
                               center_angle - angle_corr])

            augmented_images = []
            augmented_angles = []
            for (image, angle) in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(-angle)
            
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=25)
validation_generator = generator(validation_samples, batch_size=25)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# and don't model sky and hood:
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x / 127.5 - 1.))

if 0: 
    # Model from https://github.com/commaai/research/blob/master/train_steering_model.py
    # Recommended from project resources.
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))

# nvidea end to end model
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')

#model = load_model("model.h5")

# train the model
history = model.fit_generator(train_generator, 
                    samples_per_epoch = 2*3*1450, #len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), 
                    nb_epoch= 6 )  # 3

model.save("model.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('nvidia model convergence')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#from keras.utils.visualize_util import plot
#plot(model, to_file='car_model.png',show_shapes=True)
# import pydot ImportError: No module named 'pydot' You must install pydot'
#                      ' and graphviz for `pydotprint` to work.