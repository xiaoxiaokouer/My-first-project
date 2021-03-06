#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:44:29 2018

@author: mj

VGG16 based fine-tuning model, use all layers, only training dense layer.

"""

from params import FLAGS 
import params
import utils
import string
import cv2
import os
import numpy as np
from sklearn.utils import shuffle 
import dill
import keras


#-----------------------load data and shufffle data------------------------###
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_train, y_train = shuffle(X_train, y_train)
#X_val, y_val = shuffle(X_val, y_val)

##-------------------------------build model--------------------------------##
from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import model_from_json


def setup_to_transfer_learn(model, base_model): 
    for layer in base_model.layers:
        layer.trainable = False  
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)    
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

def add_new_last_layer(base_model):
    x = base_model.output
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(1164, activation='relu')(x)
    print ("------------------------------------this is x:",x)
#    x = Dense(100, activation='relu')(x)
#    x = Dense(50, activation='relu')(x)
    predictions = Dense(1, activation=None)(x)
    print ("------------------------------------this is predictions:",predictions)
    model = Model(inputs=base_model.input, outputs=predictions)
    print ("------------------------------------this is model:",model)
    return model  

# loading VGG16 model weights
base_model = VGG16(weights='imagenet', include_top=False)

#s = dill.dumps(base_model.weights)  
#f = dill.loads(s)   
 
model = add_new_last_layer(base_model)
setup_to_transfer_learn(model, base_model)
print ("------------------------------------begin to train:")

model.fit(X_train, y_train, epochs=20, batch_size=32,validation_data=(X_val,y_val))

lost=model.evaluate(x=X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None, steps=None)
print ('test loss: ', lost)

 












