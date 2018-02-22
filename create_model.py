from params import FLAGS 
import params
import utils
import string
import cv2
import os
import numpy as np


#----------------------------load data--------------------------------###
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

##-------------------------------build model--------------------------------##
from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import model_from_json
#model = VGG16(weights='imagenet', include_top=False)

def mj_model():
    input_shape = (FLAGS.img_h, FLAGS.img_w, FLAGS.img_c)
    model = Sequential()
    model.add(Lambda(lambda x:x/255.-0.5, input_shape=input_shape))
    model.add(Conv2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Conv2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Conv2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
    return model


model = mj_model()  
model.fit(x=X_train, y=y_train, batch_size=128, epochs=5, verbose=1, 
          callbacks=None, validation_split=0.0, validation_data=(X_val,y_val), 
          shuffle=True, class_weight=None, sample_weight=None, 
          initial_epoch=0, steps_per_epoch=None, validation_steps=None)

json_string = model.to_json()
open('models/model.json','w').write(json_string) 
model.save_weights('models/model.h5') 















