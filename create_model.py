from params import FLAGS 
import params
import utils
import string
import cv2
import os
import numpy as np
from sklearn.utils import shuffle 


#-----------------------load data and shufffle data------------------------###
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_train, y_train = shuffle(X_train, y_train)

##-------------------------------build model--------------------------------##
from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
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
    
    model.add(Conv2D(24,kernel_size=5,strides=(2,2),padding='same', activation='relu'))
    
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
    
    model.add(Conv2D(36,kernel_size=5,strides=(2,2),padding='same', activation='relu'))
#    
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
    
    model.add(Conv2D(48,kernel_size=5,strides=(2,2),padding='same', activation='relu'))
    
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
    
    model.add(Conv2D(64,kernel_size=3,strides=(1,1),padding='same', activation='relu'))
    
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
    
    model.add(Conv2D(64,kernel_size=3,strides=(1,1),padding='same', activation='relu'))
    
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
    
    model.add(Flatten())
    
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))  # need delete
    model.add(Dense(1, activation=None))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss="mse")
    return model


model = mj_model()  
model.fit(x=X_train, y=y_train, batch_size=64, epochs=20, verbose=1, 
          callbacks=None, validation_split=0.0, validation_data=(X_val,y_val), 
          shuffle=True, class_weight=None, sample_weight=None, 
          initial_epoch=0, steps_per_epoch=None, validation_steps=None)

json_string = model.to_json()
open('models/model.json','w').write(json_string) 
model.save_weights('models/model.h5') 

lost=model.evaluate(x=X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None, steps=None)
print ('test loss: ', lost)












