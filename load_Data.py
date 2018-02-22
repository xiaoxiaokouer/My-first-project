from params import FLAGS 
import params
import utils
import string
import cv2
import os
import numpy as np


## Training data
epoch_dir = params.data_dir
t_epoch_ids = [1,2,3,4,5,6,7,8]

## validation data
v_epoch_ids = [9]


##--------count all frames for training set and validation set-----------###
n_frames_t = np.zeros((len(t_epoch_ids),),dtype=np.uint16)
k = 0
for i in t_epoch_ids:
    video_path = utils.join_dir(epoch_dir, 'epoch{:0>2}_front.mkv'.format(i))
    n_frames_t[k] = utils.ffmpeg_frame_count(video_path)
    k += 1
    
sum_frames_t = n_frames_t.sum(axis=0)
print ("total frames number----training set:",sum_frames_t)

n_frames_v = np.zeros((len(v_epoch_ids),),dtype=np.uint16)
k = 0
for j in v_epoch_ids:
    video_path = utils.join_dir(epoch_dir, 'epoch{:0>2}_front.mkv'.format(j))
    n_frames_v[k] = utils.ffmpeg_frame_count(video_path)
    k += 1
sum_frames_v = n_frames_v.sum(axis=0)
print ("total frames number----validation set:",sum_frames_v)


#--------------------copy from run.py---------------#
def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0]/3):shape[0]-150, 0:shape[1]]
    ## Resize the image
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    ## Return the image sized as a 4D array
    return np.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h, params.FLAGS.img_c))


##-----------------load all training data and validation data-----------###
X_train = np.zeros((sum_frames_t, params.FLAGS.img_h, params.FLAGS.img_w, 3), dtype=np.uint16)
y_train = np.zeros((sum_frames_t, 1))

k1 = 0
for epoch_id in t_epoch_ids:
    print('---------- loadding video for epoch {} ----------'.format(epoch_id))
    video_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(video_path) 
    frames = utils.ffmpeg_frame_count(video_path)
    labels = utils.get_human_steering(epoch_id)
    for i in range(frames):        
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        img = img_pre_process(img)
        X_train[k1] = img
        y_train[k1] = labels[i]
        k1 += 1  
    cap.release()
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
print('----------complete! total number of training images are : {} ----------'.format(len(X_train)))

X_val = np.zeros((sum_frames_v, params.FLAGS.img_h, params.FLAGS.img_w, 3), dtype=np.uint16)
y_val = np.zeros((sum_frames_v, 1))
k2 = 0
for epoch_id in v_epoch_ids:
    print('---------- loadding video for epoch {} ----------'.format(epoch_id))
    video_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(video_path) 
    frames = utils.ffmpeg_frame_count(video_path)
    labels = utils.get_human_steering(epoch_id)
    for i in range(frames):        
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        img = img_pre_process(img)
        X_val[k2] = img
        y_val[k2] = labels[i]
        k2 += 1 
    cap.release()               
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
print('----------complete! total number of validation images are: {} ----------'.format(len(X_val)))