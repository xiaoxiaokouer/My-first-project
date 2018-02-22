
import os
#my_path= "/home/mj/7_Machine_learning_P6/Spyder_deeptesla_Project/Spyder_deeptesla/elephant.jpg"
#print ("debug mj :", os.path.isfile(my_path))  #mj
#print ("my_path:",my_path)  #mj

#### opencv video to images ############

from matplotlib import pyplot as plt
import cv2
import pandas as pd
import utils
import string
#import ffmpeg

for i in range(10):    
    cap = cv2.VideoCapture('./epochs/epoch01_front.mkv')
    ret, img = cap.read()
    my_path= "/home/mj/7_Machine_learning_P6/Spyder_deeptesla_Project/Spyder_deeptesla/output/img{}.png".format(i)
    cv2.imwrite(my_path, img)
    cap.release()

#cv2.imshow('image',img)
#cv2.waitKey(0)

# check the image shape
#print ('image shape:', img.shape)  

#### opencv images to video ############
#out_dir = os.path.abspath('./output')
out_path = "/home/mj/7_Machine_learning_P6/Spyder_deeptesla_Project/Spyder_deeptesla/output/mj_images.mkv"
vid_size = (1280, 720)

#fourcc = cv2.VideoWriter_fourcc(*'x264')
#fourcc = cv2.VideoWriter_fourcc(*'I420') # mkv ok
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #


vw = cv2.VideoWriter(out_path, fourcc, 30, vid_size)
for i in range(10):  
#    frame = cv2.imread('img{}.png'.format(i))  
    frame = cv2.imread('/home/mj/7_Machine_learning_P6/Spyder_deeptesla_Project/Spyder_deeptesla/output/img{}.png'.format(i)) 
    print ("frame is:",frame)    
    vw.write(frame)
    print("write-------write------write")
    
vw.release()

print("end-------end------end")

