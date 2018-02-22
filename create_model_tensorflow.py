
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import utils


cap = cv2.VideoCapture('./epochs/epoch01_front.mkv')
ret, img = cap.read()
cap.release()
## Show
#plt.imshow(img)

# check the image shape
print ('image shape:', img.shape)

# check the excel data
wheel_sig = pd.read_csv('./epochs/epoch01_steering.csv')
display(wheel_sig.head())

## Distribution of wheel signal
wheel_sig.wheel.hist(bins=50)

path = './epochs/epoch01_front.mkv'
#count = utils.frame_count(path, method='ffmpeg')
count = utils.ffmpeg_frame_count(path)
print ("frame count is:",count)

