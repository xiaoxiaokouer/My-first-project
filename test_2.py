import numpy as np

a = 1.00000005
b= float('%.1f'%a)
print (b)

x_train = np.zeros((10,2,1,3), dtype=np.uint8)
print(x_train)
print ("the length is:",len(x_train))