from keras.models import load_model
import numpy as np
import h5py
from keras import layers
import struct


model = load_model('norm_cldnn-matlab1.wts.h5')

weights = np.array(model.get_weights())

weight0 = weights[0]
weight1 = weights[1]
weight2 = weights[2]
weight3 = weights[3]
weight4 = weights[4]
weight5 = weights[5]
#weight6 = weights[6]


i = [0,1,2,3,4,5]
j = [0,1,2,3,4]
j=0
bytes=weights
#binfile=open("w.txt",'a')

bytes = weights[0]

#bit=bytes[:][1][:][:]
#print(bit)

bytes.tofile("w210.bin")
    #j+=1
'''
while j < 5:
    w=weights[0][0][j]
    w_name = str(i)+'w'+'.txt'
    np.savetxt(w_name,w,fmt='%f')
    w_name = str(i) + 'w' + '.bin'
    w.tofile(w_name)
    j+=1
'''