from keras.models import load_model
import numpy as np
import h5py
from keras import layers
import struct

t = 10001
f = h5py.File("image_data.h5", 'r')
X = f['X'][:, :, :]  # ndarray(2555904*1024*2)
Y = f['Y'][:]
x=X[t]
y=Y[t]
x.astype("float32")
for X in X:
    #j=0
    #while j < 28:
        w_name = '10001'+'.txt'
        np.savetxt(w_name,x,fmt='%f')
        w_name = '10001' + '.bin'
        x.astype('float32').tofile(w_name)
       # j+=1