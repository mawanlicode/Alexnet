import os
import pdb
WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,ReLU,Dropout,Softmax, ConvLSTM2D,Flatten,AveragePooling2D
from keras.layers import LSTM
from keras.layers.core import Lambda
import numpy as np
from keras.layers import Reshape


def CLDNNLikeModel3(weights=None,
             input_shape=[28,28,1], classes=10,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    input = Input(input_shape, name='input')
    x = input
    x = Reshape((28, 28, 1))(x)
    x=  Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='sigmoid',
               kernel_initializer='glorot_uniform',
               name='convx1')(x)
    x = AveragePooling2D(pool_size=(2, 2), border_mode='valid',strides=2, name='maxpoolx1')(x)
    x = Conv2D(filters=12, kernel_size=(5,5), padding='valid', activation='sigmoid',
               kernel_initializer='glorot_uniform',
               name='convx2')(x)
    x =AveragePooling2D(pool_size=(2, 2), border_mode='valid',strides=2, name='maxpoolx2')(x)
    x = Flatten()(x)
    print(x)
    x = Dense(10, activation='sigmoid', name='fcx1')(x)


    """
    # x = Reshape((4, 256, 256, 3))(x)
    x = Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu',
               kernel_initializer='glorot_uniform',
               name='convx{}'.format(1))(x)
    # x = BatchNormalization(name='conv{}-bn'.format(index + 1))(x)
    # x = Dropout(dr)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2, name='maxpoolx{}'.format(1))(x)

    x = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu',
               kernel_initializer='glorot_uniform',
               name='convx{}'.format(2))(x)
    # x = BatchNormalization(name='conv{}-bn'.format(index + 1))(x)
    # x = Dropout(dr)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2, name='maxpoolx{}'.format(2))(x)
    x = Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu',
               kernel_initializer='glorot_uniform',
               name='convx{}'.format(3))(x)
    # x = Dropout(dr)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2, name='maxpoolx{}'.format(3))(x)
    '''
    # LSTM
    # batch_size,64,2
    x = Reshape((128,2,32,1))(x)
    x = ConvLSTM2D(filters=32,kernel_size=(1,1), return_sequences = True)(x)
    x = ConvLSTM2D(filters=32,kernel_size=(1,1))(x)
    '''


    #DNN
    x = Flatten()(x)
    x = Dense(128, activation='selu', name='fcx1')(x)
    x = Dropout(dr)(x)
    x = Dense(512, activation='selu', name='fcx2')(x)
    x = Dropout(dr)(x)
    #x = Dense(128, activation='selu', name='fc2',bias_regularizer=keras.regularizers.l2(0.1))(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)
    """


    # Load weights.
    if weights is not None:
        Model.load_weights(weights)

    model = Model(inputs=input, outputs=x )

    return model

import keras
if __name__ == '__main__':
    model = CLDNNLikeModel3(None,input_shape=(28,28,1),classes=10)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())