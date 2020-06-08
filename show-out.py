import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Model
import h5py
from keras import backend as K

def load_data5ways(from_filename="image_data.h5"):
    f = h5py.File("image_data.h5", 'r')

    X = f['X'][: ,: ,: ]  # ndarray(2555904*1024*2)
    Y = f['Y'][:] # ndarray(2M*24)
    # Z = f['Z'][:]  # ndarray(2M*1)
    # amp = f['Amp'][:]
    # pha = f['Pha'][:]

    f.close()
    return X, Y
def get_output_function(model,output_layer_index):
    '''    model: 要保存的模型    output_layer_index：要获取的那一个层的索引    '''
    vector_funcrion=K.function([model.layers[0].input],[model.layers[output_layer_index].output])
    def inner(input_data):
        vector=vector_funcrion([input_data])[0]
        return vector
    return inner

X, Y = load_data5ways("image_data.h5")

#weight_Dense_1 = Model.get_layer("CONV2D").get_weights()

# 第一步：准备输入数据
x= np.expand_dims(X[2000],axis=0)  #[1,28,28,1] 的形状
# 第二步：加载已经训练的模型
Model=keras.models.load_model('norm_cldnn-matlab.wts.h5')
# 第三步：将模型作为一个层，输出第7层的输出
get_feature = get_output_function(Model, 3)  # 该函数的返回值依然是一个函数哦，获取第6层输出

feature = get_feature(x)  # 相当于调用 定义在里面的inner函数

print(feature)
# 第四步：调用新建的“曾模型”的predict方法，得到模型的输出

feature=get_feature(x)  # 相当于调用 定义在里面的inner函数

#feature = reshape（）

print(feature)