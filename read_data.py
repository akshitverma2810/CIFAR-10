import tensorflow as tf
import pandas as pd
import numpy as np
import math
import timeit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import platform
import skimage.transform

from subprocess import check_output
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



path_var_cache="G:\\CIFAR10\\cifar-10-python\\variable_Cache\\"

def reshape_to_image(data):
    data=data.reshape((data.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
#    resized_image=[]
#    for i in range(data.shape[0]):
#        print('\r progress = %s'%(i),end='')
#        
#        newImage = skimage.transform.resize(data[i], (224, 224), mode='constant')
#        resized_image.append((newImage*255).astype(np.uint8))
#    
#    return np.array(resized_image)
    return data

def save_batch(data,name):

    data=reshape_to_image(data)
    print(data.shape)
    np.save(path_var_cache+name,data)
    print('Data saved')

def one_hot_encoding(data,length):
    temp=np.zeros((data.shape[0],length))
    
    for i in range(temp.shape[0]):
        temp[i][data[i]]=1
    return temp



img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'G:/CIFAR10/cifar-10-python/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train
    x_test = X_test

#    x_train /= 255
#    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test


# Invoke the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()


x_train=reshape_to_image(x_train)
y_train=one_hot_encoding(y_train,10)

x_val=reshape_to_image(x_val)
y_val=one_hot_encoding(y_val,10)

x_test=reshape_to_image(x_test)
y_test=one_hot_encoding(y_test,10)

np.save(path_var_cache+'x_train',x_train)
np.save(path_var_cache+'y_train',y_train)

np.save(path_var_cache+'x_val',x_val)
np.save(path_var_cache+'y_val',y_val)

np.save(path_var_cache+'x_test',x_test)
np.save(path_var_cache+'y_test',y_test)

#x1,x2,y1,y2=train_test_split(x_train,y_train,test_size=0.5)
#
#x3,x4,y3,y4=train_test_split(x1,y1,test_size=0.5)
#
#x5,x6,y5,y6=train_test_split(x2,y2,test_size=0.5)
#
#
#
#
#save_batch(x3,'x3')
#y3=one_hot_encoding(y3,10)
#np.save(path_var_cache+'y3',y3)
#
#save_batch(x4,'x4')
#y4=one_hot_encoding(y4,10)
#np.save(path_var_cache+'y4',y4)
#
#save_batch(x5,'x5')
#y5=one_hot_encoding(y5,10)
#np.save(path_var_cache+'y5',y5)
#
#save_batch(x6,'x6')
#y6=one_hot_encoding(y6,10)
#np.save(path_var_cache+'y6',y6)
#    
#
#
#
#save_batch(x_val,'x_val')
#y_val=one_hot_encoding(y_val,10)
#np.save(path_var_cache+'y_val',y_val)
#
#
#save_batch(x_test,'x_test')
#y_test=one_hot_encoding(y_test,10)
#np.save(path_var_cache+'y_test',y_test)
























