import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNet
from tensorflow.keras import layers 
from keras import regularizers
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D
from keras.layers import Activation, Flatten, Dense, Dropout,Conv2D, GaussianNoise,LocallyConnected2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense,GlobalAveragePooling2D,Input, LeakyReLU
from keras.callbacks import LearningRateScheduler
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


path_var_cache="G:\\CIFAR10\\cifar-10-python\\variable_Cache\\"

   
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    

def lr_schedule(epoch):
    lrate = 0.0005
    if epoch > 25:
        lrate = 0.00001
    if epoch > 45:
        lrate = 0.000005
    return lrate    
    
def load_batch(num): 
    x=np.load(path_var_cache+'x'+str(num+3)+'.npy')
    y=np.load(path_var_cache+'y'+str(num+3)+'.npy')
    return x,y

x_train=np.load(path_var_cache+'x_train.npy')
y_train=np.load(path_var_cache+'y_train.npy')

x_train=(x_train.astype(np.float32))/255

x_val=np.load(path_var_cache+'x_val.npy')
x_val=(x_val.astype(np.float32))/255

y_val=np.load(path_var_cache+'y_val.npy')

x_test=np.load(path_var_cache+'x_test.npy')
x_test=(x_test.astype(np.float32))/255

y_test=np.load(path_var_cache+'y_test.npy')


def cnn_transfer(weight_decay):
    
    input_layer=Input(shape=(32,32,3))
    
    upsample=UpSampling2D((2,2))(input_layer)
    
    upsample=Conv2D(3, (5,5), padding='valid', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(upsample)
    upsample=UpSampling2D((2,2))(upsample)
    
    upsample=Conv2D(3, (7,7), padding='valid', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(upsample)
    upsample=UpSampling2D((2,2))(upsample)
    
    upsample=Conv2D(3, (5,5), padding='valid', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(upsample)
    
    upsample=base_model(upsample)
    
    upsample=Conv2D(512, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(upsample)
    
    upsample=MaxPooling2D((2,2))(upsample)
    upsample=Dropout(0.3)(upsample)

    out=Flatten()(upsample)
    
    
    
    
    conv_net=Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(input_layer)
    conv_net=BatchNormalization()(conv_net)
    
    conv_net=Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(conv_net)
    conv_net=BatchNormalization()(conv_net)

    conv_net=MaxPooling2D(pool_size=(2,2))(conv_net)
    conv_net=Dropout(0.2)(conv_net)

    conv_net=Conv2D(96, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(conv_net)
    conv_net=BatchNormalization()(conv_net)
    
    conv_net=Conv2D(96, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(conv_net)
    conv_net=BatchNormalization()(conv_net)

    conv_net=MaxPooling2D(pool_size=(2,2))(conv_net)
    conv_net=Dropout(0.3)(conv_net)

    conv_net=Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(conv_net)
    conv_net=BatchNormalization()(conv_net)
    
    conv_net=Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(conv_net)
    conv_net=BatchNormalization()(conv_net)
    
    conv_net=MaxPooling2D(pool_size=(2,2))(conv_net)
    conv_net=Dropout(0.3)(conv_net)
#    
#    conv_net=Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(conv_net)
#    conv_net=BatchNormalization()(conv_net)

    
    
    
    conv_net=Flatten()(conv_net)
    
    concat=keras.layers.concatenate([conv_net,out])
    
    concat=Dense(1024,kernel_regularizer=regularizers.l2(weight_decay))(concat)
    concat=LeakyReLU(0.3)(concat)
    concat=GaussianNoise(0.05)(concat)
    
    pred=Dense(10,activation='softmax')(concat)
    
    model=keras.Model(inputs=[input_layer],outputs=[pred])
    
    return model
    



    



def cnn_cifar(weight_decay):
 
    
    model = keras.Sequential()
    
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
     
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
     
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    
    model.add(LocallyConnected2D(128, (3,3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(GaussianNoise(0.05))
#    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
 

    return model

base_model=MobileNet(input_shape=(224,224,3),include_top=False,weights='imagenet')

#model=cnn_cifar(1e-4)
#print(model.summary())


#
model=cnn_transfer(1e-4)
print(model.summary())


mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std)
x_test = (x_test-mean)/(std)
x_val = (x_val-mean)/std



optimizer=keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
model.compile(optimizer=optimizer,loss=keras.losses.categorical_crossentropy,metrics=['acc'])



model.load_weights('akshit_verma.h5')
#
#epochs=125  
#datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
#datagen.fit(x_train)
#
#model_info=model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),validation_data=(x_val,y_val),steps_per_epoch=len(x_train)/32, epochs=75,callbacks=[LearningRateScheduler(lr_schedule)])
#
#plot_model_history(model_info)
#
#print('Testing data')    
#acc_test=model.evaluate(x_test,y_test)  
#print('Test ACC = '+str(acc_test))
        
    
 





























































