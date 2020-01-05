# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 21:31:28 2019

@author: boonping
"""

import cv2
import os,sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

'''
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
'''
from sklearn.model_selection import train_test_split


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model,save_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D,Conv1D,AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda,concatenate,Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt
#savetxt('data.csv', data, delimiter=',')
#data = loadtxt('data.csv', delimiter=',')
import gc
from skimage.transform import resize

'''
os.popen("copy img1_representation_0_*.csv /b  img1_merged_representation_0.csv ")        
os.popen("copy img2_representation_0_*.csv /b  img2_merged_representation_0.csv ")        
os.popen("copy img0_representation_0_*.csv /b  img0_merged_representation_0.csv ")     
os.popen("copy img3_representation_0_*.csv /b  img3_merged_representation_0.csv ")        

sleep(0.5)

X1 = loadtxt('img1_merged_representation_0.csv', delimiter=',')
X2 = loadtxt('img2_merged_representation_0.csv', delimiter=',')
X0 = loadtxt('img0_merged_representation_0.csv', delimiter=',')
X3 = loadtxt('img3_merged_representation_0.csv', delimiter=',')
#X=np.array([])

Y0=np.append( np.ones(X0.shape[0]),np.zeros(X1.shape[0]),axis=0)
Y0=np.append( Y0,np.zeros(X2.shape[0]),axis=0)
Y0=np.append( Y0,np.zeros(X3.shape[0]),axis=0)

Y1=np.append( np.zeros(X0.shape[0]),np.ones(X1.shape[0]),axis=0)
Y1=np.append( Y1,np.zeros(X2.shape[0]),axis=0)
Y1=np.append( Y1,np.zeros(X3.shape[0]),axis=0)

Y2=np.append( np.zeros(X0.shape[0]),np.zeros(X1.shape[0]),axis=0)
Y2=np.append( Y2,np.ones(X2.shape[0]),axis=0)
Y2=np.append( Y2,np.zeros(X3.shape[0]),axis=0)



#Y=np.hsplit(Y,3)
#X=np.append(X,X0,axis=0)
X=X0
X=np.append(X,X1,axis=0)
X=np.append(X,X2,axis=0)
X=np.append(X,X3,axis=0)
#print(X.shape)
#print(Y.shape)
del X2
del X3
gc.collect()
'''

X0 = loadtxt('img0_merged_representation_0.csv', delimiter=',')
X1 = loadtxt('img1_merged_representation_0.csv', delimiter=',')
X2 = loadtxt('img2_merged_representation_0.csv', delimiter=',')
X3 = loadtxt('img4_merged_representation_0.csv', delimiter=',')

X4 = loadtxt('img0_merged_representation_1.csv', delimiter=',')
X5 = loadtxt('img1_merged_representation_1.csv', delimiter=',')
X6 = loadtxt('img2_merged_representation_1.csv', delimiter=',')
X7 = loadtxt('img4_merged_representation_1.csv', delimiter=',')

Y0=np.append( np.ones(X0.shape[0]), np.ones(X4.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X1.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X5.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X2.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X6.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X3.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X7.shape[0]),axis=0)

Y1=np.append( np.zeros(X0.shape[0]), np.zeros(X4.shape[0]),axis=0)
Y1=np.append( Y1, np.ones(X1.shape[0]),axis=0)
Y1=np.append( Y1, np.ones(X5.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X2.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X6.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X3.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X7.shape[0]),axis=0)

Y2=np.append( np.zeros(X0.shape[0]), np.zeros(X4.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X1.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X5.shape[0]),axis=0)
Y2=np.append( Y2, np.ones(X2.shape[0]),axis=0)
Y2=np.append( Y2, np.ones(X6.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X3.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X7.shape[0]),axis=0)

Y3=np.append( np.zeros(X0.shape[0]), np.zeros(X4.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X1.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X5.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X2.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X6.shape[0]),axis=0)
Y3=np.append( Y3, np.ones(X3.shape[0]),axis=0)
Y3=np.append( Y3, np.ones(X7.shape[0]),axis=0)

X=X0
X=np.append(X,X4,axis=0)
X=np.append(X,X1,axis=0)
X=np.append(X,X5,axis=0)
X=np.append(X,X2,axis=0)
X=np.append(X,X6,axis=0)
X=np.append(X,X3,axis=0)
X=np.append(X,X7,axis=0)


'''
#for i in range(1,10,1):
for i in range(1,4,1):
    
    os.popen("copy img0_representation_%i_*.csv   img0_merged_representation_%i.csv " % (i,i) )            
    os.popen("copy img1_representation_%i_*.csv   img1_merged_representation_%i.csv " % (i,i) )        
    os.popen("copy img2_representation_%i_*.csv   img2_merged_representation_%i.csv " % (i,i) )        
    
    X1 = loadtxt('img1_merged_representation_%i.csv' % i, delimiter=',')
    X2 = loadtxt('img2_merged_representation_%i.csv' % i, delimiter=',')
    X0 = loadtxt('img0_merged_representation_%i.csv' % i, delimiter=',')
    Y=np.append( Y, np.ones(X0.shape[0])*(i*3),axis=0)
    Y=np.append( Y, np.ones(X1.shape[0])*(i*3+1),axis=0)
    Y=np.append( Y, np.ones(X2.shape[0])*(i*3+2),axis=0)
    X=np.append(X,X0,axis=0)
    X=np.append(X,X1,axis=0)
    X=np.append(X,X2,axis=0)
    print(X.shape)
    print(Y.shape)
'''
#raise

#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1)
X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3 = train_test_split(X,Y0,Y1,Y2,Y3,test_size = 0.1)
#X_train,X_test,Y_train1,Y_test1 = train_test_split(X,Y1,test_size = 0.1)
#X_train,X_test,Y_train2,Y_test2 = train_test_split(X,Y2,test_size = 0.1)
#X_train=X_train.reshape(X_train.shape[0],1,128)
#X_test=X_test.reshape(X_test.shape[0],1,128)
#X_train=np.hsplit(X_train,8)
#X_test=np.hsplit(X_test,8)
print(X_train)

'''
a=np.array([[2,3,5,7],[1,4,8,9]])
print(a.reshape(a.shape[0],2,2))
print(np.hsplit(a,2))
raise
'''
def createModel():
    '''
    inputShape=(16,) #128/8
    
    inputs0      = Input(shape=inputShape)
    inputs1      = Input(shape=inputShape)
    inputs2      = Input(shape=inputShape)
    inputs3      = Input(shape=inputShape)
    inputs4      = Input(shape=inputShape)
    inputs5      = Input(shape=inputShape)
    inputs6      = Input(shape=inputShape)
    inputs7      = Input(shape=inputShape)
    
    
    x0 = Dense(8,activation="relu")(inputs0)
    x1 = Dense(8,activation="relu")(inputs1)
    x2 = Dense(8,activation="relu")(inputs2)
    x3 = Dense(8,activation="relu")(inputs3)
    x4 = Dense(8,activation="relu")(inputs4)
    x5 = Dense(8,activation="relu")(inputs5)
    x6 = Dense(8,activation="relu")(inputs6)
    x7 = Dense(8,activation="relu")(inputs7)
    
    
    x0=Reshape((16,1))(inputs0)
    x1=Reshape((16,1))(inputs1)
    x2=Reshape((16,1))(inputs2)
    x3=Reshape((16,1))(inputs3)
    x4=Reshape((16,1))(inputs4)
    x5=Reshape((16,1))(inputs5)
    x6=Reshape((16,1))(inputs6)
    x7=Reshape((16,1))(inputs7)
    
    x0=Conv1D(8,kernel_size=(8,),activation="relu")(x0)
    x1=Conv1D(8,kernel_size=(8,),activation="relu")(x1)
    x2=Conv1D(8,kernel_size=(8,),activation="relu")(x2)
    x3=Conv1D(8,kernel_size=(8,),activation="relu")(x3)
    x4=Conv1D(8,kernel_size=(8,),activation="relu")(x4)
    x5=Conv1D(8,kernel_size=(8,),activation="relu")(x5)
    x6=Conv1D(8,kernel_size=(8,),activation="relu")(x6)
    x7=Conv1D(8,kernel_size=(8,),activation="relu")(x7)
    
    
    x=concatenate([x0,x1,x2,x3,x4,x5,x6,x7])
    
    x = Dense(128,activation="relu")(x)
    x = Dense(64,activation="relu")(x)
    #x= Flatten()(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(20,activation="relu")(x)
    '''
    
    inputShape=(128,)
    inputs      = Input(shape=inputShape)
    #x=Reshape((128,1))(inputs)
    x = Dense(128,activation="relu")(inputs)
    #x=Conv1D(128,kernel_size=(8,),activation="relu",padding="same")(x)
    #x=Conv1D(64,kernel_size=(4,),activation="relu",padding="same")(x)
    #x=AveragePooling1D(4)(x)
    #x = Flatten()(x)
    x = Dense(64,activation="relu")(x)
    x = Dense(32,activation="relu")(x)
    x = Dense(20,activation="relu")(x)

    outputs0 = Dense(20,activation="relu")(x)
    outputs1 = Dense(20,activation="relu")(x)
    outputs2 = Dense(20,activation="relu")(x)
    outputs3 = Dense(20,activation="relu")(x)
    
    outputs0 = Dense(2,activation="softmax")(outputs0)
    outputs1 = Dense(2,activation="softmax")(outputs1)
    outputs2 = Dense(2,activation="softmax")(outputs2)
    outputs3 = Dense(2,activation="softmax")(outputs3)
    
    model       = Model(inputs=inputs,outputs=[outputs0,outputs1,outputs2,outputs3])       
    #model       = Model(inputs=[inputs0,inputs1,inputs2,inputs3,inputs4,inputs5,inputs6,inputs7],outputs=outputs)       
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizers.Adam() ,
                metrics=['accuracy'])

    return model

model=createModel()
model.summary()
modelname="facenet_network"
def lrSchedule(epoch):
    lr  = 5e-3
    if epoch > 195:
        lr  *= 1e-4
    elif epoch > 180:
        lr  *= 1e-3
        
    elif epoch > 160:
        lr  *= 1e-2
        
    elif epoch > 140:
        lr  *= 1e-1
        
    elif epoch > 120:
        lr  *= 2e-1
    elif epoch > 60:
        lr  *= 0.5
        
    print('Learning rate: ', lr)
    
    return lr

#general setting for autoencoder training model
LRScheduler     = LearningRateScheduler(lrSchedule)

                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger,LRScheduler]




# .............................................................................


#Section 7: Training autodecoder. 140 epoch. 
#Learning rate is set higher as it requires more time to converge.
# Fit the model
# This is where the training starts
trDat=X_train
tsDat=X_test


#trLbl=Y_train
trLbl=[Y_train0,Y_train1,Y_train2,Y_train3]
       
#tsLbl=Y_test
#print(trLbl)
#print(tsLbl)
tsLbl=[Y_test0,Y_test1,Y_test2,Y_test3]
#trLbl=np.hsplit(trLbl,3)
#tsLbl=np.hsplit(tsLbl,3)

trLbl[0]       = to_categorical(trLbl[0])
tsLbl[0]       = to_categorical(tsLbl[0])
trLbl[1]       = to_categorical(trLbl[1])
tsLbl[1]       = to_categorical(tsLbl[1])
trLbl[2]       = to_categorical(trLbl[2])
tsLbl[2]       = to_categorical(tsLbl[2])
trLbl[3]       = to_categorical(trLbl[3])
tsLbl[3]       = to_categorical(tsLbl[3])


#2.5 Scaling. 500ohm is the maximum resistance value possible. Change to float type
model.fit(trDat, 
            trLbl, 
            validation_data=(tsDat, tsLbl), 
            epochs=10, 
            batch_size=1,
            callbacks=callbacks_list)
model.save_weights(modelname + ".hdf5")
