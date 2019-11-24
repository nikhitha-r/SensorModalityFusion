#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import sys
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input, Add, concatenate
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
import h5py
from matplotlib.pyplot import imshow
import glob
import math
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

from keras.models import load_model


# In[3]:


# Install a Drive FUSE wrapper.
# https://github.com/astrada/google-drive-ocamlfuse
#get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
#get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
#get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
#get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')
#
#
## In[ ]:
#
#
## Generate auth tokens for Colab
#from google.colab import auth
#auth.authenticate_user()
#
#
## In[14]:
#
#
## Generate creds for the Drive FUSE library.
#from oauth2client.client import GoogleCredentials
#creds = GoogleCredentials.get_application_default()
#import getpass
#get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
#vcode = getpass.getpass()
#get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')
#
#
## In[ ]:
#
#
#get_ipython().system('pip install -U -q PyDrive')
#
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials
#
## 1. Authenticate and create the PyDrive client.
#auth.authenticate_user()
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)
#
#
## In[ ]:
#
#
## Create a directory and mount Google Drive using that directory.
#get_ipython().system('mkdir -p drive')
#get_ipython().system('google-drive-ocamlfuse drive')
#
#print ('Files in Drive:')
#get_ipython().system('ls drive/')
#
## Create a file in Drive.
#get_ipython().system('echo "This newly created file will appear in your Drive file list." > drive/created.txt')


# In[ ]:


points =  np.load('./train_points.npy')
labels = np.load('./train_labels.npy')
labels = labels.reshape((7481,24))
classes = np.load('./train_classes.npy')


# In[14]:


intermediate_output = np.load('./intermediate_output.npy')
intermediate_output = np.squeeze(intermediate_output)
print(intermediate_output.shape)


# Model Definition:

# In[11]:



def mat_mul(A, B):
    return tf.matmul(A, B)

# number of points in each sample
num_points = 2048

# number of categories
k = 3

# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

# ------------------------------------ Pointnet Architecture
# input_Transformation_net
input_points = Input(shape=(num_points, 3))
x = Convolution1D(64, 1, activation='relu',
                  input_shape=(num_points, 3))(input_points)
#x = BatchNormalization()(x)
x = Convolution1D(128, 1, activation='relu')(x)
#x = BatchNormalization()(x)
x = Convolution1D(1024, 1, activation='relu')(x)
#x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=num_points)(x)
x = Dense(512, activation='relu')(x)
#x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
#x = BatchNormalization()(x)
x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
input_T = Reshape((3, 3))(x)

# forward net
g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
#g = BatchNormalization()(g)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
#g = BatchNormalization()(g)

# feature transform net
f = Convolution1D(64, 1, activation='relu')(g)
#f = BatchNormalization()(f)
f = Convolution1D(128, 1, activation='relu')(f)
#f = BatchNormalization()(f)
f = Convolution1D(1024, 1, activation='relu')(f)
#f = BatchNormalization()(f)
f = MaxPooling1D(pool_size=num_points)(f)
f = Dense(512, activation='relu')(f)
#f = BatchNormalization()(f)
f = Dense(256, activation='relu')(f)
#f = BatchNormalization()(f)
f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
feature_T = Reshape((64, 64))(f)

# forward net
g = Lambda(mat_mul, arguments={'B': feature_T})(g)
g = Convolution1D(64, 1, activation='relu')(g)
#g = BatchNormalization()(g)
g = Convolution1D(128, 1, activation='relu')(g)
#g = BatchNormalization()(g)
g = Convolution1D(1024, 1, activation='relu')(g)
#g = BatchNormalization()(g)

# global_feature
global_feature = MaxPooling1D(pool_size=num_points)(g)
global_feature = Flatten()(global_feature)
# point_net_cls
#c = Dense(512, activation='relu')(global_feature)
#c = BatchNormalization()(c)
#c = Dropout(rate=0.7)(c)
#c = Dense(256, activation='relu')(c)
#c = BatchNormalization()(c)
#c = Dropout(rate=0.7)(c)
#c = Dense(k, activation='softmax')(c)
#prediction = Flatten()(c)
# --------------------------------------------------end of pointnet

#Fusion

resnet_activation = Input(shape=(intermediate_output.shape[1],), name='intermediate_output')
f = Concatenate()([global_feature, resnet_activation])

#Definition of MLP Layer
f = Dense(512, activation='relu')(f)
f = Dense(128, activation='relu')(f)
f = Dense(128, activation='relu')(f)
boxes = Dense(labels.shape[-1])(f)
output_classes = Dense(classes.shape[-1])(f)


# print the model summary
model = Model(inputs=[input_points, resnet_activation], outputs=[boxes, output_classes])
print(model.summary())


# Load Data:

# In[15]:


#index = np.load('permuted_indices.npy')

index = range(7481)
train_points = points[index[0:6750]]
dev_points = points[index[6750:7115]]
test_points = points[index[7115:]]
print(points.shape)
for i in range(classes.shape[0]):
    print(classes[i])
train_classes = classes[index[0:6750]]
dev_classes = classes[index[6750:7115]]
test_classes = classes[index[7115:]]

train_labels = labels[index[0:6750]]
dev_labels = labels[index[6750:7115]]
test_labels = labels[index[7115:]]

train_intermediate = intermediate_output[index[0:6750]]
dev_intermediate = intermediate_output[index[6750:7115]]
test_intermediate = intermediate_output[index[7115:]]

print(train_points.shape)
print(train_labels.shape)
print(train_classes.shape)
print(train_intermediate.shape)

print(dev_points.shape)
print(dev_labels.shape)
print(dev_classes.shape)
print(dev_intermediate.shape)

print(test_points.shape)
print(test_labels.shape)
print(test_classes.shape)
print(test_intermediate.shape)


# Training:

# In[ ]:


HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)
  
  
#epoch number
epo = 450
# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)
# compile classification model
model.compile(optimizer='adam',
              loss=[smoothL1, 'mean_squared_error'],
              metrics=['accuracy'])

history = model.fit(x = [train_points, train_intermediate], y= [train_labels, train_classes], batch_size=32, epochs=epo, validation_data=([dev_points,dev_intermediate], [dev_labels, dev_classes]), shuffle=True, verbose=1)


# In[ ]:


#model.save('/drive/Colab Notebook/current_model')
import pickle

with open('./trainHistoryDict_history450', 'wb') as file_pi:
     pickle.dump(history.history, file_pi)


# In[ ]:


model.save_weights('./my_model_weights_450.h5')


# In[31]:


# Evaluating the model on the test data    
loss = model.evaluate([test_points, test_intermediate], [test_labels, test_classes], verbose=0)
print('Test Loss:', loss)


# In[44]:


#Evaluating model of Dev Set
loss = model.evaluate([dev_points, dev_intermediate], [dev_labels, dev_classes], verbose=0)
print('Dev Loss:', loss)

