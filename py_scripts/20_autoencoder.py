
# coding: utf-8

# # Autoencoder

# #### Dependecies

# In[1]:


import random
from skimage import io
import matplotlib.pyplot as plt
from ipywidgets import interact
from keras.models import Model, Sequential 
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Merge, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.layers.core import Lambda
import keras
import os
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import sys
sys.path.append("../")
from networks.networks import *


# ### Load Data

# In[9]:


images = np.array(io.imread_collection("../raw_data/dataset2/*.jpg"))


# In[10]:


images.shape


# In[17]:


reshaped_images = images[:, 3:-3, 1:, :]


# In[18]:


reshaped_images.shape


# #### Built Architeture

# In[14]:


input_layer = Input(shape=(224, 104, 3)) 

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#print(x.shape)
encoded = MaxPooling2D((2, 2), padding='same')(x)
#print(encoded.shape)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
print(decoded.shape)


# In[15]:


autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[19]:


autoencoder.fit(reshaped_images, reshaped_images,
                epochs=50,
                batch_size=128,
                shuffle=True)

