# coding: utf-8

# # Siamese Network

# #### dependecies

# In[14]:


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
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Merge
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
import argparse
import sys
import json

sys.path.append("../")
from utils.load_data import *
from utils.paths import *
from utils.utils import *
from networks.networks import *


# TODO: ADD option for retrain network
parser = argparse.ArgumentParser(description='Train a specific network')
parser.add_argument('-n', '--net', metavar='', required=True,
                    help='network architeture name')
parser.add_argument('-o', '--output', metavar='', required=True,
                    help='output name')
parser.add_argument('-m', '--margin', metavar='', type=float, default=1.0,
                    help='margin')
parser.add_argument('-e', '--epochs', metavar='', type=int, default=60,
                    help='epochs')
parser.add_argument('-t', '--train', metavar='', default='../datasets/dataset2/train_pairs.csv',
                    help='train data csv file')
parser.add_argument('-s', '--test', metavar='', default='../datasets/dataset2/test_pairs.csv',
                    help='test data csv file')
parser.add_argument('-p', '--optimizer', metavar='', default='SGD',
                    help='Optimizer')
parser.add_argument('-a', '--earlystop', metavar='', default=False,
                    help='earlystop flag')
parser.add_argument('-l', '--logloss', action='store_true',
                    help='contrasive log loss')
parser.add_argument('-r', '--re-train', metavar='', type=str, nargs=1,
                    help='Model to re train')
parser.add_argument('--debug', action='store_true',
                    help='debug mode')
parser.add_argument('--alpha', metavar='', type=float, default=1.0,
                    help='alpha value')


args = parser.parse_args()
NETWORK_NAME = args.net
OUTPUT_NAME = args.output
MARGIN_VALUE = args.margin
EARLY_STOP = args.earlystop
TRAIN_CSV = args.train
TEST_CSV = args.test
OPTIMIZER = args.optimizer
EPOCHS = args.epochs
LOGLOSS = args.logloss
RETRAIN = args.re_train
DEBUG = args.debug
ALPHA = args.alpha
#CONTACT_SHEET = args.no_sheet

CSV_DIR = '../data/csv/'
MODELS_DIR = '../models/'
LOGS_DIR = '../logs/'
WEIGHTS_DIR = '../weights/'
OUTPUT_PATH = "../stats/"

# ### Load Data

# In[15]:

network = base_network(NETWORK_NAME)

def read_data_metadata(fn):
    data = None
    with open(fn, "r") as f:
        data = json.load(f)
    return data

metadata = read_data_metadata(os.path.join("../datasets/dataset1", "metadata"))


print("Loading Data...")
# TODO: Clean This
# TODO: Add model and dataset Metadata 

if DEBUG:
    data = load_data(TRAIN_CSV, TEST_CSV, sample=100)
else:
    data = load_data(TRAIN_CSV, TEST_CSV)

[X1, X2] = data["X"]
y = data["y"]
[X1_fn, X2_fn] = data["X_fn"]

[X1_test, X2_test] = data["X_test"]
y_test = data["y_test"]
[X1_test_fn, X2_test_fn] = data["X_test_fn"]

del data



metadata["training"] = dict()



# ### Configure Siamese Network

# #### Loss and distance related Functions

# In[24]:

# TODO: Mve this funtion to utils folder



# In[25]:

LOSS = contrastive_loss(MARGIN_VALUE)
metadata["training"]["loss"] = "contrastive_loss"

if LOGLOSS:
    LOSS = contrastive_log_loss
    metadata["training"]["loss"] = "contrastive_log_loss"


metadata["training"]["margin"] = MARGIN_VALUE
metadata["training"]["epochs"] = EPOCHS


# In[26]:


network.summary()


# In[27]:


input_a = Input(shape=[230, 105, 3])
input_b = Input(shape=[230, 105, 3])


# In[28]:


processed_a = network(input_a)
processed_b = network(input_b)


# #### Merge networks with euclidean distance

# In[29]:


distance_merge = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])


# In[30]:


model = Model([input_a, input_b], distance_merge)


# In[31]:


# model.save(os.path.join(MODEL_DIR, "train_4.h5"))


# #### Configure loss and other parameters

# In[32]:



model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[accuracy, accuracy1, accuracy5])


# In[33]:


model.summary()


# #### Set-up TensorBoard

# In[34]:


tf_board = TensorBoard(os.path.join(LOGS_DIR, OUTPUT_NAME))


# In[35]:

callbacks = [tf_board]

if(EARLY_STOP):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=15, verbose=1, mode='min')
    callbacks = [tf_board, earlystop]


# ### Train Siamese Network 

# In[36]:

#TODO: save history in a csv format
history = model.fit([X1, X2], y, 
                    batch_size=10, epochs=EPOCHS, verbose=1, callbacks=callbacks,
                    validation_data=([X1_test, X2_test], y_test))


# In[ ]:

WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "{}-w.h5".format(OUTPUT_NAME))
model.save(os.path.join(MODELS_DIR, "{}.h5".format(OUTPUT_NAME)))
model.save_weights(WEIGHTS_FILE)

# Load Model
#######################################
print("Loading Model...")
network = base_network(NETWORK_NAME)

network_input = Input([230, 105, 3])

model = network(network_input)

model = Model(network_input, model)

model.load_weights(WEIGHTS_FILE)
#######################################


# Predict on training and testing data
print("Predicting Training Data...")
train_pred_x1 = model.predict(X1)
train_pred_x2 = model.predict(X2)

print("Predicting Testing Data...")
test_pred_x1 = model.predict(X1_test)
test_pred_x2 = model.predict(X2_test)

train_dist = distance(train_pred_x1, train_pred_x2) 
test_dist = distance(test_pred_x1, test_pred_x2) 

metadata["training"]["train_loss"] = contrastive_loss_cpu(MARGIN_VALUE)(y, train_dist)
metadata["training"]["test_loss"] = contrastive_loss_cpu(MARGIN_VALUE)(y_test, test_dist)

if(OUTPUT_NAME == ''):
    OUTPUT_NAME = NETWORK_NAME 

OUTPUT_PATH += OUTPUT_NAME
    
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

# TODO: Module this code

df_dict_train = {
    "X1": X1_fn,
    "X2": X2_fn,
    "y_pred": train_dist,
    "y_true": y
}

df_dict_test = {
    "X1": X1_test_fn,
    "X2": X2_test_fn,
    "y_pred": test_dist,
    "y_true": y_test
}

train_df = pd.DataFrame.from_dict(df_dict_train)
train_df["training"] = 1
train_df.to_csv(OUTPUT_PATH + "/{0}-train_results.csv".format(OUTPUT_NAME), index=False)

test_df = pd.DataFrame.from_dict(df_dict_test)
test_df["training"] = 0
test_df.to_csv(OUTPUT_PATH + "/{0}-test_results.csv".format(OUTPUT_NAME), index=False)

# TODO: Add option for CONTACT SHEET

with open(os.path.join(OUTPUT_PATH, "metadata"), "w") as f:
    f.write(json.dumps(metadata, indent=2))


