# coding: utf-8

# # Siamese Network

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
from utils.metadata import *
from utils.dataflows import *
from utils.generator import *
from networks.networks import *


# TODO: ADD option for retrain network
# TODO: Run from file
parser = argparse.ArgumentParser(description='Train a specific network')
parser.add_argument('-n', '--net', nargs='*', metavar='', required=True,
                    help='network architeture name')
parser.add_argument('-o', '--output', nargs='+', metavar='',
                    help='output name')
parser.add_argument('-c', '--name', type=str, metavar='', default="",
                    help='Output name convetion')
parser.add_argument('-m', '--margin', metavar='', type=float, default=1.0,
                    help='margin')
parser.add_argument('-e', '--epochs', metavar='', type=int, default=100,
                    help='epochs')
parser.add_argument('-b', '--batch', metavar='', type=int, default=128,
                    help='Batch Size')
parser.add_argument('-t', '--train', metavar='', default='../datasets/body_sept/train_pairs.csv',
                    help='train data csv file')
parser.add_argument('-s', '--test', metavar='', default='../datasets/body_sept/test_pairs.csv',
                    help='test data csv file')
parser.add_argument('-p', '--optimizer', metavar='', default='SGD',
                    help='Optimizer')
parser.add_argument('-a', '--earlystop', action='store_true',
                    help='earlystop flag')
parser.add_argument('-l', '--logloss', action='store_true',
                    help='contrasive log loss')
#parser.add_argument('-r', '--retrain', metavar='', type=str, default="",
#                    help='Model to re train')
parser.add_argument('--debug', action='store_true',
                    help='debug mode')
parser.add_argument('--alpha', metavar='', type=float, default=1.0,
                    help='alpha value')

args = parser.parse_args()
NETWORKS_NAME = args.net
OUTPUTS_NAME = args.output
MARGIN_VALUE = args.margin
EARLY_STOP = args.earlystop
TRAIN_CSV = args.train
TEST_CSV = args.test
OPTIMIZER = args.optimizer
EPOCHS = args.epochs
LOGLOSS = args.logloss
#RETRAIN = args.re_train
DEBUG = args.debug
ALPHA = args.alpha
BATCH_SIZE = args.batch
NAME_CONV = args.name
#RETRAIN = args.retrain

CSV_DIR = '../data/csv/'
MODELS_DIR = '../models/'
LOGS_DIR = '../logs/'
WEIGHTS_DIR = '../weights/'
OUTPUT_PATH = "../stats/"

# Load Metadata
meta = Metadata()
meta.read_data_metadata(os.path.join("../datasets/dataset2", "metadata"))

##########################################################
# Load DATA
print("Loading Data...")
# TODO: Clean This
# TODO: Add model and dataset Metadata 

#train_iter = FeatureDataFlow("../datasets/body_sept/train_pairs.csv")
data = TrainGenerator("../datasets/body_sept/train_pairs.csv")
test_data = TrainGenerator("../datasets/body_sept/test_pairs.csv")

df = pd.read_csv("../datasets/body_sept/train_pairs.csv")
df_t = pd.read_csv("../datasets/body_sept/test_pairs.csv")

X1_fn = df.X1
X2_fn = df.X2
y = df.y.values

X1_test_fn = df_t.X1
X2_test_fn = df_t.X2
y_test = df_t.y.values


#X1 = TestGenerator(X1_fn)
#X2 = TestGenerator(X2_fn)

#X1_test = TestGenerator(X1_test_fn)
#X2_test = TestGenerator(X2_test_fn)
##########################################################
# Setting Metadata

training_metadata = dict()

LOSS = contrastive_loss(MARGIN_VALUE)
training_metadata["loss"] = "contrastive_loss"

if LOGLOSS:
    LOSS = contrastive_log_loss
    training_metadata["loss"] = "contrastive_log_loss"

training_metadata["margin"] = MARGIN_VALUE
training_metadata["epochs"] = EPOCHS

if NAME_CONV != "" and not OUTPUTS_NAME:
    OUTPUTS_NAME = list()
    for model in NETWORKS_NAME:
        OUTPUTS_NAME.append(NAME_CONV.format(model))
elif not OUTPUTS_NAME:
    OUTPUTS_NAME = NETWORKS_NAME
elif len(OUTPUTS_NAME) != len(NETWORKS_NAME):
    print("Output list is not the same size")
    exit()

for network_name, output_name in zip(NETWORKS_NAME, OUTPUTS_NAME):

    ##########################################################
    # Load Network Architeture
    # network = base_network(NETWORK_NAME)
   
    network = base_network(network_name, input_shape=[320, 250, 3])
    training_metadata["model"] = network_name
    #network.summary()

    ##########################################################
    # Build Siamese Architecture
    # TODO: Infere Input Shape from metadata
    input_a = Input(shape=[320, 250, 3])
    input_b = Input(shape=[320, 250, 3])

    processed_a = network(input_a)
    processed_b = network(input_b)

    # Merge networks with euclidean distance
    distance_merge = Lambda(euclidean_distance,
                            output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance_merge)

    ##########################################################
    # Compile
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[accuracy, accuracy1, accuracy5])

    # Set-up TensorBoard
    tf_board = TensorBoard(os.path.join(LOGS_DIR, output_name))
    callbacks = [tf_board]


    if(EARLY_STOP):
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=15, verbose=1, mode='min')
        callbacks.append(earlystop)

    ##########################################################
    # Train Siamese Network
    # TODO: save history in a csv format
    history = model.fit_generator(data, epochs=EPOCHS, verbose=1, callbacks=callbacks,
                        validation_data=test_data)

    training_metadata["history"] = history.history
    ##########################################################
    # Save Model and Weights
    WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "{}-w.h5".format(output_name))
    MODEL_FILE = os.path.join(MODELS_DIR, "{}.h5".format(output_name))
    print("Saving model in {}".format(MODEL_FILE))
    model.save(MODEL_FILE)
    print("Saving model weights in {}".format(WEIGHTS_FILE))
    model.save_weights(WEIGHTS_FILE)

    #######################################
    # Load Model for tests network
    print("Loading Model...")
    network = base_network(network_name, [320, 250, 3])

    network_input = Input([320, 250, 3])

    model = network(network_input)
    
    model = Model(network_input, model)

    model.load_weights(WEIGHTS_FILE)
    #######################################
    # Predict on training and testing data
    
    
    #train_pred_x1 = model.predict_generator(X1)
    #train_pred_x2 = model.predict_generator(X2)

    
    #test_pred_x1 = model.predict_generator(X1_test)
    #test_pred_x2 = model.predict_generator(X2_test)
    
    train_size = len(X1_fn)
    test_size = len(X1_test_fn)
    batch = 256
    output_dim = model.output.shape.as_list()[1]

    train_pred_x1= np.empty((0,output_dim))
    train_pred_x2 = np.empty((0,output_dim))
    
    test_pred_x1= np.empty((0,output_dim))
    test_pred_x2 = np.empty((0,output_dim))
    print("Predicting Training Data with {}...".format(network_name))
    for i in range(0, train_size, batch):
        if i + batch >= train_size:
            x1_batch = X1_fn[i:].values
            x2_batch = X2_fn[i:].values
        else:   
            x1_batch = X1_fn[i:i+batch].values
            x2_batch = X2_fn[i:i+batch].values
        x1_imgs = np.array(io.imread_collection(x1_batch))/255.0
        x2_imgs = np.array(io.imread_collection(x2_batch))/255.0
        train_pred_x1 = np.append(train_pred_x1, model.predict(x1_imgs), axis=0)
        train_pred_x2 = np.append(train_pred_x2, model.predict(x2_imgs), axis=0)
    
    print("Predicting Testing Data with {} ...".format(network_name))
    for i in range(0, test_size, batch):
        if i + batch >= test_size:
            x1_batch = X1_test_fn[i:].values
            x2_batch = X2_test_fn[i:].values
        else:   
            x1_batch = X1_test_fn[i:i+batch].values
            x2_batch = X2_test_fn[i:i+batch].values
        x1_imgs = np.array(io.imread_collection(x1_batch))/255.0
        x2_imgs = np.array(io.imread_collection(x2_batch))/255.0
        test_pred_x1 = np.append(test_pred_x1, model.predict(x1_imgs), axis=0)
        test_pred_x2 = np.append(test_pred_x2, model.predict(x2_imgs), axis=0)


    train_dist = distance(train_pred_x1, train_pred_x2) 
    test_dist = distance(test_pred_x1, test_pred_x2) 

    training_metadata["train_loss"] = contrastive_loss_cpu(MARGIN_VALUE)(y, train_dist)
    training_metadata["test_loss"] = contrastive_loss_cpu(MARGIN_VALUE)(y_test, test_dist)

    CURRENT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, output_name)

    if not os.path.exists(CURRENT_OUTPUT_PATH):
        os.mkdir(CURRENT_OUTPUT_PATH)

    ##########################################################
    # Save Train and Test predictions
    # TODO: Refactor this code
    df_dict_train = {
        "X1": X1_fn.values,
        "X2": X2_fn.values,
        "y_pred": train_dist,
        "y_true": y
    }

    df_dict_test = {
        "X1": X1_test_fn.values,
        "X2": X2_test_fn.values,
        "y_pred": test_dist,
        "y_true": y_test
    }
    
    train_csv_filename = CURRENT_OUTPUT_PATH + "/{0}-train_results.csv".format(output_name)
    test_csv_filename = CURRENT_OUTPUT_PATH + "/{0}-test_results.csv".format(output_name)
    
    train_df = pd.DataFrame.from_dict(df_dict_train)
    train_df["training"] = 1
    print("Saving train csv in {}".format(train_csv_filename))
    train_df.to_csv(train_csv_filename, index=False)

    test_df = pd.DataFrame.from_dict(df_dict_test)
    test_df["training"] = 0
    print("Saving test csv in {}".format(test_csv_filename))
    test_df.to_csv(test_csv_filename, index=False)

    ##########################################################
    # Save Metadata file
    meta.add_metadata("training", training_metadata)
    metadata_filename = os.path.join(CURRENT_OUTPUT_PATH, "metadata")
    print("Saving metadata in {}".format(metadata_filename))
    meta.save(metadata_filename)

