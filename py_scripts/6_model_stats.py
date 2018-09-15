import sys
import os
import argparse
sys.path.append("../")
from utils.plot import *
from utils.load_data import *
from utils.paths import *
from utils.utils import *
from networks.networks import *
from keras.layers import Input
from keras.models import Model
import pandas as pd

parser = argparse.ArgumentParser(description='Produce stats about an model')

parser.add_argument('-n', '--net', metavar='', required=True,
                    help='network architeture name')
parser.add_argument('-w', '--weights', metavar='', default='',
                   help='model weights file name')
parser.add_argument('-o', '--output', metavar='', default='',
                   help='output name')
parser.add_argument('-t', '--train', metavar='', default='../data/csv/train_pairs.csv',
                    help='train data csv file')
parser.add_argument('-s', '--test', metavar='', default='../data/csv/test_pairs.csv',
                    help='test data csv file')

args = parser.parse_args()
NETWORK_NAME = args.net
OUTPUT_NAME = args.output
TRAIN_CSV = args.train
TEST_CSV = args.test
WEIGHTS_FILE = args.weights

OUTPUT_PATH = "../stats/"

if(OUTPUT_NAME == ''):
    OUTPUT_PATH += NETWORK_NAME 
else:
    OUTPUT_PATH += OUTPUT_NAME

if WEIGHTS_FILE == '':
    WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, NETWORK_NAME + "-w.h5")
    
# Load Data
print("Loading Data...")
([X1, X2], y), ([X1_test, X2_test], y_test) = load_data(TRAIN_CSV, TEST_CSV)

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

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

df_dict_train = {
    "y_train_pred": train_dist,
    "y_train_true": y
}

df_dict_test = {
    "y_test_pred": test_dist,
    "y_test_true": y_test
}

train_df = pd.DataFrame.from_dict(df_dict_train)
train_df.to_csv(OUTPUT_PATH + "/{0}-train_results.csv".format(NETWORK_NAME))

test_df = pd.DataFrame.from_dict(df_dict_test)
test_df.to_csv(OUTPUT_PATH + "/{0}-test_results.csv".format(NETWORK_NAME))

"""
get_roc_curve_image(y, train_dist, OUTPUT_PATH + "/train-{0}-roc.png".format(NETWORK_NAME))
print("Train ROC curve saved in {}.".format(OUTPUT_PATH + "/train-{0}-roc.png".format(NETWORK_NAME)))

get_hist_by_class_image(y, train_dist, OUTPUT_PATH + "/train-{0}-dual_hist.png".format(NETWORK_NAME))
print("Train Histograms by Class saved in {}.".format(OUTPUT_PATH + "/train-{0}-dual_hist.png".format(NETWORK_NAME)))

get_class_hist_image(y, train_dist, OUTPUT_PATH +"/train-{0}-dist_hist.png".format(NETWORK_NAME))
print("Train Dist Histograms curve saved in {}.".format(OUTPUT_PATH +"/train-{0}-dist_hist.png".format(NETWORK_NAME)))



get_roc_curve_image(y_test, test_dist, OUTPUT_PATH +"/test-{0}-roc.png".format(NETWORK_NAME))
print("Test ROC curve saved in {}.".format(OUTPUT_PATH + "/test-{0}-roc.png".format(NETWORK_NAME)))

get_hist_by_class_image(y_test, test_dist, OUTPUT_PATH + "/test-{0}-dual_hist.png".format(NETWORK_NAME))
print("Test Histograms by Class saved in {}.".format(OUTPUT_PATH + "/test-{0}-dual_hist.png".format(NETWORK_NAME)))

get_class_hist_image(y_test, test_dist, OUTPUT_PATH + "/test-{0}-dist_hist.png".format(NETWORK_NAME))
print("Test Dist Histograms curve saved in {}.".format(OUTPUT_PATH + "/test-{0}-dist_hist.png".format(NETWORK_NAME)))
"""
