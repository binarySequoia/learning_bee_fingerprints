from multiprocessing.pool import ThreadPool
import pandas as pd
from skimage import io
import numpy as np
import dask

import dask.array as da
from sklearn.decomposition import IncrementalPCA
#from dask_ml.decomposition import PCA
import sys

sys.path.append("../")
from utils.utils import *


#dask.config.set(pool=ThreadPool(20))

output_dim = 64
CURRENT_OUTPUT_PATH = "../stats/pca{}".format(output_dim)

train_df = pd.read_csv("../datasets/body_sept/gray_train.csv")
test_df = pd.read_csv("../datasets/body_sept/gray_test.csv")

train_img = np.append(train_df.X1.values[:59800], train_df.X2.values[:59800])
train_np_img = np.array(io.imread_collection(train_img)).reshape((-1, 320*250))

dX = da.from_array(train_np_img, chunks=(100,320*250))

#pca = PCA(n_components=output_dim)
pca =  IncrementalPCA(n_components=output_dim, batch_size=100)

print("Training Data...")
#with dask.config.set(pool=ThreadPool(10)):
pca.fit(dX)
print("Delete temp variables")
del dX, train_np_img
print("PCA on Train Data...")
X1 = np.array(io.imread_collection(train_df.X1.values)).reshape((-1, 320*250))
X2 = np.array(io.imread_collection(train_df.X2.values)).reshape((-1, 320*250))

dX1 = da.from_array(X1, chunks=(100,320*250))
dX2 = da.from_array(X2, chunks=(100,320*250))

pcaX1 = pca.transform(dX1)
pcaX2 = pca.transform(dX1)

del dX1, dX2, X1, X2
print("PCA on Test Data...")
X1_t = np.array(io.imread_collection(test_df.X1.values)).reshape((-1, 320*250))
X2_t = np.array(io.imread_collection(test_df.X2.values)).reshape((-1, 320*250))

dX1_t = da.from_array(X1_t, chunks=(100,320*250))
dX2_t = da.from_array(X2_t, chunks=(100,320*250))

pcaX1_t = pca.transform(dX1_t)
pcaX2_t = pca.transform(dX2_t)

del dX1_t, dX2_t, X1_t, X2_t

train_dist = distance(pcaX1, pcaX2) 
test_dist = distance(pcaX1_t, pcaX2_t) 

##########################################################
# Save Train and Test predictions
# TODO: Refactor this code
df_dict_train = {
    "X1": train_df.X1.values,
    "X2": train_df.X2.values,
    "y_pred": train_dist,
    "y_true": train_df.y.values
}

df_dict_test = {
    "X1": test_df.X1.values,
    "X2": test_df.X2.values,
    "y_pred": test_dist,
    "y_true": test_df.y.values
}

train_csv_filename = CURRENT_OUTPUT_PATH + "/pca{}-train_results.csv".format(output_dim)
test_csv_filename = CURRENT_OUTPUT_PATH + "/pca{}-test_results.csv".format(output_dim)

train_df = pd.DataFrame.from_dict(df_dict_train)
train_df["training"] = 1
print("Saving train csv in {}".format(train_csv_filename))
train_df.to_csv(train_csv_filename, index=False)

test_df = pd.DataFrame.from_dict(df_dict_test)
test_df["training"] = 0
print("Saving test csv in {}".format(test_csv_filename))
test_df.to_csv(test_csv_filename, index=False)
