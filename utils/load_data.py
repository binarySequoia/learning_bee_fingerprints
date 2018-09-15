import numpy as np
import pandas as pd
from skimage import io


def load_train_data(csv_fname, sample):
    
    
    train_df = pd.read_csv(csv_fname)
    
    y = None
    
    if sample > 0:
        X1_paths = list(train_df.X1)[:sample]
        X2_paths = list(train_df.X2)[:sample]
        y = np.array(train_df.y)[:sample]
    else:
        X1_paths = list(train_df.X1)
        X2_paths = list(train_df.X2)
        y = np.array(train_df.y)
    
    X1 = (np.array(io.imread_collection(X1_paths)) - 128.0) / 255.
    X2 = (np.array(io.imread_collection(X2_paths)) -128.0) / 255.
    
    return ([X1, X2], y), ([X1_paths, X2_paths])
    

def load_test_data(csv_fname, sample):
    test_df = pd.read_csv(csv_fname)
    
    y_test = None
    
    if sample > 0:
        X1_test_paths = list(test_df.X1)[:sample]
        X2_test_paths = list(test_df.X2)[:sample]
        y_test = np.array(test_df.y)[:sample]
    else:
        X1_test_paths = list(test_df.X1)
        X2_test_paths = list(test_df.X2)
        y_test = np.array(test_df.y)



    X1_test = (np.array(io.imread_collection(X1_test_paths))  - 128.0) / 255.
    X2_test = (np.array(io.imread_collection(X2_test_paths)) - 128.0) / 255.

    
    return ([X1_test, X2_test], y_test), ([X1_test_paths, X2_test_paths])

        

# Todo return a dictionary
def load_data(train_csv_fn, test_csv_fn, sample=-1):
    """
    Return a dictionary with the following keys:
    - X         :   Pair of images to train the model.
    - y         :   Label of X_pairs. 1 means not same class, 
                    and 0 means same class.
    - X_test    :   Pair of images to test the model.
    - y_test    :   Label of X_pairs_t. 1 means not same class, 
                    and 0 means same class.         
    """
    
    data = dict()
    (data["X"], data["y"]), (data["X_fn"]) = load_train_data(train_csv_fn, sample=sample)
    (data["X_test"], data["y_test"]), (data["X_test_fn"]) = load_test_data(test_csv_fn, sample=sample)
    
    return data
