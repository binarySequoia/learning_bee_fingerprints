import numpy as np
import pandas as pd
from skimage.io import imread
from keras.utils import Sequence

class TrainGenerator(Sequence):

    def __init__(self, csv_path, batch_size=128):
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df)
        self.X1 = self.df.X1.values
        self.X2 = self.df.X2.values
        self.y = self.df.y.values
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(self.len // float(self.batch_size)).astype(int)
        #return self.len

    def __getitem__(self, idx):
        batch_x1 = self.X1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.X2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X1_out = [imread(filename) for filename in batch_x1]
        X2_out = [imread(filename) for filename in batch_x2]
        y_out = batch_y
        return ([np.array(X1_out)/255.0, np.array(X2_out)/255.0], y_out)
    

class TestGenerator(Sequence):

    def __init__(self, df, batch_size=1):
        self.X = df
        self.batch_size = batch_size
        self.len = len(self.X)
        
    def __len__(self):
        return np.ceil(self.len // float(self.batch_size)).astype(int)

    def __getitem__(self, idx):
        
        if idx > self.len - self.batch_size:
            batch_x1 = self.X[idx * self.batch_size:]
        else:
            batch_x1 = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]

        X1_out = [imread(filename) for filename in batch_x1]

        return np.array(X1_out)/255.0