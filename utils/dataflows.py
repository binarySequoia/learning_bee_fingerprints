from tensorpack import DataFlow
import numpy as np
import pandas as pd
from skimage import io

def load_batch(file_list):
    images = np.array(io.imread_collection(file_list))
    return (images - 128.0) / 255.

class FeatureDataFlow(DataFlow):
    def __init__(self, csv_path, batch_size=128):
        self.csv_path = csv_path
        self.batch_size = batch_size        
        
    def __iter__(self):
        df = pd.read_csv(self.csv_path)
        data_size = len(df)
        i = 0
        while i < data_size:
            if data_size - i < self.batch_size:
                current_batch_size = data_size - i
            else:
                current_batch_size = self.batch_size
            batch_files = df.iloc[i:i+current_batch_size, :]
            X1 = load_batch(batch_files.X1.values)
            X2 = load_batch(batch_files.X2.values)
            y = batch_files.y.values
            yield ([X1, X2], y)
            i += current_batch_size
        
        


class LabelDataFlow(DataFlow):
    def __init__(self, csv_path, col, batch_size=128):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.col = col
        
        
    def __iter__(self):
        df = pd.read_csv(self.csv_path)
        data_size = len(df)
        i = 0
        while i < data_size:
            if data_size - i < self.batch_size:
                current_batch_size = data_size - i
            else:
                current_batch_size = self.batch_size
            current_df = df.iloc[i:i+current_batch_size, :]
            y = current_df[self.col].values
            yield y
            i += self.batch_size


class FilenameDataFlow(DataFlow):
    def __init__(self, csv_path, col, batch_size=128):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.col = col
        
        
    def __iter__(self):
        df = pd.read_csv(self.csv_path)
        data_size = len(df)
        i = 0
        while i < data_size:
            if data_size - i < self.batch_size:
                current_batch_size = data_size - i
            else:
                current_batch_size = self.batch_size
            
            batch_files = df.iloc[i:i+current_batch_size, :]
            X = batch_files[self.col].values
            yield X
            i += self.batch_size
    
def train_iter(csv_path, batch_size=64):
    df = pd.read_csv(csv_path)
    data_size = len(df)
    i = 0
    while i < data_size:
        if data_size - i < batch_size:
            current_batch_size = data_size - i
        else:
            current_batch_size = batch_size
        batch_files = df.iloc[i:i+current_batch_size, :]
        X1 = load_batch(batch_files.X1.values)
        X2 = load_batch(batch_files.X2.values)
        y = batch_files.y.values
        yield ([X1, X2], y)
        i += current_batch_size