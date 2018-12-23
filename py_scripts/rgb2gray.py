import skimage
from skimage import io
from skimage import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def gray_path(filepath):
    file = filepath.split(".")
    file[-2] += "-gray"
    file = file[-2:]
    file = ".".join(file).split("/")[-1]
    return file

df = pd.read_csv("../datasets/body_sept/test_pairs.csv")

train_data = np.append(df.X1.values, df.X2.values)

train_data =np.unique(train_data)

fname_list = list()

for fname in train_data:
    rgb = io.imread(fname)
    grey = color.rgb2grey(rgb)
    gray_filepath = os.path.join("..", "datasets", "body_sept", "gray_test", gray_path(fname))
    fname_list.append(gray_filepath)
    io.imsave(gray_filepath, grey)
    print(gray_filepath, " has been saved.")

df_dict = {
    "fname" : fname_list
}

df = pd.DataFrame(df_dict)

df.to_csv("../datasets/body_sept/gray_test.csv")