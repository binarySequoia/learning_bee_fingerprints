import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pandas as pd 


def contact_sheet(filename, df, sample_size=-1):
    
    size = len(df)
    
    if sample_size > 0:
        rnd_idx = np.random.permutation(size)
        size = sample_size if sample_size < size else size
        df = df.iloc[rnd_idx, :]
    
    df = df.reset_index()
    
    x1 = np.array(io.imread_collection(df.loc[:size, "X1"]))
    x2 = np.array(io.imread_collection(df.loc[:size, "X2"]))
    
    x1_frame = "frame : " + df.X1.str.split("/").str[-1].str.split("--").str[1].str.split(".").str[0]
    x1_id = "id : " + df.X1.str.split("/").str[-1].str.split("--").str[0].str.split("bee").str[1]
    
    x2_frame = "frame : " + df.X2.str.split("/").str[-1].str.split("--").str[1].str.split(".").str[0]
    x2_id = "id : " + df.X2.str.split("/").str[-1].str.split("--").str[0].str.split("bee").str[1]
    
    y = df.y_pred.round(2)
    columns = 20
    rows = int(size/10.0) + 1
    j = 0
    fig=plt.figure(figsize=(40, 4*rows))
    for x in range(size):
  
        fig.add_subplot(rows, columns, j+1)
        plt.imshow(x1[x, :, :, :])
        plt.xlabel(x1_id[x]  + " | SC: " + str(y[x]))
        plt.title(x1_frame[x])
        plt.xticks([], [])
        plt.yticks([], [])
        
        fig.add_subplot(rows, columns, j+2)
        plt.imshow(x2[x, :, :, :])
        plt.xlabel(x2_id[x] + " | SC: " + str(y[x]))
        plt.title(x2_frame[x])
        plt.xticks([], [])
        plt.yticks([], [])
        
        j += 2
    
    plt.show()
    fig.savefig(filename)   