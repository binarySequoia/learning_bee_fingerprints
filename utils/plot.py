import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns
from skimage import io
import math

def get_roc_curve_image(y_true, y_pred, filename):
    fpr, tpr, thr = roc_curve(y_true, y_pred)

    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate");
    plt.savefig(filename)

    
def get_hist_by_class_image(y_true, y_pred, filename):
    not_same = y_pred[y_true == 1]
    same = y_pred[y_true == 0] 
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8,8))
    
    ax[0].hist(same, label="Same")
    ax[0].set_xlabel("Distance between similar bees")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Distance between similar histogram")
    ax[0].legend()
    
    ax[1].hist(not_same, label="Not Same", color='orange')
    ax[1].set_xlabel("Distance between not similar bees")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Distance between not similar histogram")
    ax[1].legend()
    
    plt.savefig(filename)

def get_class_hist_image(y_true, y_pred, filename):
    not_same = y_pred[y_true == 1]
    same = y_pred[y_true == 0]
    
    plt.figure(figsize=(12, 8))
    sns.distplot(same, label='same')
    sns.distplot(not_same, label='not same')
    plt.legend()
    plt.title("Same or not same distance distribution",  fontsize=14)
    plt.xlabel("distance", fontsize=12)
    plt.ylabel("frequency", fontsize=12);
    
    plt.savefig(filename)
    

def plot_bees(images):
    
    images_amount = len(images)
    
    fig = plt.figure()
    
    col = 10 if (images_amount > 10) else images_amount 
    row = int(math.ceil(images_amount / col))
    print(images_amount)
    print(col * row)
    
    print(images.shape)
    for i in range(images_amount):
        fig.add_subplot(row, col, i+1)
        print(i)
        plt.imshow(images[i])

    plt.show()
    