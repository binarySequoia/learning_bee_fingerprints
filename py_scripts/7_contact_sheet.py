import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pandas as pd
import argparse
import os

import sys
sys.path.append("../")
from utils.contact_sheet import *

parser = argparse.ArgumentParser(description='Create a contact sheet from specific dataset and model')
parser.add_argument('-d', '--dataframe', metavar='', required=True,
                    help='network architeture name')
parser.add_argument('-o', '--output', metavar='', required=True,
                    help='output name')
parser.add_argument('-t', '--threshold', metavar='', type=float, default=1.0,
                    help='margin')
parser.add_argument('-s', '--sample', type=int, default=-1,
                    help='Sample Size Limit')
parser.add_argument('-f', '--folder', default="",
                   help="Folder")


args = parser.parse_args()


THRESHOLD = args.threshold
DATAFRAME = args.dataframe
OUTPUT_NAME = args.output
SAMPLE_SIZE = args.sample
OUTPUT_FOLDER = args.folder

NOT_SAME = 1
SAME = 0

df = pd.read_csv(DATAFRAME).sort_values("y_pred")


fn_index = (df.y_true == NOT_SAME) & (df.y_pred < THRESHOLD)
fp_index = (df.y_true == SAME) & (df.y_pred > THRESHOLD)
tp_index = (df.y_true == NOT_SAME) & (df.y_pred > THRESHOLD)
tn_index = (df.y_true == SAME) & (df.y_pred < THRESHOLD)

fn = df[fn_index]
fp = df[fp_index]
tp = df[tp_index]
tn = df[tn_index]

contact_sheet(os.path.join(OUTPUT_FOLDER, "false_negatives-{}.pdf".format(OUTPUT_NAME)), fn,
              sample_size=SAMPLE_SIZE)
contact_sheet(os.path.join(OUTPUT_FOLDER, "false_positives-{}.pdf".format(OUTPUT_NAME)), fp,
              sample_size=SAMPLE_SIZE)
contact_sheet(os.path.join(OUTPUT_FOLDER, "true_positives-{}.pdf".format(OUTPUT_NAME)), tp,
              sample_size=SAMPLE_SIZE)
contact_sheet(os.path.join(OUTPUT_FOLDER, "true_negatives-{}.pdf".format(OUTPUT_NAME)), tn,
              sample_size=SAMPLE_SIZE)