import numpy as np
import pandas as pd
import os

def get_frame_id(fname):
    """
    USAGE: get_frame_id(fname)
    Get frame ID of the image with the following format:
        'Bee-{frameId}-{trackId}.jpg'
    Parameters:
        - fname : filename of the image
    Return:
        - frame_id of the image      
    """
    
    return fname.split("-")[1]


def get_track_id(fname):
    """
    USAGE: get_frame_id(fname)
    Get track ID of the image with the following format:
        'Bee-{frameId}-{trackId}.jpg'
    Parameters:
        - fname : filename of the image
    Return:
        - trackId of the image      
    """
    return fname.split("-")[2].split(".")[0]

