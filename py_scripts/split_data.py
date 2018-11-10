
# coding: utf-8

# In[1]:

import numpy as np
from skimage import io
import seaborn as sns
import os
import json
import sys
from ipywidgets import interact
sys.path.append("../")
from utils.metadata import *


# In[2]:

IMG_DIR = '../raw_data/body_sept'
DOCS_DIR = '../docs/exp2/'
OUTPUT_DIR = '../datasets/body_sept'


# In[3]:


meta = Metadata()

#meta.read_data_metadata("../raw_data/dataset1/metadata", key="raw_data_meta")


# In[4]:


meta.metadata


# In[5]:


def get_frame_id(fname):
    return fname.split("-")[1]

def get_track_id(fname):
    return fname.split("-")[2].split(".")[0]

def get_name(fname):
    return fname.split('/')[-1]


# In[6]:


os.path.join(IMG_DIR, "*.jpg")


# In[7]:


images = io.imread_collection(os.path.join(IMG_DIR, "*.jpg"))


# In[8]:


files = images.files
images_size = len(files)


# In[9]:


images_np = np.array(images)

images_size = images_np.shape[0]

# ## Count Bees By Frames

# In[12]:


bees_count_by_frame = dict()
for fname in images.files:
    frame = get_frame_id(fname)
    if frame in bees_count_by_frame:
        bees_count_by_frame[frame] += 1
    else:
        bees_count_by_frame[frame] = 1


# In[13]:


frames_bee_count = list()
for key, val in bees_count_by_frame.items():
#     print(key, val)
    frames_bee_count.append(val)
frames_bee_count = np.array(frames_bee_count)

# In[14]:

# In[16]:


bees_count_by_track = dict()
for fname in images.files:
    track = get_track_id(fname)
    if track in bees_count_by_track:
        bees_count_by_track[track] += 1
    else:
        bees_count_by_track[track] = 1


# In[17]:


tracks_bee_count = list()
for key, val in bees_count_by_track.items():
#     print(key, val)
    tracks_bee_count.append(val)
tracks_bee_count = np.array(tracks_bee_count)


# In[18]:
# In[21]:


images_size = len(images.files)
images_size


# In[22]:


train_size = int(images_size * .8)
train_size


# In[23]:


test_size = int(images_size * .2)
test_size


# In[24]:


test_size + train_size


# ## Store Train data by frame

# In[26]:


train_data_path = os.path.join(OUTPUT_DIR, 'train_data')
train_frame_data_path = os.path.join(OUTPUT_DIR, 'train_data', 'frame')
train_track_data_path = os.path.join(OUTPUT_DIR, 'train_data', 'track')

test_data_path = os.path.join(OUTPUT_DIR, 'test_data')
test_frame_data_path = os.path.join(OUTPUT_DIR, 'test_data', 'frame')
test_track_data_path = os.path.join(OUTPUT_DIR, 'test_data', 'track')

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


if not os.path.isdir(train_data_path):
    os.mkdir(train_data_path)
if not os.path.isdir(train_frame_data_path):
    os.mkdir(train_frame_data_path)
if not os.path.isdir(train_track_data_path):
    os.mkdir(train_track_data_path)

if not os.path.isdir(test_data_path):
    os.mkdir(test_data_path)
if not os.path.isdir(test_frame_data_path):
    os.mkdir(test_frame_data_path)
if not os.path.isdir(test_track_data_path):
    os.mkdir(test_track_data_path)


# In[27]:


for img, fname in zip(images, images.files):
    frame = get_frame_id(fname)
    
    if int(frame) > 10000:
        continue
    
    frame_path = os.path.join(train_frame_data_path , frame)
    filepath = os.path.join(frame_path, get_name(fname))
    if os.path.isdir(frame_path):
        io.imsave(filepath, img)
    else:
        os.mkdir(frame_path)
        io.imsave(filepath, img)


train_frame_info = "Train data frames was created by taking the first 600 frames"


# ## Store Train data by track

# In[29]:


TRACK_COUNTS = 0
for img, fname in zip(images, images.files):
    frame = get_frame_id(fname)
    track = get_track_id(fname)

    if int(frame) > 10000:
        continue
    
    track_path = os.path.join(train_track_data_path , track)
    filepath = os.path.join(track_path, get_name(fname))
    
    
    if os.path.isdir(track_path):
        if len(os.listdir(track_path)) > 25:
            continue
        io.imsave(filepath, img)
    else:
        os.mkdir(track_path)
        TRACK_COUNTS += 1
        print(track_path)
        io.imsave(filepath, img)


train_tracks_info = "Train data tracks was created by taking the tracks in the first 600 frames, but each track need to be more than 25 frames long."
train_tracks_count = TRACK_COUNTS


# ## Store Test data by frame

# In[ ]:


for img, fname in zip(images, images.files):
    frame = get_frame_id(fname)
    
    if int(frame) < 16000:
        continue
    
    frame_path = os.path.join(test_frame_data_path , frame)
    filepath = os.path.join(frame_path, get_name(fname))
    if os.path.isdir(frame_path):
        io.imsave(filepath, img)
    else:
        os.mkdir(frame_path)
        io.imsave(filepath, img)


test_frame_info = "Test data frames was created by taking the frames after the first 800"


# ## Store Test data by track

# In[ ]:


TRACK_COUNTS_T = 0
for img, fname in zip(images, images.files):
    frame = get_frame_id(fname)
    track = get_track_id(fname)
    
    if int(frame) < 16000:
        continue
    
    track_path = os.path.join(test_track_data_path , track)
    filepath = os.path.join(track_path, get_name(fname))
    
    
    if os.path.isdir(track_path):
        if len(os.listdir(track_path)) > 25:
            continue
        io.imsave(filepath, img)
    else:
        os.mkdir(track_path)
        TRACK_COUNTS_T += 1
        io.imsave(filepath, img)
    


test_tracks_info = "Test data tracks was created by taking the tracks after the first 800 frames, but each track need to be more than 25 frames long."
test_tracks_count = TRACK_COUNTS_T


# In[ ]:


meta.add_metadata("dataset_meta", { 
        "info" :  {
            "train_frames" : train_frame_info,
            "train_tracks" : train_tracks_info,
            "train_track_count" : train_tracks_count,
            "test_frames" : test_frame_info,
            "test_tracks" : test_tracks_info,
            "test_track_count" : test_tracks_count,
        }
    })


# In[ ]:


meta.metadata


# In[ ]:


meta.save(os.path.join(OUTPUT_DIR, "metadata"))

