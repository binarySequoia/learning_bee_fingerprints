from keras.models import Model, Sequential 
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
import keras

def shallow(input_shape):
    model = Sequential()

    model.add(Conv2D(32, 11, activation='relu', 
                     input_shape=input_shape))
    model.add(Conv2D(64, 9, activation='relu', padding='VALID'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    
    return model