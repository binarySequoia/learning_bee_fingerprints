from keras.models import Sequential 
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Input, concatenate
from keras.layers.normalization import BatchNormalization
import keras



def one_layer_inception(input_shape):
    
    input_img = Input(shape=input_shape)
    
    tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
    
    output = concatenate([tower_1, tower_2, tower_3], axis = 3)

    output = Flatten()(output)
    
    #output = Dense(512, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    
    mod = keras.Model(input_img, output)
    
    return mod
