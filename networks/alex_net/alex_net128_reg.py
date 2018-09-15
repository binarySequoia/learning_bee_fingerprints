from keras.models import Sequential 
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def alex_net128_reg(input_shape):
    model = Sequential()

    model.add(Conv2D(32, 11, activation='relu', 
                     input_shape=input_shape))
    model.add(MaxPooling2D(3, 2))

    model.add(Conv2D(64, 9, activation='relu', padding='VALID'))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(128, 7, strides=(2,2), activation='relu', 
                     padding='SAME'))
    #model.add(Conv2D(128, 5, activation='relu', padding='VALID'))
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    
    return model