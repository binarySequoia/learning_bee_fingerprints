import sys
sys.path.append("../../")
from utils.utils import get_model_memory_usage
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def alex_net256_reg(input_shape):
    model = Sequential()

    model.add(Conv2D(32, 9, activation='relu', 
                     input_shape=input_shape))
    
    model.add(MaxPooling2D(3, 3))

    model.add(Conv2D(64, 7, activation='relu', padding='VALID'))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Conv2D(128, 5, strides=(2,2), activation='relu', 
                     padding='SAME'))
    
    
    model.add(Conv2D(256, 3, strides=(2,2), activation='relu', 
                     padding='SAME'))
    #model.add(Conv2D(128, 5, activation='relu', padding='VALID'))
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    
    return model


if __name__ == "__main__":
    model = alex_net256_reg([320, 250, 3])
    model.summary()
    print(get_model_memory_usage(256, model), "GB" )