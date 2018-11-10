from keras.models import Model, Sequential 
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
import keras

def vggnet_6(input_shape):

    model = keras.applications.vgg16.VGG16(include_top=False, 
                                           weights='imagenet', 
                                           input_tensor=None, 
                                           input_shape=input_shape, 
                                           pooling=None, 
                                           classes=64)
    for i in range(3):
        model.layers[i].trainable = False
        
    x = BatchNormalization()(model.layers[5].output)
    x = Conv2D(128, 7, strides=(2,2), activation='relu', 
                     padding='SAME')(x)
    x = Conv2D(128, 5, activation='relu', padding='VALID')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc3')(x)
    x = Dense(64, activation='relu', name='fc2')(x)
    mod = keras.Model(model.inputs, x)
    return mod
