from keras.models import Model, Sequential 
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
import keras

def vggnet_8_dbl_sigmoid(input_shape):

    model = keras.applications.vgg16.VGG16(include_top=False, 
                                           weights='imagenet', 
                                           input_tensor=None, 
                                           input_shape=input_shape, 
                                           pooling=None, 
                                           classes=64)
    for i in range(8):
        model.layers[i].trainable = False
        
    x = BatchNormalization()(model.layers[-1].output)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='sigmoid', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='sigmoid', name='fc3')(x)
    x = Dense(64, activation='sigmoid', name='fc2')(x)
    mod = keras.Model(model.inputs, x)
    return mod
