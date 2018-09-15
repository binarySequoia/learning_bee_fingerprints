from keras.models import Model, Sequential 
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
import keras

def vggnet_8v2_dbl_sigmoid(input_shape):

    model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None, classes=64)
    for i in range(8):
        model.layers[i].trainable = False
        
    x = BatchNormalization()(model.layers[-1].output)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='sigmoid', name='fc1')(x)
    x = Dense(64, activation='sigmoid', name='fc2')(x)
    mod = keras.Model(model.inputs, x)
    return mod
