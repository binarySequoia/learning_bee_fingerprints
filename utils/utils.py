import numpy as np
from keras import backend as K

def distance(x, y):
    return np.sqrt(np.sum((x - y)**2, axis=1))


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.mean(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    threshold = 0.2
    return 1 -  K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))

def accuracy5(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    threshold = 0.5
    return 1 -  K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))

def accuracy1(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    threshold = 1.0
    return 1 -  K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))


def contrastive_loss(MARGIN_VALUE):

    def c_loss(y_diff, y_dist):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = MARGIN_VALUE
        return K.mean((1 - y_diff) * K.square(y_dist) +
                      (y_diff) * K.square(K.maximum(margin - y_dist, 0)))
    return c_loss

def contrastive_loss_cpu(MARGIN_VALUE):
    def c_loss(y_diff, y_dist):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = MARGIN_VALUE
        return np.mean((1 - y_diff) * np.square(y_dist) +
                      (y_diff) * np.square(np.max(margin - y_dist, 0)))
    return c_loss