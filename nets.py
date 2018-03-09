from __future__ import division
from keras.layers import Conv2D, Input, add
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
import numpy as np


def network(input_shape, K=12):
    input = Input(shape=input_shape, name='input')
    layer = Conv2D(128, (5,5), padding='same', activation=None, name='conv1')(input)
    layer = LeakyReLU(0.2)(layer)

    for i in np.arange(2, K):
        layer = Conv2D(64, (3,3), padding='same', name='conv%d'%i, activation=None)(layer)
        layer = LeakyReLU(0.2)(layer)

    # no activation at the last layer
    layer = Conv2D(input_shape[-1], (3,3), padding='same', activation=None, name = 'conv%d'%(i+1))(layer)
    layer = add([input, layer])
    model = Model(inputs=input, outputs=layer)
    return model
