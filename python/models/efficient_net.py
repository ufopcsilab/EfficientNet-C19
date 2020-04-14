#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-05 of 2020
"""

# Imports for model definition
import efficientnet.keras as efn
import tensorflow as tf

from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout

# For model plot
from keras.utils import plot_model

# Base model definition
from models.base_model import BaseModel

# For activation definition
from keras.backend import sigmoid
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=1):
    return x * sigmoid(beta * x)


get_custom_objects().update({'swish': Swish(swish)})
tf.keras.utils.get_custom_objects().update({'swish': Swish(swish)})


# Functions
class EfficientNet(BaseModel):
    """
    Base on: https://github.com/qubvel/efficientnet
    """

    def __init__(self, net, use_weights=True, custom_weights_path=None, use_regularization=False, regularizer=None):
        self._use_weights = use_weights
        self._custom_weights = custom_weights_path
        self._net = net
        self._nets = {'b0': {'name': 'EfficientNetB0', 'class': efn.EfficientNetB0, 'layer_name': 'avg_pool', 'input_shape': (224, 224, 3)},
                      'b1': {'name': 'EfficientNetB1', 'class': efn.EfficientNetB1, 'layer_name': 'avg_pool', 'input_shape': (240, 240, 3)},
                      'b2': {'name': 'EfficientNetB2', 'class': efn.EfficientNetB2, 'layer_name': 'avg_pool', 'input_shape': (260, 260, 3)},
                      'b3': {'name': 'EfficientNetB3', 'class': efn.EfficientNetB3, 'layer_name': 'avg_pool', 'input_shape': (300, 300, 3)},
                      'b4': {'name': 'EfficientNetB4', 'class': efn.EfficientNetB4, 'layer_name': 'avg_pool', 'input_shape': (380, 380, 3)},
                      'b5': {'name': 'EfficientNetB5', 'class': efn.EfficientNetB5, 'layer_name': 'avg_pool', 'input_shape': (456, 456, 3)},
                      'b6': {'name': 'EfficientNetB6', 'class': efn.EfficientNetB6, 'layer_name': 'avg_pool', 'input_shape': (528, 528, 3)},
                      'b7': {'name': 'EfficientNetB7', 'class': efn.EfficientNetB7, 'layer_name': 'avg_pool', 'input_shape': (600, 600, 3)}
                      }
        super(EfficientNet, self).__init__(use_regularization, regularizer)

    def create_net(self,  _number_of_classes=3, _output_path=None):

        if self._use_weights:
            if self._custom_weights is None:
                weights = 'imagenet'
            else:
                weights = self._custom_weights
        else:
            weights = None

        base_model = self._nets[self._net]['class'](weights=weights, include_top=True, pooling='avg')
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer(self._nets[self._net]['layer_name']).output)

        x = base_model.output

        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='swish')(x)
        x = Dropout(0.5)(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation(activation='swish')(x)

        # Output layer
        predictions = Dense(3, activation="softmax")(x)

        base_model = Model(inputs=base_model.input, outputs=predictions)

        if self._use_regularization:
            base_model = self.add_regularization(base_model, regularizer=self._regularizer)

        if _output_path is not None:
            plot_model(base_model, to_file=_output_path + '/model_base_network.png', show_shapes=True, show_layer_names=True)

        return base_model

    def get_input_size(self):
        return self._nets[self._net]['input_shape']

    def get_net_name(self):
        return self._nets[self._net]['name']
