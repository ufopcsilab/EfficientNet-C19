#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-05 of 2020
"""
from keras import applications as ka

from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout

# For model plot
from keras.utils import plot_model

# Base model definition
from models.base_model import BaseModel


# Functions
class VGGNet(BaseModel):
    """
    Base on: https://github.com/qubvel/efficientnet
    """

    def __init__(self, net, use_weights=True, custom_weights_path=None, use_regularization=False, regularizer=None):
        self._use_weights = use_weights
        self._custom_weights = custom_weights_path
        self._net = net
        self._nets = {
            '16': {'name': 'VGG-16', 'class': ka.vgg16.VGG16, 'layer_name': 'fc2', 'input_shape': (224, 224, 3)},
            '19': {'name': 'VGG-19', 'class': ka.vgg19.VGG19, 'layer_name': 'fc2', 'input_shape': (224, 224, 3)},
        }
        super(VGGNet, self).__init__(use_regularization, regularizer)

    def create_net(self,  _number_of_classes=3, _output_path=None):

        if self._use_weights:
            if self._custom_weights is None:
                weights = 'imagenet'
            else:
                weights = self._custom_weights
        else:
            weights = None

        base_model = self._nets[self._net]['class'](weights=weights, include_top=True)
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
