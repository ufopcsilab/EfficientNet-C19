#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-02 of 2020
"""
# Imports for defining models
import keras

# Imports for general purposes
import os
import tempfile


# noinspection PyMethodMayBeStatic
class BaseModel(object):

    def __init__(self, use_regularization=False, regularizer=None):
        self._use_regularization = use_regularization
        self._regularizer = regularizer

    def create_net(self, _number_of_classes=2, _output_path=None):
        raise NotImplementedError('You need to implement the create_net function.')

    def get_input_size(self):
        raise NotImplementedError('You need to implement the get_input_size function.')

    def get_net_name(self):
        raise NotImplementedError('You need to implement the get_net_name function.')

    def is_using_regularization(self):
        return self._use_regularization

    def add_regularization(self, model, regularizer=keras.regularizers.l2(0.0001)):
        if not isinstance(regularizer, keras.regularizers.Regularizer):
            print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
            return model

        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = keras.models.model_from_json(model_json)

        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)
        return model
