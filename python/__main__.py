#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-02 of 2020
"""
# For model definition/training
import keras
from keras.optimizers import Adam
from keras.utils import plot_model

# Packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for general purposes
import datetime
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import random
import time

# Custom imports
import models
from paths import Paths
import utils


# Set seed for reproducibility
seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)


# noinspection DuplicatedCode
def protocol(_network_class,
             _load_data_function,
             _batch_size=400,
             _epochs=200,
             _number_of_classes=3,
             _shuffle_in_training=True,
             _plot_loss_epochs=5,
             _lr=0.001,
             _train_portion=0.7,
             _model_epochs_checkpoint=200,
             _save_model=False,
             _use_data_augmentation=False,
             _limit_dataset=False,
             _custom_output_dir=None,
             _save_intermediate_models=False):

    _input_image_shape = _network_class.get_input_size()
    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = _load_data_function((_input_image_shape[0], _input_image_shape[1]), use_augmentaded_data=_use_data_augmentation, limit_dataset=_limit_dataset)

    _create_base_network = _network_class.create_net
    current_dt = datetime.datetime.now()
    if _custom_output_dir is not None and os.path.exists(Paths.RESULTS_PATH + 'results/' + _custom_output_dir):
        _custom_output_dir += '{}{:02d}{:02d}-{:02d}{:02d}{:02d}/'.format(current_dt.year,
                                                                          current_dt.month,
                                                                          current_dt.day,
                                                                          current_dt.hour,
                                                                          current_dt.minute,
                                                                          current_dt.second)

    if _custom_output_dir is not None:
        output_path = Paths.RESULTS_PATH + 'results/' + _custom_output_dir + '/'
    else:
        output_path = Paths.RESULTS_PATH + 'results/{}{:02d}{:02d}-{:02d}{:02d}{:02d}/'.format(current_dt.year,
                                                                                               current_dt.month,
                                                                                               current_dt.day,
                                                                                               current_dt.hour,
                                                                                               current_dt.minute,
                                                                                               current_dt.second)
    os.mkdir(output_path)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    x_train, y_train, x_val, y_val = utils.separate_data(x_train, y_train, _input_image_shape, (0, 1, 2), _train_portion)

    y_train = keras.utils.to_categorical(y_train.reshape(-1), num_classes=_number_of_classes)
    y_val = keras.utils.to_categorical(y_val.reshape(-1), num_classes=_number_of_classes)
    y_test = keras.utils.to_categorical(y_test.reshape(-1), num_classes=_number_of_classes)

    ####################################################################################################################
    # -- Starting training
    ####################################################################################################################
    model = _create_base_network(_number_of_classes=_number_of_classes, _output_path=output_path)
    print(model.summary())
    plot_model(model, to_file='{}/model.png'.format(output_path), show_shapes=True, show_layer_names=True)

    # train session
    opt = Adam(lr=_lr)  # choose optimiser. RMS is good too!
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1)]

    if _save_intermediate_models:
        filepath = "%s/model_semi_ep{epoch:02d}_BS%d.hdf5" % (output_path, _batch_size)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=_model_epochs_checkpoint)
        callbacks_list += [checkpoint]

    start = time.time()
    history_model = model.fit(
        x=x_train,
        y=y_train,
        batch_size=_batch_size,
        shuffle=_shuffle_in_training,
        epochs=_epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list
    )
    stop = time.time()

    if _save_model:
        model.save('{}/model_final.hdf5'.format(output_path), overwrite=True)

    print('Time spent to train: {}s'.format(stop - start))

    plt.figure(figsize=(8, 8))
    plt.plot(history_model.history['loss'], label='training loss')
    plt.plot(history_model.history['val_loss'], label='validation loss')
    plt.legend()
    plt.suptitle('Loss history after {} epochs'.format(_epochs))
    plt.savefig(output_path + 'loss{}.png'.format(_epochs))
    plt.show()

    ####################################################################################################################
    # -- Done training
    ####################################################################################################################

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64)
    print('\nTest loss: %.4f\naccuracy: %.4f' % (test_loss, test_accuracy))

    plt.close('all')
    test_pred = model.predict(x_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))
    ax = sns.heatmap(cm, cmap="binary", annot=True, fmt="d")
    print(cm)
    figure = ax.get_figure()
    figure.savefig('{}/result-heatmap.png'.format(output_path), dpi=600)


if __name__ == "__main__":

    protocol(
        _custom_output_dir='V19_DA1000_LIMIT',
        _use_data_augmentation=True,
        _limit_dataset=True,
        # _network_class=models.EfficientNet(net='b7',
        # _network_class=models.MobileNet(net='v1',
        # _network_class=models.VGGNet(net='19',
        _network_class=models.ResNet(net='50',
                                     use_weights=True,
                                     use_regularization=False,
                                     regularizer=keras.regularizers.l2(0.01)),
        _load_data_function=utils.load_covid_dataset,
        _number_of_classes=3,
        _batch_size=10,
        _epochs=20,
        _shuffle_in_training=True,
        _plot_loss_epochs=2,
        _lr=0.001,
        _train_portion=0.9,
        _model_epochs_checkpoint=200,
        _save_intermediate_models=False,
        _save_model=True)
