#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-05 of 2020
"""
# Packages for model definition/training
import keras

# Packages for general purposes
import numpy as np
from sklearn.metrics import confusion_matrix

# Packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Custom packages
from paths import Paths
import utils


def load_test_data(img_shape=(224, 224)):
    class_map = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

    dataset_path = Paths.DATASET_BASE_PATH + 'data/test/'
    csv_content = utils.read_txt(Paths.DATASET_BASE_PATH + 'COVID-Net/test_COVIDx.txt')

    _x_test_paths = []
    _y_test = []
    for c in csv_content:
        c = c.split(' ')
        _y_test.append(class_map[c[-1].replace('\n', '')])
        _x_test_paths.append(dataset_path + c[-2])

    dataset_path = '/media/share/pedro/2020-Covid/data/'
    csv_content = utils.read_txt('/media/share/pedro/2020-Covid/data/test.txt')

    for c in csv_content:
        _y_test.append(class_map['COVID-19'])
        _x_test_paths.append(dataset_path + c.replace('\n', ''))

    _y_test = np.asarray(_y_test)
    _x_test = utils.load_images(_x_test_paths, img_shape) / 255.

    return _x_test, _y_test


if __name__ == '__main__':

    x_test, y_test = load_test_data()
    y_test = keras.utils.to_categorical(y_test.reshape(-1), num_classes=3)

    model = keras.models.load_model('/media/share/pedro/2020-Covid/results/B0_DA1000_LIMIT/model_final.hdf5')
    print(model.summary())

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64)
    print('\nTest loss: %.4f\naccuracy: %.4f' % (test_loss, test_accuracy))

    plt.close('all')
    test_pred = model.predict(x_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))
    sns.set(font_scale=1.5)
    ax = sns.heatmap(cm, cmap="binary", annot=True, fmt="d")
    print(cm)
    figure = ax.get_figure()
    figure.savefig('./result-heatmap.png', dpi=600)
