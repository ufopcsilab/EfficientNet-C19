#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: fev-11 of 2020
"""
# Packages for general purposes
import numpy as np
import os

# Handle images
import cv2

# For visualizing
from paths import Paths
from progress.bar import IncrementalBar


def load_covid_dataset(img_shape=(224, 224),
                       use_augmentaded_data=False,
                       limit_dataset=False):
    class_map = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

    dataset_path = Paths.DATASET_BASE_PATH + 'data/train/'
    csv_content = read_txt(Paths.DATASET_BASE_PATH + 'COVID-Net/train_COVIDx.txt')

    _x_train_paths = []
    _y_train = []
    for c in csv_content:
        c = c.split(' ')
        _y_train.append(class_map[c[-1].replace('\n', '')])
        _x_train_paths.append(dataset_path + c[-2])

    if use_augmentaded_data:
        files = get_all_files_names(Paths.DATASET_BASE_PATH + 'data/augmented/', '')
        _x_train_paths += files
        _y_train += [class_map['COVID-19']] * len(files)

    _y_train = np.asarray(_y_train)

    if limit_dataset:
        number_of_samples = min([len(_y_train[_y_train == 0]), len(_y_train[_y_train == 1]), len(_y_train[_y_train == 2])])

        _x_train_paths = (
            [x for x, y in zip(_x_train_paths, _y_train == 0) if y][:number_of_samples] +
            [x for x, y in zip(_x_train_paths, _y_train == 1) if y][:number_of_samples] +
            [x for x, y in zip(_x_train_paths, _y_train == 2) if y][:number_of_samples]
        )

        _y_train = np.hstack((
            _y_train[_y_train == 0][:number_of_samples],
            np.hstack((
                _y_train[_y_train == 1][:number_of_samples],
                _y_train[_y_train == 2][:number_of_samples]
            ))
        ))

    _x_train = load_images(_x_train_paths, img_shape)

    dataset_path = Paths.DATASET_BASE_PATH + 'data/test/'
    csv_content = read_txt(Paths.DATASET_BASE_PATH + 'COVID-Net/test_COVIDx.txt')

    _x_test_paths = []
    _y_test = []
    for c in csv_content:
        c = c.split(' ')
        _y_test.append(class_map[c[-1].replace('\n', '')])
        _x_test_paths.append(dataset_path + c[-2])

    _y_test = np.asarray(_y_test)
    _x_test = load_images(_x_test_paths, img_shape)

    return (_x_train, _y_train), (_x_test, _y_test)


def load_images(file_names, img_shape):
    imgs = []
    bar = IncrementalBar('Countdown', max=len(file_names))
    for f in file_names:
        bar.next()
        imgs.append(cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), img_shape, interpolation=cv2.INTER_AREA).reshape(-1, img_shape[0], img_shape[1], 3))
    bar.finish()
    return np.asarray(imgs).reshape(shape=(-1, img_shape[0], img_shape[1], 3))


def read_txt(file_name):
    with open(file_name) as f:
        return f.readlines()


def get_all_files_names(dir_name, extension):
    list_of_file = os.listdir(dir_name)
    all_files = list()
    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_all_files_names(full_path, extension)
        elif full_path.endswith(extension):
            all_files.append(full_path)
    return all_files


def separate_data(_x_train, _y_train, _input_image_shape, _classes, _train_portion):
    x_val, y_val = [], []
    px_train, py_train = [], []
    for i in range(_y_train.shape[0]):
        if i % 8 == 0:
            x_val.append(_x_train[i, :, :, :])
            y_val.append(_y_train[i])
        else:
            px_train.append(_x_train[i, :, :, :])
            py_train.append(_y_train[i])

    x_val = np.asarray(x_val).reshape(shape=(-1, _input_image_shape[0], _input_image_shape[1], _input_image_shape[2]))
    y_val = np.asarray(y_val)

    x_train = np.asarray(px_train).reshape(shape=(-1, _input_image_shape[0], _input_image_shape[1], _input_image_shape[2]))
    y_train = np.asarray(py_train)
    del px_train, py_train
    return x_train, y_train, x_val, y_val
