#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-03 of 2020
"""
# Packages for general purposes
import shutil
import os

# Packages to handle images
import Augmentor as Augmentor
import cv2
import pydicom

# Custom packages
import utils
from paths import Paths


def convert_dcm_to_png(data_type):

    inputdir = Paths.DATASET_BASE_PATH + 'rsna-pneumonia-detection-challenge/stage_2_{}_images/'.format(data_type)
    outdir = Paths.DATASET_BASE_PATH + 'data/{}/'.format(data_type)
    test_list = [f for f in os.listdir(inputdir)]

    for f in test_list:
        print('\t{}'.format(inputdir + f))
        ds = pydicom.read_file(inputdir + f)  # read dicom image
        img = ds.pixel_array  # get image array
        print('\t\t{}'.format(outdir + f))
        cv2.imwrite(outdir + f.replace('.dcm', '.png'), img)  # write png image


def copy_missing_files(data_type):
    output_path = Paths.DATASET_BASE_PATH + 'data/{}/'.format(data_type)
    train_dataset_path = [
        Paths.DATASET_BASE_PATH + 'covid-chestxray-dataset/images/',
        Paths.DATASET_BASE_PATH + 'rsna-pneumonia-detection-challenge/stage_2_{}_images/'.format(data_type)
    ]

    csv_content = utils.read_txt(Paths.DATASET_BASE_PATH + 'COVID-Net/{}_COVIDx.txt'.format(data_type))

    _x_train_paths = []
    for c in csv_content:
        full_path = None
        img_path = c.split(' ')[-2]
        if not img_path.endswith('.dcm'):
            if os.path.exists(train_dataset_path[0] + img_path):
                full_path = train_dataset_path[0] + img_path
            elif os.path.exists(train_dataset_path[1] + img_path):
                full_path = train_dataset_path[1] + img_path
            if full_path is not None:
                img = cv2.imread(full_path)
                cv2.imwrite(output_path + img_path, img)  # write png image


def check_if_all_files_exist(data_type):
    csv_content = utils.read_txt(Paths.DATASET_BASE_PATH + 'COVID-Net/{}_COVIDx.txt'.format(data_type))
    imgs_path = Paths.DATASET_BASE_PATH + 'data/{}/'.format(data_type)

    for c in csv_content:
        img_path = c.split(' ')[-2]
        if not os.path.exists(imgs_path + img_path):
            print('The following image was not found [{}].'.format(img_path))
            shutil.move(imgs_path.replace('/test/', '/train/') + img_path, imgs_path + img_path)


def copy_covid_dataset():
    output_path = Paths.DATASET_BASE_PATH + 'data/augmented-orig/'
    dataset_path = Paths.DATASET_BASE_PATH + 'data/train/'

    for c in utils.read_txt(Paths.DATASET_BASE_PATH + 'COVID-Net/train_COVIDx.txt'):
        c = c.split(' ')
        if 'COVID-19' in c[-1]:
            img = cv2.imread(dataset_path + c[-2], cv2.IMREAD_COLOR)
            cv2.imwrite(output_path + c[-2], img)  # write png image


def augmentate_data_from_path():
    copy_covid_dataset()

    input_path = Paths.DATASET_BASE_PATH + 'data/augmented-orig/'
    output_path = Paths.DATASET_BASE_PATH + 'data/augmented/'

    p = Augmentor.Pipeline(
        source_directory=input_path,
        output_directory=output_path)
    p.rotate(probability=1, max_left_rotation=0.15, max_right_rotation=0.15)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.2, percentage_area=0.8)
    p.status()

    p.sample(1000)
    p.process()


if __name__ == '__main__':
    print('Converting training images')
    convert_dcm_to_png('train')
    print('Converting testing images')
    convert_dcm_to_png('test')

    print('Copying missing training images')
    copy_missing_files('train')
    print('Copying missing testing images')
    copy_missing_files('test')

    print('\n', '*' * 200)
    print('Train')
    check_if_all_files_exist('train')
    print('\n', '*' * 200)
    print('Test')
    check_if_all_files_exist('test')

    print('\n', '*' * 200)
    print('\nData augmentation')
    augmentate_data_from_path()
