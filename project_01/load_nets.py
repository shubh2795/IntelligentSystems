########################################################
# module: load_nets.py
# Shubham Swami
# A02315672
# descrption: starter code for loading your project 1 nets.
########################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression


### ======================= ANNs ===========================

def load_ann_audio_model_buzz1(model_path):
    # Validation Accuracy: 0.577391304347826
    # Validation DataSet: BUZZ1_valid_X, BUZZ1_valid_Y
    input_layer = input_data(shape=[None, 4000, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def load_ann_audio_model_buzz2(model_path):
    # Validation Accuracy: 0.5863333333333334
    # Validation DataSet: BUZZ2_valid_X, BUZZ2_valid_Y
    input_layer = input_data(shape=[None, 4000, 1, 1])

    fc_layer_1 = fully_connected(input_layer, 1024, activation='relu', name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 256, activation='relu', name='fc_layer_2')

    fc_layer_3 = fully_connected(fc_layer_2, 64, activation='relu', name='fc_layer_3')

    fc_layer_4 = fully_connected(fc_layer_3, 32, activation='relu', name='fc_layer_4')

    fc_layer_5 = fully_connected(fc_layer_4, 8, activation='relu', name='fc_layer_5')

    fc_layer_6 = fully_connected(fc_layer_5, 3, activation='softmax', name='fc_layer_6')
    model = tflearn.DNN(fc_layer_6)
    model.load(model_path)
    return model

def load_ann_audio_model_buzz3(model_path):
    # Validation Accuracy: 0.7559863169897377
    # Validation DataSet: BUZZ3_valid_X, BUZZ3_valid_Y
    input_layer = input_data(shape=[None, 4000, 1, 1])

    fc_layer_1 = fully_connected(input_layer, 128, activation='relu', name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 32, activation='relu', name='fc_layer_2')

    fc_layer_3 = fully_connected(fc_layer_2, 3, activation='softmax', name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model


def load_ann_image_model_bee1_gray(model_path):
    # Validation Accuracy: 0.9143990929705216
    # Validation Dataset: BEE1_gray_valid_X, BEE1_gray_valid_Y
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


def load_ann_image_model_bee2_1s_gray(model_path):
    # Validation Accuracy: 0.8497811017876687
    # Validation dataset: BEE2_1S_gray_valid_X, BEE2_1S_gray_valid_Y
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


def load_ann_image_model_bee4_gray(model_path):
    # Validation Accuracy: 0.7862524703557312
    # Validation dataset:BEE4_gray_valid_X, BEE4_gray_valid_Y
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


### ======================= ConvNets ===========================

def load_cnn_audio_model_buzz1(model_path):
    # Validation Accuracy: 0.7260869565217392
    # Validation Data: BUZZ1_valid_X, BUZZ1_valid_Y
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=64,
                           filter_size=10,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 4, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=16,
                           filter_size=4,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 4, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 512,
                                 activation='relu',
                                 name='fc_layer_1')

    d1 = dropout(fc_layer_1, 0.5)

    fc_layer_2 = fully_connected(d1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    d2 = dropout(fc_layer_2, 0.5)
    fc_layer_3 = fully_connected(d2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model


def load_cnn_audio_model_buzz2(model_path):
    # Validation Accuracy: 0.5853333333333334
    # Validation Data: BUZZ2_valid_X, BUZZ2_valid_Y
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=64,
                           filter_size=10,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 4, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=16,
                           filter_size=4,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 4, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 512,
                                 activation='relu',
                                 name='fc_layer_1')

    d1 = dropout(fc_layer_1, 0.5)

    fc_layer_2 = fully_connected(d1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    d2 = dropout(fc_layer_2, 0.5)
    fc_layer_3 = fully_connected(d2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model


def load_cnn_audio_model_buzz3(model_path):
    # Validation Accuracy: 0.9002280501710376
    # Validation Data: BUZZ3_valid_X, BUZZ3_valid_Y
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=64,
                           filter_size=10,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 4, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=16,
                           filter_size=4,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 4, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer_2, 512,
                                 activation='relu',
                                 name='fc_layer_1')

    d1 = dropout(fc_layer_1, 0.5)

    fc_layer_2 = fully_connected(d1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    d2 = dropout(fc_layer_2, 0.5)
    fc_layer_3 = fully_connected(d2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model


def load_cnn_image_model_bee1(model_path):
    # Validation Accuracy: 0.9651360544217688
    # Validation Data: BEE1_valid_X, BEE1_valid_Y
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


def load_cnn_image_model_bee2_1s(model_path):
    # Validation Accuracy: 0.8471360817219993
    # Validation Data: BEE2_1S_valid_X, BEE2_1S_valid_Y
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


def load_cnn_image_model_bee4(model_path):
    # Validation Accuracy: 0.751173418972332
    # Validation Data: BEE4_valid_X, BEE4_valid_Y
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

#
# ann_image_model_bee1_gray.tfl         validation accuracy:0.914399093             # Validation DataSet: BEE1_gray_valid_X, BEE1_gray_valid_Y
# ann_image_model_bee2_1s_gray.tfl      validation accuracy:  0.849781102           # Validation DataSet: BEE2_1s_gray_valid_X, BEE2_1s_gray_valid_Y
# ann_image_model_bee4_gray.tfl         validation accuracy: 0.78625247             # Validation DataSet: BEE4_gray_valid_X, BEE4_gray_valid_Y
# ann_audio_model_buzz3.tfl             validation accuracy:0.7559                  # Validation DataSet: BUZZ3_valid_X, BUZZ3_valid_Y
# ann_audio_model_buzz2.tfl             validation accuracy:0.586                   # Validation DataSet: BUZZ2_valid_X, BUZZ2_valid_Y
# ann_audio_model_buzz1.tfl             validation accuracy:0.577                   # Validation DataSet: BUZZ1_valid_X, BUZZ1_valid_Y

# cnn_image_model_bee1.tfl              validation accuracy: 0.965136054            # Validation DataSet: BEE1_valid_X, BEE1_valid_Y
# cnn_image_model_bee4.tfl              validation accuracy: 0.751173419            # Validation DataSet: BEE4_valid_X, BEE4_valid_Y
# cnn_image_model_bee2_1s.tfl           validation accuracy: 0.847136082            # Validation DataSet: BEE2_1s_valid_X, BEE2_1s_valid_Y
# cnn_audio_model_buzz1.tfl             validation accuracy:0.72608                 # Validation DataSet: BUZZ1_valid_X, BUZZ1_valid_Y
# cnn_audio_model_buzz2.tfl             validation accuracy:0.58                    # Validation DataSet: BUZZ2_valid_X, BUZZ2_valid_Y
# cnn_audio_model_buzz3.tfl             validation accuracy:0.9022                  # Validation DataSet: BUZZ3_valid_X, BUZZ3_valid_Y

