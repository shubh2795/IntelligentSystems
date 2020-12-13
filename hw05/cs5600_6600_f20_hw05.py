#/usr/bin/python

###########################################
# module: cs5600_6600_f20_hw05.py
# Shubham Swami
# A02315672
###########################################

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

def make_tfl_convnet_slide22():
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer = conv_2d(input_layer, nb_filter=20,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer, 100,
                                 activation='sigmoid',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 10,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
    model = tflearn.DNN(network)
    return model

def load_tfl_convnet_slide22(model_path):
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer = conv_2d(input_layer, nb_filter=20,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer, 100,
                                 activation='sigmoid',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 10,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def make_shallow_tfl_ann():
    input_layer = input_data(shape=[None, 28, 28, 1])
    fc_layer_1 = fully_connected(input_layer, 20,
                                 activation='sigmoid',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 10,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
    model = tflearn.DNN(network)
    return model

def make_deeper_tfl_convnet():
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer = conv_2d(input_layer, nb_filter=20,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    conv_layer2 = conv_2d(pool_layer, nb_filter=40,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer2, 100,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_1 = dropout(fc_layer_1, 0.5)
    fc_layer_2 = fully_connected(fc_layer_1, 200,
                                 activation='sigmoid',
                                 name='fc_layer_2')
    fc_layer_2 = dropout(fc_layer_2, 0.5)
    fc_layer_3 = fully_connected(fc_layer_2, 10,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
    model = tflearn.DNN(network)
    return model

def load_deeper_tfl_convnet(model_path):
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer = conv_2d(input_layer, nb_filter=20,
                         filter_size=5,
                         activation='sigmoid',
                         name='conv_layer_1')
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    conv_layer2 = conv_2d(pool_layer, nb_filter=40,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer2, 100,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_1 = dropout(fc_layer_1, 0.5)
    fc_layer_2 = fully_connected(fc_layer_1, 200,
                                 activation='sigmoid',
                                 name='fc_layer_2')
    fc_layer_2 = dropout(fc_layer_2, 0.5)
    fc_layer_3 = fully_connected(fc_layer_2, 10,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model

def load_shallow_tfl_ann(model_path):
    input_layer = input_data(shape=[None, 28, 28, 1])
    fc_layer_1 = fully_connected(input_layer, 20,
                                 activation='sigmoid',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 10,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def fit_tfl_model(model, trainX, trainY, testX, testY, model_name, net_path, n_epoch=1, mbs=10):
    model.fit(trainX, trainY, n_epoch=n_epoch, shuffle=True,
              validation_set=(testX, testY),
              show_metric=True, batch_size=mbs,
              run_id='MNIST_ConvNet_1')
    model.save(net_path + model_name)

def test_tfl_model(model, X, Y):
    correctNumberOfPredictions = 0
    for i in range(0, len(X)):
        prediction = model.predict(X[i].reshape([-1, 28, 28, 1]))
        prediction = np.argmax(prediction, axis=1)[0]
        ground_truth = np.argmax(Y[i])
        if prediction == ground_truth:
            correctNumberOfPredictions += 1
    return correctNumberOfPredictions / len(X)