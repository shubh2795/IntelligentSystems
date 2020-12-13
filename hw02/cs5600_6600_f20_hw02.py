#!/usr/bin/python

#########################################
# module: cs5600_6600_f20_hw02.py
# Shubham Swami
# A02315672
#########################################

from numpy import *
import pickle
from cs5600_6600_f20_hw02_data import *


# sigmoid function and its derivative.
# you'll use them in the training and fitting
# functions below.
def sigmoidf(x):
    return 1 / (1 + exp(-x))


def sigmoidf_prime(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
    # for bool i used approximation i.e x(1-x) here


# save() function to save the trained network to a file
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)


# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def build_nn_wmats(a):
    np.random.seed(0)
    arr = []
    for i in range(len(a) - 1):
        arr.append(np.random.randn(a[i], a[i + 1]))

    return arr


def build_231_nn():
    return build_nn_wmats((2, 3, 1))


def build_2331_nn():
    return build_nn_wmats((2, 3, 3, 1))


def build_221_nn():
    return build_nn_wmats((2, 2, 1))


def build_838_nn():
    return build_nn_wmats((8, 3, 8))


def build_949_nn():
    return build_nn_wmats((9, 4, 9))


def build_4221_nn():
    return build_nn_wmats((4, 2, 2, 1))


def build_421_nn():
    return build_nn_wmats((4, 2, 1))


def build_121_nn():
    return build_nn_wmats((1, 2, 1))


def build_1221_nn():
    return build_nn_wmats((1, 2, 2, 1))


## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    weights = build()
    for _ in range(numIters):
        # Feed Forward
        Z2 = dot(X, weights[0])
        a2 = sigmoidf(Z2)
        Z3 = dot(a2, weights[1])
        y_hat = sigmoidf(Z3)

        # Back propagation
        y_error = (y - y_hat)
        y_hat_delta = y_error * sigmoidf_prime(y_hat)
        a2_error = y_hat_delta.dot(weights[1].T)
        a2_delta = a2_error * sigmoidf_prime(a2)
        weights[1] += a2.T.dot(y_hat_delta)
        weights[0] += X.T.dot(a2_delta)
    return weights


def train_4_layer_nn(numIters, X, y, build):
    weights = build()
    for _ in range(numIters):
        # feed forward
        l1 = sigmoidf(np.dot(X, weights[0]))
        l2 = sigmoidf(np.dot(l1, weights[1]))
        l3 = sigmoidf(np.dot(l2, weights[2]))

        # Backpropagation

        delta_l3 = (y - l3) * sigmoidf_prime(l3)
        delta_l2 = delta_l3.dot(weights[2].T) * sigmoidf_prime(l2)
        delta_l1 = delta_l2.dot(weights[1].T) * sigmoidf_prime(l1)
        # update weights
        weights[2] += (np.dot(l2.T, delta_l3))
        weights[1] += (np.dot(l1.T, delta_l2))
        weights[0] += (np.dot(X.T, delta_l1))

    return weights


def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    W1, W2 = wmats[0], wmats[1]
    Z2 = np.dot(x, W1)
    a2 = sigmoidf(Z2)
    Z3 = np.dot(a2, W2)
    y_hat = sigmoidf(Z3)
    if (thresh_flag):
        if (y_hat > thresh):
            return np.array([1])
        return np.array([0])
    return np.array([y_hat])


def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    W1, W2, W3 = wmats[0], wmats[1], wmats[2]
    Z2 = np.dot(x, W1)
    a2 = sigmoidf(Z2)
    Z3 = np.dot(a2, W2)
    a3 = sigmoidf(Z3)
    Z4 = np.dot(a3, W3)
    y_hat = sigmoidf(Z4)
    if (thresh_flag):
        if (y_hat > thresh):
            return np.array([1])
        return np.array([0])
    return np.array([y_hat])


#     Network                     Structure               NumberOfIterations              Threshold             Seed
# 1. and_3_layer_ann.pck            2x3x1                       250                         0.44                  1
# 2. and_4_layer_ann.pck            2x3x3x1                     100                         0.36                  1
# 3. or_3_layer_ann.pck             2x3x1                       500                         0.4                   1
# 4. or_4_layer_ann.pck             2x3x3x1                     500                         0.4                   1
# 5. not_3_layer_ann.pck            1x2x1                       500                         0.4                   1
# 6. not_4_layer_ann.pck            1x2x2x1                     500                         0.4                   1
# 7. xor_3_layer_ann.pck            2x3x1                       700                         0.4                   0
# 8. xor_4_layer_ann.pck            2x3x3x1                     700                         0.4                   0
# 9. bool_3_layer_ann.pck           4x2x1                       150                         0.35                  0
# 10. bool_4_layer_ann.pck          4x2x2x1                     150                         0.5                   0



#Is it really true that the deeper we go, the faster it trains?
# As far as i understand The number of layers in a model is referred to as its depth.
# Increasing the depth increases the capacity of the model. Training deep models, those with many hidden layers,
# can be computationally more efficient than training a single layer network with a vast number of nodes.

# while executing the test cases I found that the number of iterations reduced for some cases like and_4_layer_ann and
# where as in some cases it took nearly the same iterations for both 3 and 4 layer Ann's
