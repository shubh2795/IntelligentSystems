# /usr/bin/python

############################
# Module: cs5600_6600_f20_hw03.py
# Shubham Swami
# A02315672
############################
from ann import *
from mnist_loader import load_data_wrapper
import random
import pickle as cPickle
import numpy as np
import os

# load training, validation, and testing MNIST data
train_d, valid_d, test_d = load_data_wrapper()


def train_1_hidden_layer_anns(lwr=10, upr=50, eta=0.25, mini_batch_size=10, num_epochs=10):
    hidden_layer_size = lwr
    while (hidden_layer_size <= upr):
        nn = ann([784, hidden_layer_size, 10])
        print("==== Training 784x" + str(hidden_layer_size) + "x10 ANN ======")
        nn.mini_batch_sgd(train_d, num_epochs, mini_batch_size, eta, test_data=test_d)
        print("==== Training 784x" + str(hidden_layer_size) + "x10 ANN DONE... ======")
        hidden_layer_size = hidden_layer_size + 10


def train_2_hidden_layer_anns(lwr=10, upr=50, eta=0.25, mini_batch_size=10, num_epochs=10):
    hidden_layer_size = lwr
    while (hidden_layer_size <= upr):
        HL2S = lwr
        while (HL2S <= upr):
            nn = ann([784, hidden_layer_size, HL2S, 10])
            print("==== Training 784x" + str(hidden_layer_size) + "x" + str(HL2S) + "x10 ANN ======")
            nn.mini_batch_sgd(train_d, num_epochs, mini_batch_size, eta, test_data=test_d)
            print("==== Training 784x" + str(hidden_layer_size) + "x" + str(HL2S) + "x10 ANN DONE... ======")
            HL2S = HL2S + 10
        hidden_layer_size = hidden_layer_size + 10


# define your networks
net1 = ann([784, 20, 30, 50, 40, 20, 30, 10])
net2 = ann([784, 20, 50, 40, 30, 20, 10])
net3 = ann([784, 20, 50, 40, 10])
net4 = ann([784, 20, 40, 10])
net5 = ann([784, 30, 10])

# define an ensemble of 5 nets
networks = (net1, net2, net3, net4, net5)
eta_vals = (0.1, 0.25, 0.3, 0.4, 0.5)
mini_batch_sizes = (5, 10, 15, 20)


# train networks
def train_nets(networks, eta_vals, mini_batch_sizes, num_epochs, path):
    for n in networks:
        s = "\\net"
        for nodes in n.sizes:
            s = s + "_" + str(nodes)
        eta = random.choice(eta_vals)
        min_batch_size = random.choice(mini_batch_sizes)
        s = s + "_" + str(int(eta * 100)) + "_" + str(min_batch_size) + ".pck"
        print("==== Training ANN " + str(n.sizes) + "eta= " + str(eta) + " minBatchSize = " + str(
            min_batch_size) + " ======")
        n.mini_batch_sgd(train_d, num_epochs, min_batch_size, eta, test_data=test_d)
        print("==== Training ANN " + str(n.sizes) + "eta= " + str(eta) + " minBatchSize = " + str(
            min_batch_size) + " Done... ======")
        save(n, path + s)


def load_nets(path):
    listOfNetworks = []
    for filename in os.listdir(path):
        if (filename.endswith(".pck")):
            net = load(path + "\\" + filename)
            tupleOfNet = (filename, net)
            listOfNetworks.append(tupleOfNet)
    return listOfNetworks


# evaluate net ensemble.
def evaluate_net_ensemble(net_ensemble, test_data):
    out = 0
    for (x, y) in test_data:
        votes = np.zeros(10, dtype=int)
        for n in net_ensemble:
            index = np.argmax(n.feedforward(x))
            votes[index] += 1
        if (np.argmax(votes) == y):
            out += 1
    return (out, len(test_data))


# save() function to save the trained network to a file
##code of last assignment
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(obj, fp)


# restore() function to restore the file
# code of last assignment
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = cPickle.load(fp)
    return obj
