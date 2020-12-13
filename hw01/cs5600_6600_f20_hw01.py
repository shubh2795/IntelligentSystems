####################################################
# CS 5600/6600: F20: Assignment 1
# Shubham Swami
# A02315672
#####################################################

import numpy as np


class unit_step:

    def unitStep(v):
        if v > 0:
            return 1
        else:
            return 0


class and_percep:

    def __init__(self):
        self.w = np.array([1, 1])
        self.b = -1
        pass

    def output(self, x):
        v = np.dot(self.w, x) + self.b
        y = unit_step.unitStep(v)
        return y
        pass


class or_percep:

    def __init__(self):
        self.w = np.array([2, 2])
        self.b = -1
        pass

    def output(self, x):
        v = np.dot(self.w, x) + self.b
        y = unit_step.unitStep(v)
        return y
        pass


class not_percep:

    def __init__(self):
        self.w = -1
        self.b = 1
        pass

    def output(self, x):
        v = np.dot(self.w, x) + self.b
        y = unit_step.unitStep(v)
        return y
        pass


class xor_percep:

    def __init__(self):
        self.and_perceptron = and_percep()
        self.or_perceptron = or_percep()
        self.not_perceptron = not_percep()
        pass

    def output(self, x):
        y1 = self.and_perceptron.output(x)
        y2 = self.or_perceptron.output(x)
        y3 = self.not_perceptron.output(y1)
        y4 = np.array([y2, y3])
        return self.and_perceptron.output(y4)
        pass


class xor_percep2:
    def __init__(self):
        self.w1 = np.array([2, 2])
        self.w2 = np.array([-1, -1])
        self.w3 = np.array([1, 1])
        self.b0 = -1
        self.b1 = 2
        self.b2 = -1
        pass

    def threshold(self, x):
        v1 = np.dot(self.w1, x) + self.b0
        v2 = np.dot(self.w2, x) + self.b1
        y1 = unit_step.unitStep(v1)
        y2 = unit_step.unitStep(v2)
        return np.array([y1, y2])
        pass

    def output(self, x):
        v = np.dot(self.w3, self.threshold(x)) + self.b2
        y = unit_step.unitStep(v)
        return np.array([y])
        pass


class percep_net:

    def __init__(self):
        self.and_perceptron = and_percep()
        self.or_perceptron = or_percep()
        self.not_perceptron = not_percep()
        pass

    def output(self, x):
        y1 = self.or_perceptron.output(np.array([x[0], x[1]]))
        y2 = self.not_perceptron.output(np.array([x[2]]))
        y3 = self.and_perceptron.output(np.array([y1, y2]))
        y = self.or_perceptron.output(np.array([y3, x[3]]))
        return y
        pass
