# encoding:utf-8
import sys

sys.path.append("..")

import numpy as np
from numpy.linalg import norm
from configx.configx import ConfigX

config = ConfigX()


def l1(x):
    return norm(x, ord=1)


def l2(x):
    return norm(x)


def normalize(rating, minVal=config.min_val, maxVal=config.max_val):
    'get the normalized value using min-max normalization'
    if maxVal > minVal:
        return float(rating - minVal) / (maxVal - minVal) + 0.01
    elif maxVal == minVal:
        return rating / maxVal
    else:
        print('error... maximum value is less than minimum value.')
        raise ArithmeticError


def denormalize(rating, minVal=config.min_val, maxVal=config.max_val):
    return minVal + (rating - 0.01) * (maxVal - minVal)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def sigmoid_2(z):
    return 1.0 / (1.0 + np.exp(-z / 2.0))
