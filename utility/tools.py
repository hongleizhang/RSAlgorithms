#encoding:utf-8


import numpy as np
from numpy.linalg import norm
from configx.configx import ConfigX

config=ConfigX()



def l1(x):
	return norm(x,ord=1)

def l2(x):
	return norm(x)

def normalize(rating,minVal=config.min_val,maxVal=config.max_val):
    'get the normalized value using min-max normalization'
    if maxVal > minVal:
        return float(rating-minVal)/(maxVal-minVal)+0.01
    elif maxVal==minVal:
        return rating/maxVal
    else:
        print('error... maximum value is less than minimum value.')
        raise ArithmeticError


def denormalize(rating,minVal=config.min_val,maxVal=config.max_val):
    return minVal+(rating-0.01)*(maxVal-minVal)
