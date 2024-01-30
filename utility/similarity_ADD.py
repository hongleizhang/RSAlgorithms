# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
from math import sqrt
from utility.tools import sigmoid_2

# x1,x2 is the form of np.array.

def jaccard(x1, x2):
    num = 0
    den = 0
    for k in x1:
        if k in x2:
            num += 1
            den += 1
        else:
            den += 2
    return num / den
