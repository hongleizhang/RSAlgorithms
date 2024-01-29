# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
from math import sqrt
from utility.tools import sigmoid_2


# x1,x2 is the form of np.array.

def betweenness_centrality(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    if len(x1) == 0 or len(x2) == 0:
        return 0
    else:
        return np.sum(np.abs(x1 - x2)) / (len(x1) * len(x2))