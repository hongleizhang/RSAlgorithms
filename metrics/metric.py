# encoding:utf-8
import math


class Metric(object):
    '''
    the two metrics to measure the prediction accuracy for rating prediction task
    '''

    def __init__(self):
        pass

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return float(error) / count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(float(error) / count)
