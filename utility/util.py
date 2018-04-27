# encoding:utf-8
import sys

sys.path.append("..")

import pickle


def save_data(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))
    pass


def load_data(filename):
    f = open(filename, 'rb')
    model = pickle.load(f)
    print(filename + ' load data model finished.')
    return model
