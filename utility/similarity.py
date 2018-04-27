# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
from math import sqrt
from utility.tools import sigmoid_2


# x1,x2 is the form of np.array.

def euclidean(x1, x2):
    # find common ratings
    new_x1, new_x2 = common(x1, x2)
    # compute the euclidean between two vectors
    diff = new_x1 - new_x2
    denom = sqrt((diff.dot(diff)))
    try:
        return 1 / denom
    except ZeroDivisionError:
        return 0


def cosine(x1, x2):
    # find common ratings
    new_x1, new_x2 = common(x1, x2)
    # compute the cosine similarity between two vectors
    sum = new_x1.dot(new_x2)
    denom = sqrt(new_x1.dot(new_x1) * new_x2.dot(new_x2))
    try:
        return float(sum) / denom
    except ZeroDivisionError:
        return 0


def pearson(x1, x2):
    # find common ratings
    new_x1, new_x2 = common(x1, x2)
    # compute the pearson similarity between two vectors
    ind1 = new_x1 > 0
    ind2 = new_x2 > 0
    try:
        mean_x1 = float(new_x1.sum()) / ind1.sum()
        mean_x2 = float(new_x2.sum()) / ind2.sum()
        new_x1 = new_x1 - mean_x1
        new_x2 = new_x2 - mean_x2
        sum = new_x1.dot(new_x2)
        denom = sqrt((new_x1.dot(new_x1)) * (new_x2.dot(new_x2)))
        return float(sum) / denom
    except ZeroDivisionError:
        return 0


def common(x1, x2):
    # find common ratings
    common = (x1 != 0) & (x2 != 0)
    new_x1 = x1[common]
    new_x2 = x2[common]
    return new_x1, new_x2


# x1,x2 is the form of dict.

def cosine_sp(x1, x2):
    'x1,x2 are dicts,this version is for sparse representation'
    total = 0
    denom1 = 0
    denom2 = 0
    # x1_l,x2_l=len(x1),len(x2)
    # if x2_l>x1_l:
    # x1,x2=x2,x1
    for k in x1:
        if k in x2:
            total += x1[k] * x2[k]
            denom1 += x1[k] ** 2
            denom2 += x2[k] ** 2  # .pop(k)
        # else:
        # denom1+=x1[k]**2
    # for j in x2:
    # 	denom2+=x2[j]**2
    try:
        return (total + 0.0) / (sqrt(denom1) * sqrt(denom2))
    except ZeroDivisionError:
        return 0


def cosine_improved_sp(x1, x2):
    'x1,x2 are dicts,this version is for sparse representation'
    total = 0
    denom1 = 0
    denom2 = 0
    nu = 0
    for k in x1:
        if k in x2:
            nu += 1
            total += x1[k] * x2[k]
            denom1 += x1[k] ** 2
            denom2 += x2[k] ** 2
    try:
        return (total + 0.0) / (sqrt(denom1) * sqrt(denom2)) * sigmoid_2(nu)
    except ZeroDivisionError:
        return 0


# def pearson_sp(x1, x2):
#     total = 0
#     denom1 = 0
#     denom2 = 0
#     try:
#         mean1 = sum(x1.values()) / (len(x1) + 0.0)
#         mean2 = sum(x2.values()) / (len(x2) + 0.0)
#         for k in x1:
#             if k in x2:
#                 total += (x1[k] - mean1) * (x2[k] - mean2)
#                 denom1 += (x1[k] - mean1) ** 2
#                 denom2 += (x2[k] - mean2) ** 2
#         return (total + 0.0) / (sqrt(denom1) * sqrt(denom2))
#     except ZeroDivisionError:
#         return 0

# improved pearson
def pearson_sp(x1, x2):
    common = set(x1.keys()) & set(x2.keys())
    if len(common) == 0:
        return 0
    ratingList1 = []
    ratingList2 = []
    for i in common:
        ratingList1.append(x1[i])
        ratingList2.append(x2[i])
    if len(ratingList1) == 0 or len(ratingList2) == 0:
        return 0
    avg1 = sum(ratingList1) / len(ratingList1)
    avg2 = sum(ratingList2) / len(ratingList2)
    mult = 0.0
    sum1 = 0.0
    sum2 = 0.0
    for i in range(len(ratingList1)):
        mult += (ratingList1[i] - avg1) * (ratingList2[i] - avg2)
        sum1 += pow(ratingList1[i] - avg1, 2)
        sum2 += pow(ratingList2[i] - avg2, 2)
    if sum1 == 0 or sum2 == 0:
        return 0
    return mult / (sqrt(sum1) * sqrt(sum2))


# TrustWalker userd
def pearson_improved_sp(x1, x2):
    total = 0.0
    denom1 = 0
    denom2 = 0
    nu = 0
    try:
        mean1 = sum(x1.values()) / (len(x1) + 0.0)
        mean2 = sum(x2.values()) / (len(x2) + 0.0)
        for k in x1:
            if k in x2:
                # print('k'+str(k))
                nu += 1
                total += (x1[k] - mean1) * (x2[k] - mean2)
                # print('t'+str(total))
                denom1 += (x1[k] - mean1) ** 2
                denom2 += (x2[k] - mean2) ** 2
        # print('nu:'+str(nu))
        # print(total)
        return (total + 0.0) / (sqrt(denom1) * sqrt(denom2)) * sigmoid_2(nu)
    except ZeroDivisionError:
        return 0


def euclidean_sp(x1, x2):
    total = 0.0
    for k in x1:
        if k in x2:
            total += sqrt(x1[k] - x2[k])
    try:
        return 1.0 / total
    except ZeroDivisionError:
        return 0
