# encoding:utf-8
import sys

sys.path.append("..")
import os
from collections import defaultdict
import numpy as np

from utility.tools import normalize
from configx.configx import ConfigX


class RatingGetter(object):
    """
    docstring for RatingGetter
    read rating data and save the global parameters
    """

    def __init__(self, k):
        super(RatingGetter, self).__init__()
        self.config = ConfigX()
        self.k_current = k
        self.user = {}
        self.item = {}
        self.all_User = {}
        self.all_Item = {}
        self.id2user = {}
        self.id2item = {}
        self.dataSet_u = defaultdict(dict)
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)  # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = defaultdict(dict)  # used to store the test set by hierarchy item:[user,rating]
        self.testColdUserSet_u = defaultdict(dict)  # cold start users in test set
        self.trainHotUserSet = []  # hot users in train set
        self.trainSetLength = 0
        self.testSetLength = 0

        self.userMeans = {}  # used to store the mean values of users's ratings
        self.itemMeans = {}  # used to store the mean values of items's ratings
        self.globalMean = 0

        self.generate_data_set()  # generate train and test set
        self.getDataSet()
        self.get_data_statistics()
        self.get_cold_start_users()

    def generate_data_set(self):
        for index, line in enumerate(self.trainSet()):
            u, i, r = line
            # print(u,i,r)
            if not u in self.user:
                self.user[u] = len(self.user)
                self.id2user[self.user[u]] = u
            if not i in self.item:
                self.item[i] = len(self.item)
                self.id2item[self.item[i]] = i

            self.trainSet_u[u][i] = r
            self.trainSet_i[i][u] = r
            self.trainSetLength = index + 1
        self.all_User.update(self.user)
        self.all_Item.update(self.item)

        for index, line in enumerate(self.testSet()):
            u, i, r = line
            if not u in self.user:
                self.all_User[u] = len(self.all_User)
            if not i in self.item:
                self.all_Item[i] = len(self.all_Item)
            self.testSet_u[u][i] = r
            self.testSet_i[i][u] = r
            self.testSetLength = index + 1
        # print(self.trainSetLength)
        # print(self.testSetLength)
        pass

    # for cross validation
    def trainSet(self):
        k = self.k_current
        for i in range(self.config.k_fold_num):
            if i != k:
                data_path = self.config.rating_cv_path + self.config.dataset_name + "-" + str(i) + ".csv"
                # if not os.path.exists
                if not os.path.isfile(data_path):
                    print("the format of ratings data is wrong!")
                    sys.exit()
                with open(data_path, 'r') as f:
                    for index, line in enumerate(f):
                        u, i, r = line.strip('\r\n').split(self.config.sep)
                        r = normalize(float(r))  # scale the rating score to [0-1]
                        yield (int(float(u)), int(float(i)), float(r))

    def testSet(self):
        k = self.k_current
        data_path = self.config.rating_cv_path + self.config.dataset_name + "-" + str(k) + ".csv"
        if not os.path.isfile(data_path):
            print("the format of ratings data is wrong!")
            sys.exit()
        with open(data_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(self.config.sep)
                yield (int(float(u)), int(float(i)), float(r))

    # for random
    # def trainSet(self):
    #     np.random.seed(self.config.random_state)
    #     with open(self.config.rating_path,'r') as f:
    #         for index,line in enumerate(f):
    #             rand_num=np.random.rand()
    #             if  rand_num < self.config.size:
    #                 u,i,r=line.strip('\r\n').split(self.config.sep)
    #                 r=normalize(float(r)) #scale the rating score to [0-1]
    #                 yield (int(u),int(i),float(r))

    # def testSet(self):
    #     np.random.seed(self.config.random_state)
    #     with open(self.config.rating_path,'r') as f:
    #         for index,line in enumerate(f):
    #             rand_num=np.random.rand()
    #             if  rand_num >= self.config.size:
    #                 u,i,r=line.strip('\r\n').split(self.config.sep)
    #                 yield (int(u),int(i),float(r))

    def getDataSet(self):
        with open(self.config.rating_path, 'r') as f:
            for index, line in enumerate(f):
                u, i, r = line.strip('\r\n').split(self.config.sep)
                self.dataSet_u[int(u)][int(i)] = float(r)

    def get_train_size(self):
        return (len(self.user), len(self.item))

    # get cold start users in test set
    def get_cold_start_users(self):
        for user in self.testSet_u.keys():
            rating_length = len(self.trainSet_u[user])
            if rating_length <= self.config.coldUserRating:
                self.testColdUserSet_u[user] = self.testSet_u[user]
        # print('cold start users count', len(self.testColdUserSet_u))

    def get_data_statistics(self):

        total_rating = 0.0
        total_length = 0
        for u in self.user:
            u_total = sum(self.trainSet_u[u].values())
            u_length = len(self.trainSet_u[u])
            total_rating += u_total
            total_length += u_length
            self.userMeans[u] = u_total / float(u_length)

        for i in self.item:
            self.itemMeans[i] = sum(self.trainSet_i[i].values()) / float(len(self.trainSet_i[i]))

        if total_length == 0:
            self.globalMean = 0
        else:
            self.globalMean = total_rating / total_length

    def containsUser(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def containsItem(self, i):
        'whether item is in training set'
        if i in self.item:
            return True
        else:
            return False

    def containsUserItem(self, user, item):
        if user in self.trainSet_u:
            if item in self.trainSet_u[user]:
                # print(user)
                # print(item)
                # print(self.trainSet_u[user][item])
                return True
        return False

    def get_row(self, u):
        return self.trainSet_u[u]

    def get_col(self, c):
        return self.trainSet_i[c]

    def user_rated_items(self, u):
        return self.trainSet_u[u].keys()


if __name__ == '__main__':
    rg = RatingGetter(0)
    # for ind,entry in enumerate(rg.testSet()):
    # 	if ind<80:
    # 		print(entry)
    # # 		user,item,rating = entry

    # print(rg.trainSet_u[52])
    # print(rg.trainSet_u[10])
