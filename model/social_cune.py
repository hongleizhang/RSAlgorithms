# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
from collections import defaultdict
from random import randint
from random import shuffle,choice
from math import log
import gensim.models.word2vec as w2v
from prettyprinter import cpprint

from mf import MF
from reader.trust import TrustGetter
from utility.matrix import SimMatrix
from utility.similarity import cosine
from utility import util


class CUNE(MF):
    """
    docstring for CUNE

    Zhang et al. Collaborative User Network Embedding for Social Recommender Systems. SDM
    """

    def __init__(self):
        super(CUNE, self).__init__()
        self.config.lambdaP = 0.01
        self.config.lambdaQ = 0.01
        self.config.alpha = 0.01
        self.config.isEarlyStopping = True
        self.tg = TrustGetter()
        self.config.walkCount = 30
        self.config.walkLength = 20
        self.config.walkDim = 20
        self.config.winSize = 5
        self.config.topK = 50

    def init_model(self, k):
        super(CUNE, self).init_model(k)
        self.user_sim = SimMatrix()
        self.generate_cu_net()
        self.deep_walk()
        self.compute_social_sim()

    def generate_cu_net(self):
        print('Building collaborative user network...')
        itemNet = {}
        for item in self.rg.trainSet_i:
            if len(self.rg.trainSet_i[item])>1:
                itemNet[item] = self.rg.trainSet_i[item]

        filteredRatings = defaultdict(list)
        for item in itemNet:
            for user in itemNet[item]:
                if itemNet[item][user] > 0:
                    filteredRatings[user].append(item)

        self.CUNet = defaultdict(list)

        for user1 in filteredRatings:
            s1 = set(filteredRatings[user1])
            for user2 in filteredRatings:
                if user1 != user2:
                    s2 = set(filteredRatings[user2])
                    weight = len(s1.intersection(s2))
                    if weight > 0:
                        self.CUNet[user1]+=[user2] # * weight

        # cpprint(self.CUNet)
        pass
    def deep_walk(self):
        print('Generating random deep walks...')
        self.walks = []
        self.visited = defaultdict(dict)
        for user in self.CUNet:
            for t in range(self.config.walkCount):
                path = [str(user)]
                lastNode = user
                for i in range(1,self.config.walkLength):
                    nextNode = choice(self.CUNet[lastNode])
                    count=0
                    while(nextNode in self.visited[lastNode]):
                        nextNode = choice(self.CUNet[lastNode])
                        #break infinite loop
                        count+=1
                        if count==self.config.walkLength: # 10
                            break
                    path.append(str(nextNode))
                    self.visited[user][nextNode] = 1
                    lastNode = nextNode
                self.walks.append(path)
        # shuffle(self.walks)
        # cpprint(self.walks)
        print('Generating user embedding...')
        self.model = w2v.Word2Vec(self.walks, size=self.config.walkDim, window=5, min_count=0, iter=3)
        print('User embedding generated.')
        pass

    def compute_social_sim(self):
        print('Constructing similarity matrix...')
        # self.W = np.zeros((self.rg.get_train_size()[0], self.config.walkDim))
        self.topKSim = defaultdict(dict)
        i = 0
        for user1 in self.CUNet:
            sims = {}
            for user2 in self.CUNet:
                if user1 != user2:
                    wu1 = self.model[str(user1)]
                    wu2 = self.model[str(user2)]
                    sims[user2]=cosine(wu1,wu2) #若为空咋整
            self.topKSim[user1] = sorted(sims.items(), key=lambda d: d[1], reverse=True)[:self.config.topK]
            i += 1
            if i % 200 == 0:
                print('progress:', i, '/', len(self.CUNet))
        # print(self.topKSim)
        #构建被关注列表
        print('Constructing desimilarity matrix...')
        self.topKSimBy = defaultdict(dict)
        for user in self.topKSim:
            users=self.topKSim[user]
            for user2 in users:
                self.topKSimBy[user2[0]][user] = user2[1]
        print('Similarity matrix finished.')

    def train_model(self, k):
        super(CUNE, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2
                p, q = self.P[u], self.Q[i]

                social_term_p, social_term_loss = np.zeros((self.config.factor)), 0.0
                followees = self.topKSim[user] #self.tg.get_followees(user) #self.topKSim[user]
                # print(followees)
                for followee in followees:
                    if self.rg.containsUser(followee[0]):
                        # s = self.user_sim[user][followee]
                        uf = self.P[self.rg.user[followee[0]]]
                        social_term_p += followee[1]* (p - uf) 
                        social_term_loss += followee[1]* ((p - uf).dot(p - uf)) 

                social_term_m = np.zeros((self.config.factor))
                followers = self.topKSimBy[user]
                followers = sorted(followers.items(), key=lambda d: d[1], reverse=True)[:self.config.topK]
                for follower in followers:
                    if self.rg.containsUser(follower[0]):
                        ug = self.P[self.rg.user[follower[0]]]
                        social_term_m += follower[1]*(p - ug) 


                # update latent vectors
                self.P[u] += self.config.lr * (
                        error * q - self.config.alpha * (social_term_p + social_term_m) - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                self.loss += 0.5 * self.config.alpha * social_term_loss

            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
                    self.Q * self.Q).sum()

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    # srg = CUNE()
    # srg.train_model(0)
    # coldrmse = srg.predict_model_cold_users()
    # print('cold start user rmse is :' + str(coldrmse))
    # srg.show_rmse()

    rmses = []
    maes = []
    cunemf = CUNE()
    # cunemf.init_model(0)
    # cunemf.generate_cu_net()
    # cunemf.deep_walk()
    # print(bmf.rg.trainSet_u[1])
    cunemf.config.k_fold_num = 5
    for i in range(cunemf.config.k_fold_num):
        print('the %dth cross validation training' % i)
        cunemf.train_model(i)
        rmse, mae = cunemf.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / cunemf.config.k_fold_num
    mae_avg = sum(maes) / cunemf.config.k_fold_num
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)
