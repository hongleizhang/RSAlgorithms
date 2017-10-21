# encoding:utf-8
import sys

sys.path.append("..")  # 将该目录加入到环境变量
import numpy as np
from mf import MF
from os import listdir
from reader.trust import TrustGetter
from utility.tools import sigmoid, sigmoid_derivative
from utility.similarity import cosine_sp


class SocialRecSigmoid(MF):
    """
    docstring for SocialRec

    Ma H, Yang H, Lyu M R, et al. Sorec: social recommendation using probabilistic matrix factorization[C]//Proceedings of the 17th ACM conference on Information and knowledge management. ACM, 2008: 931-940.

    """

    def __init__(self):
        super(SocialRecSigmoid, self).__init__()
        # self.config.lr=0.0001
        self.config.alpha = 20
        self.config.lambdaZ = 0.001
        self.tg = TrustGetter()
        self.init_model()

    def init_model(self):
        super(SocialRecSigmoid, self).init_model()
        self.Z = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
        self.config.factor ** 0.5)  # latent user social matrix

    def train_model(self):
        iteration = 0
        while iteration < self.config.maxIter:
            # tempP=np.zeros((self.rg.get_train_size()[0], self.config.factor))
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                rating_pre = self.predict(user, item)
                error = rating - sigmoid(rating_pre)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                followees = self.tg.get_followees(user)
                zs = np.zeros(self.config.factor)
                for followee in followees:
                    if self.rg.containsUser(user) and self.rg.containsUser(followee):
                        vminus = len(self.tg.get_followers(followee))  # ~ d - (k)
                        uplus = len(self.tg.get_followees(user))  # ~ d + (i)
                        # import math
                        # try:
                        #     weight = math.sqrt(vminus / (uplus + vminus + 0.0))
                        # except ZeroDivisionError:
                        #     weight = 1
                        zid = self.rg.user[followee]
                        z = self.Z[zid]
                        weight = self.get_sim(u, zid)
                        social_pre = z.dot(p)
                        err = weight - sigmoid(social_pre)
                        self.loss += self.config.alpha * err ** 2
                        zs += -1.0 * err * z * sigmoid_derivative(social_pre)

                        self.Z[zid] += self.config.lr * (self.config.alpha * sigmoid_derivative(social_pre) * err * p - self.config.lambdaZ * z)

                self.P[u] += self.config.lr * (sigmoid_derivative(rating_pre) * error * q - self.config.alpha * zs - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (sigmoid_derivative(rating_pre) * error * p - self.config.lambdaQ * q)

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaZ * (self.Z * self.Z).sum()

            iteration += 1
            if self.isConverged(iteration):
                break

    def get_sim(self, u, k):
        return cosine_sp(self.rg.get_row(u), self.rg.get_row(k))

if __name__ == '__main__':
    src = SocialRecSigmoid()
    src.train_model()
    # rmse, mae = src.cross_validation()
    # print(rmse)
    # print(mae)
