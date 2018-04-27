# encoding:utf-8
import sys

sys.path.append("..")

import math
import numpy as np
from mf import MF
from reader.trust import TrustGetter


class TrustSVD(MF):
    """
    docstring for TrustSVD
    implement the TrustSVD

    Koren Y. Factor in the neighbors: Scalable and accurate collaborative filtering[J]. ACM Transactions on Knowledge Discovery from Data (TKDD), 2010, 4(1): 1.
    """

    def __init__(self):
        super(TrustSVD, self).__init__()

        self.config.lr = 0.01  # 0.005
        self.config.maxIter = 100
        self.config.lambdaP = 1.2
        self.config.lambdaQ = 1.2

        self.config.lambdaB = 1.2
        self.config.lambdaY = 1.2
        self.config.lambdaW = 1.2
        self.config.lambdaT = 0.9

        self.tg = TrustGetter()
        # self.init_model()

    def init_model(self, k):
        super(TrustSVD, self).init_model(k)
        self.Bu = np.random.rand(self.rg.get_train_size()[0]) / (self.config.factor ** 0.5)  # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1]) / (self.config.factor ** 0.5)  # bias value of item
        self.Y = np.random.rand(self.rg.get_train_size()[1], self.config.factor) / (
                self.config.factor ** 0.5)  # implicit preference
        self.W = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
                self.config.factor ** 0.5)  # implicit preference

    def train_model(self, k):
        super(TrustSVD, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += error ** 2

                p, q = self.P[u], self.Q[i]
                nu, sum_y = self.get_sum_y(user)
                nv, sum_w = self.get_sum_w(user)

                frac = lambda x: 1.0 / math.sqrt(x)

                # update latent vectors
                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * frac(nu) * self.Bu[u])
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * frac(nv) * self.Bi[i])

                self.Q[i] += self.config.lr * (error * (p + sum_y + sum_w) - self.config.lambdaQ * frac(nu) * q)

                followees = self.tg.get_followees(user)
                ws = np.zeros(self.config.factor)
                for followee in followees:
                    if self.rg.containsUser(user) and self.rg.containsUser(followee):
                        nw = len(self.tg.get_followers(followee))
                        vid = self.rg.user[followee]
                        w = self.W[vid]
                        weight = 1  # followees[followee]
                        err = w.dot(p) - weight
                        self.loss += err ** 2
                        ws += err * w
                        self.W[vid] += self.config.lr * (
                                err * frac(nv) * q - self.config.lambdaT * err * p - self.config.lambdaW * frac(
                            nw) * w)  # 更新w
                self.P[u] += self.config.lr * (error * q - self.config.lambdaT * ws - (
                        self.config.lambdaP * frac(nu) + self.config.lambdaT * frac(nv)) * p)

                u_items = self.rg.user_rated_items(u)
                for j in u_items:
                    idj = self.rg.item[j]
                    self.Y[idj] += self.config.lr * (
                            error * frac(nu) * q - self.config.lambdaY * frac(nv) * self.Y[idj])

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaB * (
                                 (self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum()) + self.config.lambdaY * (
                                 self.Y * self.Y).sum() + self.config.lambdaW * (self.W * self.W).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

    def predict(self, u, i):
        if self.rg.containsUser(u) and self.rg.containsItem(i):
            _, sum_y = self.get_sum_y(u)
            _, sum_w = self.get_sum_w(u)
            u = self.rg.user[u]
            i = self.rg.item[i]
            return self.Q[i].dot(self.P[u] + sum_y + sum_w) + self.rg.globalMean + self.Bi[i] + self.Bu[u]
        elif self.rg.containsUser(u) and not self.rg.containsItem(i):
            return self.rg.userMeans[u]
        elif not self.rg.containsUser(u) and self.rg.containsItem(i):
            return self.rg.itemMeans[i]
        else:
            return self.rg.globalMean

    def get_sum_y(self, u):
        u_items = self.rg.user_rated_items(u)
        nu = len(u_items)
        sum_y = np.zeros(self.config.factor)
        for j in u_items:
            sum_y += self.Y[self.rg.item[j]]
        sum_y /= (np.sqrt(nu))
        return nu, sum_y

    def get_sum_w(self, u):
        followees = self.tg.get_followees(u)
        nu = 1
        sum_w = np.zeros(self.config.factor)
        for v in followees.keys():
            if self.rg.containsUser(v):
                nu += 1
                sum_w += self.W[self.rg.user[v]]
        sum_w /= np.sqrt(nu)
        return nu, sum_w


if __name__ == '__main__':
    bmf = TrustSVD()
    bmf.train_model(0)
    coldrmse = bmf.predict_model_cold_users()
    print('cold start user rmse is :' + str(coldrmse))
    bmf.show_rmse()
