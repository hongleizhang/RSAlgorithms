# encoding:utf-8
import sys

sys.path.append("..")
from prettyprinter import cpprint
import numpy as np
from mf import MF


class SVDPP(MF):
    """
    docstring for SVDPP
    implement the SVDPP

    Koren Y. Factor in the neighbors: Scalable and accurate collaborative filtering[J]. ACM Transactions on Knowledge Discovery from Data (TKDD), 2010, 4(1): 1.
    """

    def __init__(self):
        super(SVDPP, self).__init__()
        self.config.lambdaP = 0.001
        self.config.lambdaQ = 0.001

        self.config.lambdaY = 0.001
        self.config.lambdaB = 0.001
        # self.init_model()

    def init_model(self, k):
        super(SVDPP, self).init_model(k)
        self.Bu = np.random.rand(self.rg.get_train_size()[0]) / (self.config.factor ** 0.5)  # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1]) / (self.config.factor ** 0.5)  # bias value of item
        self.Y = np.random.rand(self.rg.get_train_size()[1], self.config.factor) / (
                self.config.factor ** 0.5)  # implicit preference
        self.SY = dict()

    def train_model(self, k):
        super(SVDPP, self).train_model(k)
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

                # update latent vectors
                self.P[u] += self.config.lr * (error * q - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * (p + sum_y) - self.config.lambdaQ * q)

                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * self.Bu[u])
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * self.Bi[i])

                u_items = self.rg.user_rated_items(u)
                for j in u_items:
                    idj = self.rg.item[j]
                    self.Y[idj] += self.config.lr * (error / np.sqrt(nu) * q - self.config.lambdaY * self.Y[idj])

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaB * (
                                 (self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum()) + self.config.lambdaY * (
                                     self.Y * self.Y).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

    def predict(self, u, i):
        if self.rg.containsUser(u) and self.rg.containsItem(i):
            _, sum_y = self.get_sum_y(u)
            u = self.rg.user[u]
            i = self.rg.item[i]
            return self.Q[i].dot(self.P[u] + sum_y) + self.rg.globalMean + self.Bi[i] + self.Bu[u]
        else:
            return self.rg.globalMean

    def get_sum_y(self, u):
        if u in self.SY:
            return self.SY[u]
        u_items = self.rg.user_rated_items(u)
        nu = len(u_items)
        sum_y = np.zeros(self.config.factor)
        for j in u_items:
            sum_y += self.Y[self.rg.item[j]]
        sum_y /= (np.sqrt(nu))
        self.SY[u] = [nu, sum_y]
        return nu, sum_y


if __name__ == '__main__':
    rmses = []
    maes = []
    tcsr = SVDPP()
    # print(bmf.rg.trainSet_u[1])
    for i in range(tcsr.config.k_fold_num):
        print('the %dth cross validation training' % i)
        tcsr.train_model(i)
        rmse, mae = tcsr.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / 5
    mae_avg = sum(maes) / 5
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)
