# encoding:utf-8
import sys

sys.path.append("..")
from prettyprinter import cpprint
import numpy as np
from mf import MF
from collections import defaultdict
from utility.similarity import pearson_sp
from utility import util


class IntegSVD(MF):
    """
    docstring for IntegSVD
    implement the IntegSVD

    Koren Y. Factor in the neighbors: Scalable and accurate collaborative filtering[J]. ACM Transactions on Knowledge Discovery from Data (TKDD), 2010, 4(1): 1.
    """

    def __init__(self):
        super(IntegSVD, self).__init__()

        # self.config.lr=0.001
        # self.config.maxIter=200 #400
        self.config.item_near_num = 10  # 10

        self.config.lambdaP = 0.001
        self.config.lambdaQ = 0.001

        # self.config.lambdaY = 0.01
        self.config.lambdaB = 0.01

        self.config.lambdaW = 0.01
        # self.config.lambdaC = 0.015
        # self.init_model()

    def init_model(self, k):
        super(IntegSVD, self).init_model(k)

        self.Bu = np.random.rand(self.rg.get_train_size()[0]) / (self.config.factor ** 0.5)  # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1]) / (self.config.factor ** 0.5)  # bias value of item
        # self.Y = np.random.rand(self.rg.get_train_size()[1], self.config.factor) / (self.config.factor ** 0.5)  # implicit preference

        self.user_item_nei = defaultdict(dict)
        self.W = np.random.rand(self.rg.get_train_size()[1], self.rg.get_train_size()[1])
        # self.C=np.random.rand(self.rg.get_train_size()[1],self.rg.get_train_size()[1])

        # print('initializinig neighbors')
        # for user in self.rg.trainSet_u:
        #     for item in self.rg.trainSet_u[user]:
        #         self.get_neighbor(user,item)

    def train_model(self, k):
        super(IntegSVD, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                ui_neighbors = self.get_neighbor(user, item)
                ui_nei_len = len(ui_neighbors)
                error = rating - self.predict(user, item)
                self.loss += error ** 2

                p, q = self.P[u], self.Q[i]
                # nu, sum_y = self.get_sum_y(user)

                # update latent vectors
                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * self.Bu[u])
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * self.Bi[i])

                self.P[u] += self.config.lr * (error * q - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)  # + sum_y

                # 更新Y
                # u_items = self.rg.user_rated_items(u)
                # for j in u_items:
                #     idj = self.rg.item[j]
                #     self.Y[idj] += self.config.lr * (error / np.sqrt(nu) * q - self.config.lambdaY * self.Y[idj])
                # 更新W,C
                for neighbor in ui_neighbors:
                    j = self.rg.item[neighbor]
                    ruj = self.rg.trainSet_u[user][neighbor]
                    buj = self.rg.globalMean + self.Bu[u] + self.Bi[j]
                    self.W[i][j] += self.config.lr * (
                            error / (ui_nei_len ** 0.5) * (ruj - buj) - self.config.lambdaW * self.W[i][j])
                    # self.C[i][j] += self.config.lr * (error / (ui_nei_len ** 0.5) - self.config.lambdaC * self.C[i][j])

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaB * ( \
                                     (self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum()) + self.config.lambdaW * (
                                 self.W * self.W).sum()  # + self.config.lambdaY * (self.Y * self.Y).sum() \
            # +self.config.lambdaC * (self.C * self.C).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

        util.save_data(self.user_item_nei, '../data/neibor/ft_intsvd_useritemnei_08.pkl')

    def predict(self, user, item):
        if self.rg.containsUser(user) and self.rg.containsItem(item):
            # _, sum_y = self.get_sum_y(user)
            sum_w = 0.0
            u = self.rg.user[user]
            i = self.rg.item[item]
            bui = self.rg.globalMean + self.Bi[i] + self.Bu[u]
            ui_neighbors = self.get_neighbor(user, item)
            ui_len = len(ui_neighbors)
            for neighbor in ui_neighbors:
                j = self.rg.item[neighbor]
                ruj = self.rg.trainSet_u[user][neighbor]
                buj = self.rg.globalMean + self.Bi[j] + self.Bu[u]
                sum_w += (ruj - buj) * self.W[i][j]  # +self.C[i][j]
            if ui_len != 0:
                sum_w *= 1.0 / ui_len  # 这的事
            return bui + self.Q[i].dot(self.P[u]) + sum_w  # + sum_y
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

    def get_neighbor(self, user, item):
        if user in self.user_item_nei and item in self.user_item_nei[user]:
            return self.user_item_nei[user][item]
        items = self.rg.user_rated_items(user)
        u_item_d = {}
        for u_item in items:
            if item != u_item:
                sim = pearson_sp(self.rg.get_col(item), self.rg.get_col(u_item))
                u_item_d[u_item] = round(sim, 4)
        matchItems = sorted(u_item_d.items(), key=lambda x: x[1], reverse=True)[:self.config.item_near_num]
        matchItems = list(zip(*matchItems))
        if len(matchItems) > 0:
            self.user_item_nei[user][item] = matchItems[0]
            return matchItems[0]
        else:
            return []


if __name__ == '__main__':
    rmses = []
    bmf = IntegSVD()
    # print(bmf.rg.trainSet_u[1])
    for i in range(bmf.config.k_fold_num):
        bmf.train_model(i)
        rmse, mae = bmf.predict_model()
        rmses.append(rmse)
    print(rmses)
    # bmf.config.k_current = 1
    # print(bmf.rg.trainSet_u[1])
    # bmf.train_model()
    # bmf.predict_model()
