# encoding:utf-8
import sys
import os
sys.path.append("..")
import numpy as np
from mf import MF
from prettyprinter import cpprint
from collections import defaultdict
from prettyprinter import cpprint
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp
from utility import util


class TriCFBias(MF):
    """
    docstring for TriCFBias

    """

    def __init__(self):
        super(TriCFBias, self).__init__()
        # self.config.lr=0.001
        self.config.lambdaU = 0.002
        self.config.lambdaI = 0.001

        self.config.lambdaP = 0.02
        self.config.lambdaQ = 0.03
        self.config.lambdaB = 0.01

        self.config.user_near_num = 50
        self.config.item_near_num = 50
        # self.init_model()

    def init_model(self, k):
        super(TriCFBias, self).init_model(k)
        self.Bu = np.random.rand(self.rg.get_train_size()[0])  # bias value of user
        self.Bi = np.random.rand(self.rg.get_train_size()[1])  # bais value of item
        self.build_user_item_sim_CF()

    # construct the u-u,i-i similarity matirx and their's k neighbors
    def build_user_item_sim_CF(self):
        from collections import defaultdict
        self.user_sim = SimMatrix()
        self.item_sim = SimMatrix()
        self.user_k_neibor = defaultdict(dict)
        self.item_k_neibor = defaultdict(dict)

        # compute item-item similarity matrix
        print('constructing user-user similarity matrix...')
        # self.user_sim = util.load_data('../data/sim/ft_08_uu_tricf.pkl')
        for u1 in self.rg.user:
            for u2 in self.rg.user:
                if u1 != u2:
                    if self.user_sim.contains(u1, u2):
                        continue
                    sim = pearson_sp(self.rg.get_row(u1), self.rg.get_row(u2))
                    sim = round(sim, 5)
                    self.user_sim.set(u1, u2, sim)
        if not os.path.exists('../data/sim'):
            os.makedirs('../data/sim')
            print('../data/sim folder has been established.')

        util.save_data(self.user_sim, '../data/sim/ft_08_uu_tricf_cv0.pkl')

        # compute the k neighbors of user
        # self.user_k_neibor = util.load_data(
        #     '../data/neibor/ft_08_uu_' + str(self.config.user_near_num) + '_neibor_tricf.pkl')
        for user in self.rg.user:
            matchUsers = sorted(self.user_sim[user].items(), key=lambda x: x[1], reverse=True)[
                         :self.config.user_near_num]
            matchUsers = matchUsers[:self.config.user_near_num]
            self.user_k_neibor[user] = dict(matchUsers)

        if not os.path.exists('../data/neibor'):
            os.makedirs('../data/neibor')
            print('../data/neibor folder has been established.')
            
        util.save_data(self.user_k_neibor,
                       '../data/neibor/ft_08_uu_' + str(self.config.user_near_num) + '_neibor_tricf_cv0.pkl')

        # compute item-item similarity matrix
        print('constructing item-item similarity matrix...')
        # self.item_sim = util.load_data('../data/sim/ft_08_ii_tricf.pkl')
        for i1 in self.rg.item:
            for i2 in self.rg.item:
                if i1 != i2:
                    if self.item_sim.contains(i1, i2):
                        continue
                    sim = pearson_sp(self.rg.get_col(i1), self.rg.get_col(i2))
                    sim = round(sim, 5)
                    self.item_sim.set(i1, i2, sim)
        util.save_data(self.item_sim, '../data/sim/ft_08_ii_tricf_cv0.pkl')

        # compute the k neighbors of item
        # self.item_k_neibor = util.load_data(
        #     '../data/neibor/ft_08_ii_' + str(self.config.item_near_num) + '_neibor_tricf.pkl')
        for item in self.rg.item:
            matchItems = sorted(self.item_sim[item].items(), key=lambda x: x[1], reverse=True)[
                         :self.config.item_near_num]
            matchItems = matchItems[:self.config.item_near_num]
            self.item_k_neibor[item] = dict(matchItems)
        util.save_data(self.item_k_neibor,
                       '../data/neibor/ft_08_ii_' + str(self.config.item_near_num) + '_neibor_tricf_cv0.pkl')
        pass

    def train_model(self, k):
        super(TriCFBias, self).train_model(k)
        print('training model...')
        iteration = 0
        # faflag=True
        while iteration < self.config.maxIter:
            self.loss = 0
            self.u_near_total_dict = defaultdict()
            self.i_near_total_dict = defaultdict()
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]

                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                # get the k neighbors of user and item
                matchUsers = self.user_k_neibor[user]
                matchItems = self.item_k_neibor[item]

                u_near_sum, u_near_total, s = np.zeros((self.config.factor)), 0.0, 0.0
                for suser in matchUsers.keys():
                    near_user, sim_value = suser, matchUsers[suser]
                    if sim_value != 0.0:
                        s += sim_value
                        pn = self.P[self.rg.user[near_user]]
                        u_near_sum += sim_value * (pn - p)
                        u_near_total += sim_value * ((pn - p).dot(pn - p))
                if s != 0.0:
                    u_near_sum /= s

                i_near_sum, i_near_total, ss = np.zeros((self.config.factor)), 0.0, 0.0
                for sitem in matchItems:
                    near_item, sim_value = sitem, matchItems[sitem]
                    if sim_value != 0.0:
                        ss += sim_value
                    qn = self.Q[self.rg.item[near_item]]
                    i_near_sum += sim_value * (qn - q)
                    i_near_total += sim_value * ((qn - q).dot(qn - q))
                if ss != 0.0:
                    i_near_sum /= ss

                if u not in self.u_near_total_dict:
                    self.u_near_total_dict[u] = u_near_total
                if i not in self.i_near_total_dict:
                    self.i_near_total_dict[i] = i_near_total

                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * self.Bu[u])
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * self.Bi[i])

                self.P[u] += self.config.lr * (error * q - self.config.lambdaU * u_near_sum - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaI * i_near_sum - self.config.lambdaQ * q)

                self.loss += 0.5 * (self.config.lambdaU * u_near_total + self.config.lambdaI * i_near_total)

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaB * ((self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum())

            iteration += 1
            if self.isConverged(iteration):
                break

    # test cold start users among test set
    def predict_model_cold_users_improved(self):
        res = []
        for user in self.rg.testColdUserSet_u.keys():
            for item in self.rg.testColdUserSet_u[user].keys():
                rating = self.rg.testColdUserSet_u[user][item]
                pred = self.predict_improved(user, item)
                # denormalize
                pred = denormalize(pred, self.config.min_val, self.config.max_val)
                pred = self.checkRatingBoundary(pred)
                res.append([user, item, rating, pred])
        rmse = Metric.RMSE(res)
        return rmse


if __name__ == '__main__':
    rmses = []
    maes = []
    tcsr = TriCFBias()
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