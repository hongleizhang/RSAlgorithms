# encoding:utf-8
import sys

sys.path.append("..")  # 将该目录加入到环境变量
import numpy as np
from mf import MF
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp
from utility import util


class TriCF(MF):
    """
    docstring for TriCF

    """

    def __init__(self):
        super(TriCF, self).__init__()
        # self.config.lr=0.001
        self.config.lambdaU = 0.002  # 0.001 alpha->lambdaU #平滑项系数
        self.config.lambdaI = 0.002  # 0.001  beta-> lambdaI

        self.config.lambdaP = 0.05  # 0.05
        self.config.lambdaQ = 0.05  # 0.05
        self.config.user_near_num = 10  # 10
        self.config.item_near_num = 10  # 10
        self.init_model()

    def init_model(self):
        super(TriCF, self).init_model()
        self.build_user_item_sim_CF()

    # 通过UI矩阵构建user，item的相似度以及k近邻
    def build_user_item_sim_CF(self):
        from collections import defaultdict
        self.user_sim = SimMatrix()  # 保存用户相似度矩阵-UI
        self.item_sim = SimMatrix()  # 保存项目相似度矩阵-UI
        self.user_k_neibor = defaultdict(dict)  # 保存用户k近邻
        self.item_k_neibor = defaultdict(dict)  # 保存项目k近邻
        # 用户
        # print('constructing user-user similarity matrix...')
        self.user_sim = util.load_data('../data/sim/ft_08_uu_tricf.pkl')
        # for u1 in self.rg.user:
        # 	for u2 in self.rg.user:
        # 		if u1!=u2:
        # 			if self.user_sim.contains(u1,u2):
        # 				continue
        # 			sim = pearson_sp(self.rg.get_row(u1),self.rg.get_row(u2))
        #			sim=round(sim,5)
        # 			self.user_sim.set(u1,u2,sim)
        # util.save_data(self.user_sim,'../data/sim/ft_08_uu_tricf.pkl')

        # 寻找用户的k近邻
        self.user_k_neibor = util.load_data(
            '../data/neibor/ft_08_uu_' + str(self.config.user_near_num) + '_neibor_tricf.pkl')
        # for user in self.rg.user:
        # 	matchUsers = sorted(self.user_sim[user].items(),key = lambda x:x[1],reverse=True)[:self.config.user_near_num]
        # 	matchUsers=matchUsers[:self.config.user_near_num]
        # 	self.user_k_neibor[user]=dict(matchUsers)
        # util.save_data(self.user_k_neibor,'../data/neibor/ft_08_uu_'+str(self.config.user_near_num)+'_neibor_tricf.pkl')

        # 项目
        # print('constructing item-item similarity matrix...')
        self.item_sim = util.load_data('../data/sim/ft_08_ii_tricf.pkl')
        # for i1 in self.rg.item:
        # 	for i2 in self.rg.item:
        # 		if i1!=i2:
        # 			if self.item_sim.contains(i1,i2):
        # 				continue
        # 			sim = pearson_sp(self.rg.get_col(i1),self.rg.get_col(i2))
        #			sim=round(sim,5)
        # 			self.item_sim.set(i1,i2,sim)
        # util.save_data(self.item_sim,'../data/sim/ft_08_ii_tricf.pkl')

        # 寻找项目的k近邻
        self.item_k_neibor = util.load_data(
            '../data/neibor/ft_08_ii_' + str(self.config.item_near_num) + '_neibor_tricf.pkl')
        # for item in self.rg.item:
        # 	matchItems = sorted(self.item_sim[item].items(),key = lambda x:x[1],reverse=True)[:self.config.item_near_num]
        # 	matchItems=matchItems[:self.config.item_near_num]
        # 	self.item_k_neibor[item]=dict(matchItems)
        # util.save_data(self.item_k_neibor,'../data/neibor/ft_08_ii_'+str(self.config.item_near_num)+'_neibor_tricf.pkl')
        pass

    def train_model(self):
        print('training model...')
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

                # 获取user和item的k近邻
                matchUsers = self.user_k_neibor[user]
                matchItems = self.item_k_neibor[item]

                u_near_sum, u_near_total = np.zeros((self.config.factor)), 0.0
                for suser in matchUsers.keys():
                    near_user, sim_value = suser, matchUsers[suser]
                    near_user_id = self.rg.user[near_user]
                    u_near_sum += sim_value * (self.P[near_user_id] - p)
                    u_near_total += sim_value * (sum(pow((self.P[near_user_id] - p), 2)))

                i_near_sum, i_near_total = np.zeros((self.config.factor)), 0.0
                for sitem in matchItems:
                    near_item, sim_value = sitem, matchItems[sitem]
                    near_item_id = self.rg.item[near_item]
                    i_near_sum += sim_value * (self.Q[near_item_id] - q)
                    i_near_total += sim_value * (sum(pow((self.Q[near_item_id] - q), 2)))

                self.P[u] += self.config.lr * (error * q - self.config.lambdaU * u_near_sum - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaI * i_near_sum - self.config.lambdaQ * q)

                self.loss += 0.5 * (self.config.lambdaU * u_near_total + self.config.lambdaI * i_near_total)

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum()

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

    def predict_improved(self, user, item):
        if str(user) in self.w2v_model:
            user_mf = self.get_user_url_to_mf(user)
            if self.rg.containsItem(item):
                return user_mf.dot(self.Q[self.rg.item[item]])[0]
            elif self.rg.containsUser(user):
                return self.rg.userMeans[user]
            else:
                return self.rg.globalMean
        else:
            if self.rg.containsUser(user) and self.rg.containsItem(item):
                return self.P[self.rg.user[user]].dot(self.Q[self.rg.item[item]])
            elif self.rg.containsUser(user) and not self.rg.containsItem(item):
                return self.rg.userMeans[user]
            elif not self.rg.containsUser(user) and self.rg.containsItem(item):
                return self.rg.itemMeans[item]
            else:
                return self.rg.globalMean


if __name__ == '__main__':
    tc = TriCF()
    tc.train_model()
    coldrmse = tc.predict_model_cold_users()
    print('cold start user rmse is :' + str(coldrmse))
    # srg.show_rmse()
