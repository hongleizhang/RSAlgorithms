# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
from mf import MF
from reader.trust import TrustGetter


class SocialMF(MF):
    """
    docstring for SocialMF

    Jamali M, Ester M. A matrix factorization technique with trust propagation for recommendation in social networks[C]//Proceedings of the fourth ACM conference on Recommender systems. ACM, 2010: 135-142.
    """

    def __init__(self):
        super(SocialMF, self).__init__()
        # self.config.lr=0.0001
        self.config.alpha = 1  # 0.8 rmse=0.87605
        self.tg = TrustGetter()  # loading trust data
        # self.init_model()

    def train_model(self, k):
        super(SocialMF, self).train_model(k)
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

                total_weight = 0.0
                social_term = np.zeros(self.config.factor)
                followees = self.tg.get_followees(user)  # get user u's focus lsit
                for followee in followees:
                    weight = followees[followee]
                    if self.rg.containsUser(followee):
                        uk = self.P[self.rg.user[followee]]
                        social_term += weight * uk
                        total_weight += weight

                if total_weight != 0:
                    social_term = p - social_term / total_weight

                social_term_a = np.zeros(self.config.factor)
                total_count = 0
                followers = self.tg.get_followers(user)
                for follower in followers:
                    if self.rg.containsUser(follower):
                        total_count += 1
                        uv = self.P[self.rg.user[follower]]
                        social_term_m = np.zeros(self.config.factor)
                        total_weight = 0.0
                        followees = self.tg.get_followees(follower)
                        for followee in followees:
                            weight = followees[followee]
                            if self.rg.containsUser(followee):
                                uw = self.P[self.rg.user[followee]]
                                social_term_m += weight * uw
                                total_weight += weight
                        if total_weight != 0:
                            social_term_a += uv - social_term_m / total_weight
                if total_count != 0:
                    social_term_a /= total_count

                # update latent vectors
                self.P[u] += self.config.lr * (
                        error * q - self.config.alpha * social_term + self.config.alpha * social_term_a - self.config.lambdaP * p)  #
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                self.loss += self.config.alpha * social_term.dot(social_term).sum()

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum()

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    rmses = []
    maes = []
    tcsr = SocialMF()
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
