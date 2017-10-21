# encoding:utf-8
import sys

sys.path.append("..")  # 将该目录加入到环境变量
import numpy as np
from mf import MF
from reader.trust import TrustGetter
from utility.tools import sigmoid, sigmoid_derivative
from utility.draw_figure import plot_para
from utility.similarity import cosine_sp


class Model1Sigmoid(MF):
    """
    docstring for SocialRec

    Ma H, Yang H, Lyu M R, et al. Sorec: social recommendation using probabilistic matrix factorization[C]//Proceedings of the 17th ACM conference on Information and knowledge management. ACM, 2008: 931-940.

    """

    def __init__(self):
        super(Model1Sigmoid, self).__init__()
        # self.config.lr=0.0001
        self.config.alpha = 10
        self.config.lambdaZ = 0.001
        self.config.lambdaP = 0.01
        self.config.lambdaZ = 0.001
        self.lamda = np.random.rand(self.rg.get_train_size()[0], 1)
        self.lamda.fill(1 / 2)
        self.tg = TrustGetter()
        self.init_model()

    def init_model(self):
        super(Model1Sigmoid, self).init_model()
        self.Z = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
            self.config.factor ** 0.5)  # latent user social matrix

    def train_model(self):
        iteration = 0
        lamda_mean = []
        lamda_sum = []
        loss_para = []
        while iteration < self.config.maxIter:
            # tempP=np.zeros((self.rg.get_train_size()[0], self.config.factor))
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                rating_pre = self.predict(user, item)
                error = rating - sigmoid(rating_pre)
                rating_loss = (1 - self.lamda[u]) * error ** 2
                self.loss += rating_loss
                p, q = self.P[u], self.Q[i]

                followees = self.tg.get_followees(user)
                zs = np.zeros(self.config.factor)
                social_loss = 0
                err_sum = 0
                for followee in followees:
                    if self.rg.containsUser(user) and self.rg.containsUser(followee):
                        vminus = len(self.tg.get_followers(followee))  # ~ d - (k)
                        uplus = len(self.tg.get_followees(user))  # ~ d + (i)
                        # import math
                        # # try:
                        #     weight = math.sqrt(vminus / (uplus + vminus + 0.0))
                        # except ZeroDivisionError:
                        #     weight = 1
                        zid = self.rg.user[followee]
                        z = self.Z[zid]
                        weight = self.get_sim(user, followee)
                        social_pre = z.dot(p)
                        err = weight - sigmoid(social_pre)
                        err_sum += err ** 2
                        social_loss = self.lamda[u] * err ** 2
                        self.loss += social_loss
                        zs += -1.0 * err * z * sigmoid_derivative(social_pre)

                        self.Z[zid] += self.config.lr * (
                        self.lamda[u] * sigmoid_derivative(social_pre) * err * p - self.config.lambdaZ * z)

                self.P[u] += self.config.lr * (
                (1 - self.lamda[u]) * sigmoid_derivative(rating_pre) * error * q - self.lamda[
                    u] * zs - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (
                (1 - self.lamda[u]) * sigmoid_derivative(rating_pre) * error * p - self.config.lambdaQ * q)
                # 求闭解
                # self.lamda[u] = sigmoid(error ** 2 - (rating_loss + social_loss)) / (error ** 2 - err_sum)
                # 用比例的形式l
                if social_loss != 0:
                    self.lamda[u] = rating_loss / (rating_loss + social_loss)
                else:
                    self.lamda[u] = 0

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaZ * (self.Z * self.Z).sum()

            lamda_sum.append(self.lamda.sum())
            lamda_mean.append(self.lamda.mean())
            loss_para.append(self.loss)

            # write to log file
            epoch_log = "Loss: %.4f learning rate: %.4f lambda sum: %.2f lambda mean: %.2f lambda length: %d \n" \
                        % (self.loss, self.config.lr, self.lamda.sum(), self.lamda.mean(), len(self.lamda))
            log_file_name = "../log/log_model2.txt"
            with open(log_file_name, "a") as log_file:
                log_file.write(epoch_log)

            iteration += 1
            if self.isConverged(iteration):
                break

        plot_para(lamda_mean, "lambda mean", "lambda_mean")
        plot_para(lamda_sum, "lambda sum", "lambda_sum")
        plot_para(loss_para, "loss", "loss")

    def isConverged(self, iter):
        from math import isnan
        if isnan(self.loss):
            print(
                'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try '
                'again!')
            exit(-1)
        # measure = self.performance()
        # value = [item.strip()for item in measure]
        # with open(self.algorName+' iteration.txt')
        deltaLoss = (self.lastLoss - self.loss)
        rmse, mae = self.predict_model()
        print(
            '%s iteration %d: loss=%.4f, delta_loss=%.5f learning_Rate=%.5f lamda_mean=% .5f rmse=%.5f mae=%.5f' % \
            (self.__class__, iter, self.loss, deltaLoss, self.config.lr, self.lamda.mean(), rmse, mae))
        # check if converged
        cond = abs(deltaLoss) < self.config.threshold
        converged = cond
        self.lastLoss = self.loss
        return converged

    def get_sim(self, u, k):
        return cosine_sp(self.rg.get_row(u), self.rg.get_row(k))

if __name__ == '__main__':
    src = Model1Sigmoid()
    is_cross_validation = False
    if is_cross_validation:
        rmse, mae = src.cross_validation()
        print(rmse)
        print(mae)
    else:
        src.train_model()
        coldrmse = src.predict_model_cold_users()
        print('cold start user rmse is :' + str(coldrmse))
