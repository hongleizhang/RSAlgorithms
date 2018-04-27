#encoding:utf-8
import  sys
sys.path.append("..")

import numpy as np
import matplotlib.pylab as plt

from prettyprinter import cpprint
from metrics.metric import Metric
from utility.tools import denormalize,sigmoid
from reader.rating import RatingGetter
from configx.configx import ConfigX


class MF(object):
    """
    docstring for MF
    the base class for matrix factorization based model-parent class

    """

    def __init__(self):
        super(MF, self).__init__()
        self.config = ConfigX()
        cpprint(self.config.__dict__)  #print the configuration

        # self.rg = RatingGetter()  # loading raing data
        # self.init_model()
        self.iter_rmse = []
        self.iter_mae = []
        pass

    def init_model(self,k):
        self.read_data(k)
        self.P = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
        self.config.factor ** 0.5)  # latent user matrix
        self.Q = np.random.rand(self.rg.get_train_size()[1], self.config.factor) / (
        self.config.factor ** 0.5)  # latent item matrix
        self.loss, self.lastLoss = 0.0, 0.0
        self.lastRmse, self.lastMae = 10.0,10.0
        pass

    def read_data(self,k):
        self.rg = RatingGetter(k)
        pass

    def train_model(self,k):
        self.init_model(k)
        pass

    # test all users in test set
    def predict_model(self):
        res = []
        for ind, entry in enumerate(self.rg.testSet()):
            user, item, rating = entry
            rating_length = len(self.rg.trainSet_u[user]) # remove cold start users for test
            if rating_length <= self.config.coldUserRating:
                continue

            prediction = self.predict(user, item)
            # denormalize
            prediction = denormalize(prediction, self.config.min_val, self.config.max_val)

            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            res.append([user, item, rating, pred])
        rmse = Metric.RMSE(res)
        mae = Metric.MAE(res)
        self.iter_rmse.append(rmse)  # for plot
        self.iter_mae.append(mae)
        return rmse, mae

    # test cold start users among test set
    def predict_model_cold_users(self):
        res = []
        for user in self.rg.testColdUserSet_u.keys():
            for item in self.rg.testColdUserSet_u[user].keys():
                rating = self.rg.testColdUserSet_u[user][item]
                pred = self.predict(user, item)
                # pred = sigmoid(pred)
                # denormalize
                pred = denormalize(pred, self.config.min_val, self.config.max_val)
                pred = self.checkRatingBoundary(pred)
                res.append([user, item, rating, pred])
        rmse = Metric.RMSE(res)
        mae = Metric.MAE(res)
        return rmse,mae

    def predict(self, u, i):
        if self.rg.containsUser(u) and self.rg.containsItem(i):
            return self.P[self.rg.user[u]].dot(self.Q[self.rg.item[i]])
        elif self.rg.containsUser(u) and not self.rg.containsItem(i):
            return self.rg.userMeans[u]
        elif not self.rg.containsUser(u) and self.rg.containsItem(i):
            return self.rg.itemMeans[i]
        else:
            return self.rg.globalMean

    def checkRatingBoundary(self, prediction):
        prediction =round( min( max( prediction , self.config.min_val ) , self.config.max_val ) ,3)
        return prediction

    def isConverged(self, iter):
        from math import isnan
        if isnan(self.loss):
            print(
                'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)

        deltaLoss = (self.lastLoss - self.loss)
        rmse, mae = self.predict_model()

        # early stopping
        if self.config.isEarlyStopping == True:
            cond = self.lastRmse < rmse
            if cond:
                print('test rmse increase, so early stopping')
                return cond
            self.lastRmse = rmse
            self.lastMae = mae

        print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f rmse=%.5f mae=%.5f' % \
              (self.__class__, iter, self.loss, deltaLoss, self.config.lr, rmse, mae))

        # check if converged
        cond = abs(deltaLoss) < self.config.threshold
        converged = cond
        # if not converged:
        # 	self.updateLearningRate(iter)
        self.lastLoss = self.loss
        # shuffle(self.dao.trainingData)
        return converged

    def updateLearningRate(self, iter):
        if iter > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.config.lr *= 1.05
            else:
                self.config.lr *= 0.5
        if self.config.lr > 1:
            self.config.lr = 1

    def show_rmse(self):
        '''
        show figure for rmse and epoch
        '''
        nums = range(len(self.iter_rmse))
        plt.plot(nums, self.iter_rmse, label='RMSE')
        plt.plot(nums, self.iter_mae, label='MAE')
        plt.xlabel('# of epoch')
        plt.ylabel('metric')
        plt.title(self.__class__)
        plt.legend()
        plt.show()
        pass
    def show_loss(self,loss_all,faloss_all):
        '''
        show figure for rmse and epoch
        '''
        nums = range(len(loss_all))
        plt.plot(nums, loss_all, label='front')
        plt.plot(nums, faloss_all, label='rear')
        plt.xlabel('# of epoch')
        plt.ylabel('loss')
        plt.title('loss experiment')
        plt.legend()
        plt.show()
        pass
