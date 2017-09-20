
import numpy as np
import matplotlib.pylab as plt

from metrics.metric import Metric
from utility.tools import denormalize
from reader.rating import RatingGetter
from configx.configx import ConfigX
class MF(object):
	"""
	docstring for MF
	the base class for matrix factorization based model-parent class

	"""
	def __init__(self):
		super(MF, self).__init__()
		self.config=ConfigX()
		self.rg=RatingGetter() #loading raing data
		# self.init_model()
		self.iter_rmse=[]
		pass

	def init_model(self):
		self.P = np.random.rand(self.rg.get_train_size()[0], self.config.factor)/(self.config.factor**0.5)  # latent user matrix
		self.Q = np.random.rand(self.rg.get_train_size()[1], self.config.factor)/(self.config.factor**0.5)  # latent item matrix
		self.loss, self.lastLoss = 0.0, 0.0
		pass

	def train_model(self):
		pass


	def predict_model(self):
		res=[]
		for ind,entry in enumerate(self.rg.testSet()):
			user,item,rating = entry
			#predict
			prediction = self.predict(user,item)
			#denormalize
			prediction = denormalize(prediction,self.config.min_val,self.config.max_val)
			
			pred = self.checkRatingBoundary(prediction)
			# add prediction in order to measure
			# self.dao.testData[ind].append(pred)
			res.append([user,item,rating,pred])
		rmse=Metric.RMSE(res)
		self.iter_rmse.append(rmse) #for plot
		return rmse




	def predict(self,u,i):
		if self.rg.containsUser(u) and self.rg.containsItem(i):
			return self.P[self.rg.user[u]].dot(self.Q[self.rg.item[i]])
		elif self.rg.containsUser(u) and not self.rg.containsItem(i):
			return self.rg.userMeans[u]
		elif not self.rg.containsUser(u) and self.rg.containsItem(i):
			return self.rg.itemMeans[i]
		else:
			return self.rg.globalMean

	def checkRatingBoundary(self,prediction):
		if prediction > self.config.max_val:
			return self.config.max_val
		elif prediction < self.config.min_val:
			return self.config.min_val
		else:
			return round(prediction,3)
			
	def isConverged(self,iter):
		from math import isnan
		if isnan(self.loss):
			print('Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
			exit(-1)
		# measure = self.performance()
		# value = [item.strip()for item in measure]
		#with open(self.algorName+' iteration.txt')
		deltaLoss = (self.lastLoss-self.loss)
		print('%s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f rmse=%.5f' % \
			(self.__class__,iter,self.loss,deltaLoss,self.config.lr,self.predict_model()))
		#check if converged
		cond = abs(deltaLoss) < self.config.threshold
		converged = cond
		if not converged:
			self.updateLearningRate(iter)
		self.lastLoss = self.loss
		# shuffle(self.dao.trainingData)
		return converged


	def updateLearningRate(self,iter):
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
		nums=range(len(self.iter_rmse))
		plt.plot(nums,self.iter_rmse,label='RMSE')
		plt.xlabel('# of epoch')
		plt.ylabel('RMSE')
		plt.title(self.__class__)
		plt.legend()
		plt.show()
		pass