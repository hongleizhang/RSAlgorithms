#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量
import numpy as np
from mf import MF
from reader.trust import TrustGetter

class SocialRec(MF):
	"""
	docstring for SocialRec
	
	Ma H, Yang H, Lyu M R, et al. Sorec: social recommendation using probabilistic matrix factorization[C]//Proceedings of the 17th ACM conference on Information and knowledge management. ACM, 2008: 931-940.
	
	"""
	def __init__(self):
		super(SocialRec, self).__init__()
		# self.config.lr=0.0001
		self.config.alpha=10
		self.config.lambdaZ=0.001
		self.tg=TrustGetter()
		self.init_model()

	def init_model(self): 
		super(SocialRec, self).init_model()
		self.Z=np.random.rand(self.rg.get_train_size()[0], self.config.factor)/(self.config.factor**0.5)  # latent user social matrix


	def train_model(self):
		iteration = 0
		while iteration < self.config.maxIter:
			# tempP=np.zeros((self.rg.get_train_size()[0], self.config.factor))
			self.loss = 0
			for index,line in enumerate(self.rg.trainSet()):
				user, item, rating = line
				u = self.rg.user[user]
				i = self.rg.item[item]
				error = rating - self.predict(user,item)
				self.loss += error**2
				p,q = self.P[u],self.Q[i]

				followees=self.tg.get_followees(user)
				zs=np.zeros(self.config.factor)
				for followee in followees:
					if self.rg.containsUser(user) and self.rg.containsUser(followee):
						vminus = len(self.tg.get_followers(followee))# ~ d - (k)
						uplus = len(self.tg.get_followees(user))#~ d + (i)
						import math
						try:
							weight = math.sqrt(vminus / (uplus + vminus + 0.0))
						except ZeroDivisionError:
							weight = 1
						zid=self.rg.user[followee]
						z=self.Z[zid]
						err=weight-z.dot(p)
						self.loss+=err**2
						zs+=-1.0*err*p
						self.Z[zid] += self.config.lr * (self.config.alpha * err * p - self.config.lambdaZ * z)


				#update latent vectors
				# tempP[u]+=self.config.lr*(error*q-self.config.lambdaP*p)

				self.P[u] += self.config.lr*(error*q-self.config.alpha*zs-self.config.lambdaP*p)
				self.Q[i] += self.config.lr*(error*p-self.config.lambdaQ*q)

			# for entry in self.tg.relations:
			# 	u, v, t = entry
			# 	if self.rg.containsUser(u) and self.rg.containsUser(v):
			# 		vminus = len(self.tg.getFollowers(v))# ~ d - (k)
			# 		uplus = len(self.tg.getFollowees(u))#~ d + (i)
			# 		try:
			# 			weight = math.sqrt(vminus / (uplus + vminus + 0.0))
			# 		except ZeroDivisionError:
			# 			weight = 1
			# 		v = self.rg.user[v]
			# 		u = self.rg.user[u]
			# 		euv = weight * t - self.P[u].dot(self.Z[v])  # weight * tuv~ cik *
			# 		self.loss += self.config.alpha * (euv ** 2)
			# 		p = self.P[u]
			# 		z = self.Z[v]

			# 		# update latent vectors
			# 		tempP[u]+=self.config.lr * (self.config.alpha * euv * z)
			# 		# self.P[u] += self.config.lr * (self.config.alpha * euv * z)
			# 		self.Z[v] += self.config.lr * (self.config.alpha * euv * p - self.config.lambdaZ * z)

			# self.P+=tempP

			self.loss+=self.config.lambdaP*(self.P*self.P).sum() + self.config.lambdaQ*(self.Q*self.Q).sum()\
				+self.config.lambdaZ*(self.Z*self.Z).sum()

			iteration += 1
			if self.isConverged(iteration):
				break

if __name__ == '__main__':
	src=SocialRec()
	src.train_model()
	coldrmse=src.predict_model_cold_users()
	print('cold start user rmse is :'+str(coldrmse))
	src.show_rmse()