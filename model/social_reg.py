#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量
import numpy as np
from mf import MF
from reader.trust import TrustGetter
from utility.similarity import pearson_sp

class SocialReg(MF):
	"""
	docstring for SocialReg
	
	Ma et al. 2011
	"""
	def __init__(self):
		super(SocialReg, self).__init__()
		self.config.alpha=0.1
		self.tg=TrustGetter()
		self.init_model()


	def init_model(self): 
		super(SocialReg, self).init_model()
		from collections import defaultdict
		self.Sim = defaultdict(dict)
		print('constructing similarity matrix...')
		self.tg
		for user in self.rg.user:
			for k in self.tg.get_followees(user):
				if user in self.Sim and k in self.Sim[user]:
					pass
				else:
					self.Sim[user][k]=self.get_sim(user,k)
					# self.Sim[k][user]=self.Sim[user][k]



	def train_model(self):
		iteration = 0
		while iteration < self.config.maxIter:
			self.loss = 0
			for index,line in enumerate(self.rg.trainSet()):
				user, item, rating = line
				u = self.rg.user[user]
				i = self.rg.item[item]
				error = rating - self.predict(user,item)
				self.loss += error**2
				p,q = self.P[u],self.Q[i]
				#update latent vectors
				self.P[u] += self.config.lr*(error*q-self.config.lambdaP*p)
				self.Q[i] += self.config.lr*(error*p-self.config.lambdaQ*q)

			for user in self.tg.user:
				if self.rg.containsUser(user):
					u=self.rg.user[user]
					ui=self.P[u]
					# total_weight=0
					social_term_loss=0.0
					social_term = np.zeros(self.config.factor)
					followees = self.tg.get_followees(user) #获得u所关注的用户列表
					for followee in followees:
						if self.rg.containsUser(followee):
							weight=self.Sim[user][followee]
							uk = self.P[self.rg.user[followee]]
							social_term += weight * (ui-uk)
							# total_weight += weight
							social_term_loss+=weight*((ui-uk).dot(ui-uk))

					# update latent vectors
					self.P[u] -= self.config.lr * self.config.alpha * social_term

					self.loss +=  self.config.alpha * social_term_loss

			self.loss+=self.config.lambdaP*(self.P*self.P).sum() + self.config.lambdaQ*(self.Q*self.Q).sum()

			iteration += 1
			if self.isConverged(iteration):
				break



	def get_sim(self,u,k):
		return (pearson_sp(self.rg.get_row(u), self.rg.get_row(k))+self.tg.weight(u,k))/2.0



if __name__ == '__main__':
	srg=SocialReg()
	srg.train_model()
	srg.show_rmse()