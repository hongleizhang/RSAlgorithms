#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量
import numpy as np
from mf import MF
from reader.trust import TrustGetter
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp

class SocialReg(MF):
	"""
	docstring for SocialReg
	
	Ma H, Zhou D, Liu C, et al. Recommender systems with social regularization[C]//Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011: 287-296.
	"""
	def __init__(self):
		super(SocialReg, self).__init__()
		# self.config.lr=0.015
		self.config.beta=0.02
		self.tg=TrustGetter()
		self.init_model()


	def init_model(self): 
		super(SocialReg, self).init_model()
		from collections import defaultdict
		self.user_sim = SimMatrix()
		print('constructing user-user similarity matrix...')

		for u1 in self.rg.user:
			for u2 in self.rg.user:
				if u1!=u2:
					if self.user_sim.contains(u1,u2):
						continue
					sim = self.get_sim(u1,u2)
					self.user_sim.set(u1,u2,sim)


	def get_sim(self,u,k):
		return (pearson_sp(self.rg.get_row(u), self.rg.get_row(k))+1.0)/2.0 #为了让范围在[0,1]


	def train_model(self):
		iteration = 0
		while iteration < self.config.maxIter:
			self.loss = 0
			for index,line in enumerate(self.rg.trainSet()):
				user, item, rating = line
				u = self.rg.user[user]
				i = self.rg.item[item]
				error = rating - self.predict(user,item)
				self.loss += 0.5*error**2
				p,q = self.P[u],self.Q[i]

				social_term_p,social_term_loss=np.zeros((self.config.factor)),0.0 #用户u的朋友
				followees = self.tg.get_followees(user) 
				for followee in followees:
					if self.rg.containsUser(followee):
						s=self.user_sim[user][followee]
						uf = self.P[self.rg.user[followee]]
						social_term_p += s * (p-uf)
						social_term_loss+=s*((p-uf).dot(p-uf))


				social_term_m=np.zeros((self.config.factor)) #把u看做朋友的用户
				followers=self.tg.get_followers(user)
				for follower in followers:
					if self.rg.containsUser(follower):
						s=self.user_sim[user][follower]
						ug = self.P[self.rg.user[follower]]
						social_term_p += s * (p-ug)

				#update latent vectors
				self.P[u] += self.config.lr*(error*q- self.config.beta*(social_term_p+social_term_m) -self.config.lambdaP*p)
				self.Q[i] += self.config.lr*(error*p-self.config.lambdaQ*q)

				self.loss +=  0.5*self.config.beta * social_term_loss

			self.loss+=0.5*self.config.lambdaP*(self.P*self.P).sum() + 0.5*self.config.lambdaQ*(self.Q*self.Q).sum()

			iteration += 1
			if self.isConverged(iteration):
				break



if __name__ == '__main__':
	srg=SocialReg()
	srg.train_model()
	srg.show_rmse()