#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量
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
		self.config.lr=0.01
		self.config.alpha=1 #0.8溢出
		self.tg=TrustGetter() #loading trust data
		self.init_model()


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

				# total_weight=0
				social_term = np.zeros(self.config.factor)
				followees = self.tg.get_followees(user) #获得u所关注的用户列表
				for followee in followees:
					weight= followees[followee]
					if self.rg.containsUser(followee):
						uk = self.P[self.rg.user[followee]]
						social_term += weight * uk
				social_term = p - social_term


				social_term_a=np.zeros(self.config.factor)
				followers = self.tg.get_followers(user)
				for follower in followers:
					if self.rg.containsUser(follower):
						uv = self.P[self.rg.user[follower]]
						social_term_m=np.zeros(self.config.factor)
						followees = self.tg.get_followees(follower) #获得u所关注的用户列表
						for followee in followees:
							weight= followees[followee]
							if self.rg.containsUser(followee):
								uw = self.P[self.rg.user[followee]]
								social_term_m += weight * uw
						social_term_a+=uv - social_term_m



				#update latent vectors
				self.P[u] += self.config.lr*(error*q-self.config.alpha*social_term+self.config.alpha*social_term_a -self.config.lambdaP*p) #
				self.Q[i] += self.config.lr*(error*p-self.config.lambdaQ*q)

				self.loss +=  self.config.alpha * social_term.dot(social_term).sum()

			self.loss+=self.config.lambdaP*(self.P*self.P).sum() + self.config.lambdaQ*(self.Q*self.Q).sum()

			iteration += 1
			if self.isConverged(iteration):
				break


if __name__ == '__main__':
	smf=SocialMF()
	smf.train_model()