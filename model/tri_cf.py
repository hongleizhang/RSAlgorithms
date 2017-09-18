#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量
import numpy as np
from mf import MF
from utility.matrix import SimMatrix
from utility.similarity import cosine_sp

class TriCF(MF):
	"""
	docstring for TriCF
	
	"""
	def __init__(self):
		super(TriCF, self).__init__()
		self.lr=0.01
		self.config.alpha=0.01
		self.config.beta=0.01
		self.config.user_near_num=10
		self.config.item_near_num=10
		self.init_model()


	def init_model(self): 
		super(TriCF, self).init_model()
		from collections import defaultdict
		self.user_sim = SimMatrix()
		self.item_sim = SimMatrix()
		print('constructing user-user similarity matrix...')
		for u1 in self.rg.user:
			for u2 in self.rg.user:
				if u1!=u2:
					if self.user_sim.contains(u1,u2):
						continue
					sim = cosine_sp(self.rg.get_row(u1),self.rg.get_row(u2))
					self.user_sim.set(u1,u2,sim)


		print('constructing item-item similarity matrix...')
		for i1 in self.rg.item:
			for i2 in self.rg.item:
				if i1!=i2:
					if self.item_sim.contains(i1,i2):
						continue
					sim = cosine_sp(self.rg.get_col(i1),self.rg.get_col(i2))
					self.item_sim.set(i1,i2,sim)



	def train_model(self):
		print('training model...')
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
				
				matchUsers = sorted(self.user_sim[user].items(),key = lambda x:x[1],reverse=True)[:self.config.user_near_num]
				matchItems = sorted(self.item_sim[item].items(),key = lambda x:x[1],reverse=True)[:self.config.item_near_num]

				self.config.user_near_num = len(matchUsers) if self.config.user_near_num > len(matchUsers) \
					else self.config.user_near_num
				matchUsers=matchUsers[:self.config.user_near_num]

				self.config.item_near_num = len(matchItems) if self.config.item_near_num > len(matchItems) \
					else self.config.item_near_num
				matchItems=matchItems[:self.config.item_near_num]

				
				u_near_sum,u_near_total=np.zeros((self.config.factor)),0.0
				for x in matchUsers:
					near_user,sim_value=x
					near_user_id=self.rg.user[near_user]
					u_near_sum += sim_value*(self.P[near_user_id] - self.P[u])
					u_near_total+=sim_value*(sum(pow((self.P[near_user_id] - self.P[u]),2)))

				i_near_sum,i_near_total=np.zeros((self.config.factor)),0.0
				for x in matchItems:
					near_item,sim_value=x
					near_item_id=self.rg.item[near_item]
					i_near_sum += sim_value*(self.Q[near_item_id] - self.Q[i])
					i_near_total+=sim_value*(sum(pow((self.Q[near_item_id] - self.Q[i]),2)))

				self.P[u] += self.config.lr*(error*q-self.config.alpha*u_near_sum-self.config.lambdaP*p)
				self.Q[i] += self.config.lr*(error*p-self.config.beta*i_near_sum-self.config.lambdaQ*q)

				self.loss+=self.config.alpha*u_near_total+self.config.beta*i_near_total
			

			self.loss+=self.config.lambdaP*(self.P*self.P).sum() + self.config.lambdaQ*(self.Q*self.Q).sum()

			iteration += 1
			if self.isConverged(iteration):
				break



if __name__ == '__main__':
	tc=TriCF()
	tc.train_model()
	# srg.show_rmse()