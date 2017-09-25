#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量

from mf import MF

class FunkSVDwithR(MF):
	"""
	docstring for FunkSVDwithR
	implement the FunkSVD with regularization
	http://sifter.org/~simon/journal/20061211.html
	"""

	def __init__(self):#继承父类的方法
		super(FunkSVDwithR, self).__init__()
		# self.lr=0.01 # 0.01 92   0.02 0.85119
		self.init_model()


	# def init_model(self): 
	# 	super(FunkSVDwithR, self).init_model()

	def train_model(self):
		iteration = 0
		while iteration < self.config.maxIter:
			self.loss = 0
			for index,line in enumerate(self.rg.trainSet()):
				user, item, rating = line
				u = self.rg.user[user]
				i = self.rg.item[item]
				error = rating - self.predict(user,item)#self.predict(user,item)
				self.loss += error**2
				p,q = self.P[u],self.Q[i]
				#update latent vectors
				self.P[u] += self.config.lr*(error*q-self.config.lambdaP*p)
				self.Q[i] += self.config.lr*(error*p-self.config.lambdaQ*q)

			self.loss+=self.config.lambdaP*(self.P*self.P).sum() + self.config.lambdaQ*(self.Q*self.Q).sum()

			iteration += 1
			if self.isConverged(iteration):
				break

	

if __name__ == '__main__':
	bmf=FunkSVDwithR()
	bmf.train_model()
	bmf.predict_model()
	bmf.show_rmse()