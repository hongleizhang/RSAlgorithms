#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量

from mf import MF
from utility.matrix import SimMatrix
from utility.similarity import cosine_sp

class UserCF(MF):
	"""
	docstring for UserCF
	implement the UserCF
	
	Resnick P, Iacovou N, Suchak M, et al. GroupLens: an open architecture for collaborative filtering of netnews[C]//Proceedings of the 1994 ACM conference on Computer supported cooperative work. ACM, 1994: 175-186.
	"""

	def __init__(self):#继承父类的方法
		super(UserCF, self).__init__()
		self.config.n=10
		self.init_model()

	def init_model(self):
		self.user_sim=SimMatrix()

		for u_test in self.rg.testSet_u:
			for u_train in self.rg.user:
				if u_test!=u_train:
					if self.user_sim.contains(u_test,u_train):
						continue
					sim = cosine_sp(self.rg.get_row(u_test),self.rg.get_row(u_train))
					self.user_sim.set(u_test,u_train,sim)

	def predict(self,u,i):
		matchUsers = sorted(self.user_sim[u].items(),key = lambda x:x[1],reverse=True)
		userCount = self.config.n
		if userCount > len(matchUsers):
			userCount = len(matchUsers)

		sum,denom = 0,0
		for n in range(userCount):
			similarUser = matchUsers[n][0]
			if self.rg.containsUserItem(similarUser,i):
				similarity = matchUsers[n][1]
				rating=self.rg.trainSet_u[similarUser][i]
				sum += similarity*(rating-self.rg.userMeans[similarUser])
				denom += similarity
		if sum == 0:
			if not self.rg.containsUser(u):
				return self.rg.globalMean
			return self.rg.userMeans[u]
		pred = self.rg.userMeans[u]+sum/float(denom)
		return pred


if __name__ == '__main__':
	uc=UserCF()
	print(uc.predict_model())
