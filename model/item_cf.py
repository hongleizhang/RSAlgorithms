#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量

from mf import MF
from utility.matrix import SimMatrix
from utility.similarity import cosine_sp

class ItemCF(MF):
	"""
	docstring for ItemCF
	implement the ItemCF

	Sarwar B, Karypis G, Konstan J, et al. Item-based collaborative filtering recommendation algorithms[C]//Proceedings of the 10th international conference on World Wide Web. ACM, 2001: 285-295.
	"""

	def __init__(self):#继承父类的方法
		super(ItemCF, self).__init__()
		self.config.n=200
		self.init_model()

	def init_model(self):
		self.item_sim=SimMatrix()

		for i_test in self.rg.testSet_i:
			for i_train in self.rg.item:
				if i_test!=i_train:
					if self.item_sim.contains(i_test,i_train):
						continue
					sim = cosine_sp(self.rg.get_col(i_test),self.rg.get_col(i_train))
					self.item_sim.set(i_test,i_train,sim)

	def predict(self,u,i):
		matchItems = sorted(self.item_sim[i].items(),key = lambda x:x[1],reverse=True)
		itemCount = self.config.n
		if itemCount > len(matchItems):
			itemCount = len(matchItems)

		sum,denom = 0,0
		for n in range(itemCount):
			similarItem = matchItems[n][0]
			if self.rg.containsUserItem(u,similarItem):
				similarity = matchItems[n][1]
				rating=self.rg.trainSet_u[u][similarItem]
				sum += similarity*(rating-self.rg.itemMeans[similarItem])
				denom += similarity
		if sum == 0:
			if not self.rg.containsItem(i):
				return self.rg.globalMean
			return self.rg.itemMeans[i]
		pred = self.rg.itemMeans[i]+sum/float(denom)
		return pred
		pass


if __name__ == '__main__':
	ic=ItemCF()
	print(ic.predict_model())
