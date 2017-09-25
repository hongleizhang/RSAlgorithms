#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量
import numpy as np
from mf import MF
from utility.tools import sigmoid_2
from utility.similarity import cosine_improved_sp
from reader.trust import TrustGetter

class TrustWalker(MF):
	"""
	docstring for TrustWalker
	
	Jamali M, Ester M. Trustwalker: a random walk model for combining trust-based and item-based recommendation[C]//Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2009: 397-406.
	"""
	def __init__(self):
		super(TrustWalker, self).__init__()
		np.random.seed(0)
		self.tg=TrustGetter()
		self.init_model()


	def init_model(self): 
		self.p=1.0
		pass


	def single_random_walk(self,user=5,item=3,k=0):
		print(user,item,k)
		print('%s%d'%('k=',k))
		if self.rg.containsUserItem(user,item): #判断用户user是否对item评分，若评分则返回r
			return self.p,self.rg.trainSet_u[user][item]
		else:
			rand_num=np.random.rand(1) #获取随机数
			#计算停止概率
			stop_prob,max_item,p_j=self.get_stop_prob(user,item,k)
			print('stop probability:'+str(stop_prob))
			# print('%s%d'%('stop probbability:',stop_prob))
			#停止游走的概率
			print(rand_num,stop_prob)
			if rand_num < stop_prob or k >= 6: #（不超过6步）
				#获取相似项目j，然后返回r(u,j)
				rating=self.rg.trainSet_u[user][max_item]
				self.p=self.p*stop_prob*p_j
				return (self.p,rating)
			#获取继续游走的概率
			else: 
				#得到下一个游走的用户v
				next_user,tu_prob=self.get_followee_user(user) #对于user在信任网络中没有朋友的情况
				print('next step user is:'+str(next_user))
				if next_user==None: #不存在下一个用户的情况
					_,max_item,p_j=self.get_stop_prob(user,item,-1) #k=-1无意义
					if max_item==0: #没有一个用户并且不存在相似的用户的情况
						return self.p,0
					rating=self.rg.trainSet_u[user][max_item]
					self.p=self.p*p_j
					return (self.p,rating)

				self.p=self.p*(1-stop_prob)*tu_prob
				k+=1
				return self.single_random_walk(user=next_user,item=item,k=k) #特么忘了return 当然返回None啦

	def get_followee_user(self,user):
		p=0
		followees=list(self.tg.get_followees(user))
		num_foll=len(followees)
		if num_foll==0:
			return None,0
		#随机选一个
		ind=np.random.randint(num_foll)
		p=1.0/num_foll
		return followees[ind],p

	# def get_max_item(self,user,item):
	# 	u_items=self.rg.user_rated_items(user)
	# 	sum_sim=0.0
	# 	max_sim=0
	# 	max_prob_item=0
	# 	if len(u_items)==0:
	# 		return 0,0
	# 	print(u_items)
	# 	for i,u_item in enumerate(u_items):
	# 		sim=self.get_sim(item,u_item)
	# 		sum_sim+=sim
	# 		if sim>max_sim:
	# 			max_sim=sim
	# 			max_prob_item=u_item
	# 	return max_prob_item,max_sim/sum_sim

	def get_stop_prob(self,user,item,k):
		p=1.0
		sum_sim=0.0 #用于计算P(Y=j)-分母
		max_sim=0 #用于计算P(Y=j)-分子
		max_prob=0.0 #用于计算停止概率
		max_prob_item=0 #用于方便选择相似item-j
		if k==0:#k==0时，停留概率为0
			self.p=1.0
			return 0,0,0

		param=sigmoid_2(k)

		u_items=self.rg.user_rated_items(user)
		print(u_items)
		if len(u_items)==0:
			return 0,0,0
		for u_item in u_items:
			sim=self.get_sim(item,u_item)
			sum_sim+=sim
			prob=sim*param
			if prob>max_prob:
				max_sim=sim
				max_prob=prob
				max_prob_item=u_item
		return max_prob,max_prob_item,max_sim/sum_sim #返回停止概率，最相似item，选择item j的概率

	def get_sim(self,item1,item2):
		return cosine_improved_sp(self.rg.get_col(item1), self.rg.get_col(item2))


if __name__ == '__main__':
	tw=TrustWalker()
	s=tw.single_random_walk(16,235)  #1,8 由于user1没有trust信息，所以不存在其他用户
	print(s)
