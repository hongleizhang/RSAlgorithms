#encoding:utf-8
import sys
sys.path.append("..") #将该目录加入到环境变量
import numpy as np
from mf import MF
from reader.trust import TrustGetter

class LOCABAL(MF):
	"""
	docstring for LOCABAL
	
	Tang et al. 2013
	
	"""
	def __init__(self):
		super(LOCABAL, self).__init__()
		self.config.alpha=0.01
		self.config.lambdaH=0.01
		self.tg=TrustGetter()
		self.init_model()

	def init_model(self): 
		super(LOCABAL, self).init_model()
		user_length=self.rg.get_train_size()[0]
		self.H=np.random.rand(user_length, user_length) #/(self.config.factor**0.5)  # social correlation matrix
