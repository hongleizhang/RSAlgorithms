
class ConfigX(object):
	"""
	docstring for ConfigX

	configurate the global parameters and hyper parameters

	"""

	def __init__(self):
		super(ConfigX, self).__init__()

		self.rating_path='../data/ft_ratings.txt'
		self.trust_path='../data/ft_trust.txt'
		self.sep=' '
		self.random_state=0
		self.size=0.8
		self.min_val=.5#0.5
		self.max_val=4.0#4.0

		#HyperParameter
		self.factor=15 #隐含因子个数
		self.threshold=1e-4 #收敛的阈值
		self.lr=0.02 #学习率
		self.maxIter=500
		self.lambdaP=0.02
		self.lambdaQ=0.02
		self.beta=0.1 #偏置项系数


		