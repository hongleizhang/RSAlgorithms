class ConfigX(object):
    """
    docstring for ConfigX

    configurate the global parameters and hyper parameters

    """

    def __init__(self):
        super(ConfigX, self).__init__()

        self.rating_path = '../data/ft_ratings.txt'
        self.trust_path = '../data/ft_trust.txt'
        self.train_file = ''
        self.test_file = ''
        self.is_cross_validation = False
        self.sep = ' '
        self.random_state = 0
        self.size = 0.8  # 0.8 0.7 0.6 0.5 0.4
        self.min_val = 0.5  # 0.5 1.0
        self.max_val = 4.0  # 4.0 5.0

        # HyperParameter
        self.coldUserRating = 5  # 用户评分数少于5条定为cold start users 5,10
        self.hotUserRating = 30  # 397个，50 40个， 30
        self.factor = 10  # 隐含因子个数户 或者5
        self.threshold = 1e-4  # 收敛的阈值
        self.lr = 0.01  # 学习率
        self.maxIter = 100
        self.lambdaP = 0.001  # 0.02
        self.lambdaQ = 0.001  # 0.02

    def set_rating_path(self, path):
        self.rating_path = path

    def set_train_file(self, file):
        self.train_file = file

    def set_test_file(self, file):
        self.test_file = file

    def set_cross_validation(self, is_cross=False):
        self.is_cross_validation = is_cross
