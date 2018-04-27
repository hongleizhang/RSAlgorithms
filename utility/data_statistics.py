# encoding:utf-8
import sys

sys.path.append("..")

from metrics.metric import Metric
from utility.tools import denormalize, sigmoid
from reader.rating import RatingGetter
from reader.trust import TrustGetter
from configx.configx import ConfigX


class DataStatis(object):
    """docstring for DataStatis"""

    def __init__(self):
        super(DataStatis, self).__init__()
        self.config = ConfigX()
        self.rg = RatingGetter()  # loading raing data
        self.tg = TrustGetter()
        self.cold_rating = 0
        self.cold_social = 0
        self.cold_rating_social = 0
        self.cold_rating_warm_social = 0
        self.warm_rating_cold_social = 0
        self.warm_rating_warm_social = 0

    def getDataStatis(self):
        # print(self.rg.dataSet_u[2])
        for user in self.rg.dataSet_u:
            # print(user)
            num_rating = len(self.rg.dataSet_u[user])
            num_social = len(self.tg.followees[user])

            if (num_rating < 5):
                self.cold_rating += 1
            if (num_social < 5):
                self.cold_social += 1

            if (num_rating < 5 and num_social < 5):
                self.cold_rating_social += 1
            if (num_rating < 5 and num_social >= 5):
                self.cold_rating_warm_social += 1
            if (num_rating >= 5 and num_social <= 5):
                self.warm_rating_cold_social += 1
            if (num_rating >= 5 and num_social >= 5):
                self.warm_rating_warm_social += 1

        pass


if __name__ == '__main__':
    ds = DataStatis()
    ds.getDataStatis()
    print(ds.cold_rating)
    print(ds.cold_social)
    print(ds.cold_rating_social)
    print(ds.cold_rating_warm_social)
    print(ds.warm_rating_cold_social)
    print(ds.warm_rating_warm_social)
