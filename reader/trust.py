# encoding:utf-8
import sys

sys.path.append("..")
import numpy as np
import os
from collections import defaultdict

from configx.configx import ConfigX


class TrustGetter(object):
    """
    docstring for TrustGetter
    read trust data and save the global parameters

    """

    def __init__(self):
        super(TrustGetter, self).__init__()
        self.config = ConfigX()

        self.user = {}  # used to store the order of users
        self.relations = self.get_relations()
        self.followees = defaultdict(dict)
        self.followers = {}
        self.matrix_User = {}
        self.matrix_Item = {}
        self.generate_data_set()

    def generate_data_set(self):
        triple = []
        for line in self.relations:
            userId1, userId2, weight = line
            # add relations to dict
            if not userId1 in self.followees:
                self.followees[userId1] = {}
            self.followees[userId1][userId2] = weight
            if not userId2 in self.followers:
                self.followers[userId2] = {}
            self.followers[userId2][userId1] = weight
            # order the user
            if not userId1 in self.user:
                userid1 = self.user[userId1] = len(self.user)
            if not userId2 in self.user:
                userid2 = self.user[userId2] = len(self.user)
            if not userid1 in self.matrix_User:
                self.matrix_User[userid1] = {}
            if not userid2 in self.matrix_User:
                self.matrix_Item[userid2] = {}
            self.matrix_User[userid1][userid2] = weight
            self.matrix_Item[userid2][userid1] = weight

    def get_relations(self):
        if not os.path.isfile(self.config.trust_path):
            print("the format of trust data is wrong")
            sys.exit()
        with open(self.config.trust_path, 'r') as f:
            for index, line in enumerate(f):
                u_from, u_to, t = line.strip('\r\n').split(self.config.sep)
                yield (int(u_from), int(u_to), float(t))

    def get_followees(self, u):
        if u in self.followees:
            return self.followees[u]
        else:
            return {}

    def get_followers(self, u):
        if u in self.followers:
            return self.followers[u]
        else:
            return {}

    def weight(self, u, k):
        if u in self.followees and k in self.followees[u]:
            return self.followees[u][k]
        else:
            return 0


if __name__ == '__main__':
    tg = TrustGetter()
    s = tg.get_followees(2).keys()
    print(s)
