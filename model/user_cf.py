# encoding:utf-8
import sys

sys.path.append("..")
from prettyprinter import cpprint
from mf import MF
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp


class UserCF(MF):
    """
    docstring for UserCF
    implement the UserCF

    Resnick P, Iacovou N, Suchak M, et al. GroupLens: an open architecture for collaborative filtering of netnews[C]//Proceedings of the 1994 ACM conference on Computer supported cooperative work. ACM, 1994: 175-186.
    """

    def __init__(self):
        super(UserCF, self).__init__()
        self.config.n = 10
        # self.init_model(k)

    def init_model(self, k):
        super(UserCF, self).init_model(k)
        self.user_sim = SimMatrix()

        for u_test in self.rg.testSet_u:
            for u_train in self.rg.user:
                if u_test != u_train:
                    if self.user_sim.contains(u_test, u_train):
                        continue
                    sim = pearson_sp(self.rg.get_row(u_test), self.rg.get_row(u_train))
                    self.user_sim.set(u_test, u_train, sim)

    def predict(self, u, i):
        matchUsers = sorted(self.user_sim[u].items(), key=lambda x: x[1], reverse=True)
        userCount = self.config.n
        if userCount > len(matchUsers):
            userCount = len(matchUsers)

        sum, denom = 0, 0
        for n in range(userCount):
            similarUser = matchUsers[n][0]
            if self.rg.containsUserItem(similarUser, i):
                similarity = matchUsers[n][1]
                rating = self.rg.trainSet_u[similarUser][i]
                sum += similarity * (rating - self.rg.userMeans[similarUser])
                denom += similarity
        if sum == 0:
            if not self.rg.containsUser(u):
                return self.rg.globalMean
            return self.rg.userMeans[u]
        pred = self.rg.userMeans[u] + sum / float(denom)
        return pred


if __name__ == '__main__':
    uc = UserCF()
    uc.init_model(0)
    print(uc.predict_model())
    print(uc.predict_model_cold_users())
    uc.init_model(1)
    print(uc.predict_model())
    print(uc.predict_model_cold_users())
