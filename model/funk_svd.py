# encoding:utf-8
import sys

sys.path.append("..")
from mf import MF


class FunkSVD(MF):
    """
    docstring for FunkSVD
    implement the FunkSVD without regularization

    http://sifter.org/~simon/journal/20061211.html
    """

    def __init__(self):
        super(FunkSVD, self).__init__()
        # self.init_model(0)

    def train_model(self, k):
        super(FunkSVD, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]
                # update latent vectors
                self.P[u] += self.config.lr * error * q
                self.Q[i] += self.config.lr * error * p

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':

    rmses = []
    bmf = FunkSVD()
    # print(bmf.rg.trainSet_u[1])
    for i in range(bmf.config.k_fold_num):
        bmf.train_model(i)
        rmse, mae = bmf.predict_model()
        rmses.append(rmse)
    print(rmses)
    # bmf.config.k_current = 1
    # print(bmf.rg.trainSet_u[1])
    # bmf.train_model()
    # bmf.predict_model()
