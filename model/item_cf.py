# encoding:utf-8
import sys

sys.path.append("..")
from prettyprinter import cpprint
from mf import MF
from utility.matrix import SimMatrix
from utility.similarity import cosine_sp, pearson_sp


class ItemCF(MF):
    """
    docstring for ItemCF
    implement the ItemCF

    Sarwar B, Karypis G, Konstan J, et al. Item-based collaborative filtering recommendation algorithms[C]//Proceedings of the 10th international conference on World Wide Web. ACM, 2001: 285-295.
    """

    def __init__(self):
        super(ItemCF, self).__init__()
        self.config.n = 50
        # self.init_model()

    def init_model(self, k):
        super(ItemCF, self).init_model(k)
        self.item_sim = SimMatrix()

        for i_test in self.rg.testSet_i:
            for i_train in self.rg.item:
                if i_test != i_train:
                    if self.item_sim.contains(i_test, i_train):
                        continue
                    sim = pearson_sp(self.rg.get_col(i_test), self.rg.get_col(i_train))
                    self.item_sim.set(i_test, i_train, sim)

    def predict(self, u, i):

        # item_sim=dict()
        # for i_train in self.rg.item:
        #     if i != i_train:
        #         if i_train in item_sim :
        #             continue
        #         sim=cosine_sp(self.rg.get_col(i), self.rg.get_col(i_train))
        #         item_sim[i_train]=sim

        matchItems = sorted(self.item_sim[i].items(), key=lambda x: x[1], reverse=True)
        itemCount = self.config.n
        if itemCount > len(matchItems):
            itemCount = len(matchItems)

        sum, denom = 0, 0
        for n in range(itemCount):
            similarItem = matchItems[n][0]
            if self.rg.containsUserItem(u, similarItem):
                similarity = matchItems[n][1]
                rating = self.rg.trainSet_u[u][similarItem]
                sum += similarity * (rating - self.rg.itemMeans[similarItem])
                denom += similarity
        if sum == 0:
            if not self.rg.containsItem(i):
                return self.rg.globalMean
            return self.rg.itemMeans[i]
        pred = self.rg.itemMeans[i] + sum / float(denom)
        # print('finished user:'+str(u)+" item:"+str(i))
        return pred
        pass


if __name__ == '__main__':
    ic = ItemCF()
    ic.init_model(0)
    print(ic.predict_model())
    print(ic.predict_model_cold_users())
    ic.init_model(1)
    print(ic.predict_model())
    print(ic.predict_model_cold_users())