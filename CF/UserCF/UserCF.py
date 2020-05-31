import random
import time
from math import sqrt, log
from tools import timer


class Solution():
    def __init__(self, k, count):
        # 读入的原始数据集
        self.data = []

        self.test_dct = {}
        self.train_dct = {}

        # 用于记录用户之间的相似度，格式为 { user1:{ user2:value, user3:value...}, ... }
        self.user_similarity_martix = dict()

        # 记录每个用户评分的物体 { 1:{2,4,6...} , ... }
        self.user_item_dct = dict()
        # 记录每个物体评过分的用户 { 1:{1,2,4...} , ... }
        self.item_user_dct = dict()
        # 相似用户数量
        self.k = k
        # 10个推荐物品
        self.count = count

    @timer
    def readData(self):
        # 读取data.dat文件，文件内容为user,item
        # 本次实验不涉及用户评分，所以不需要存储分数
        with open('../../DataSet/ml-latest-small/ratings.csv', encoding='UTF8') as fb:
            # with open('../../ml-1m/ratings.dat', encoding='UTF8') as fb:
            fb.readline()
            for line in fb:
                line = line.strip()
                line = line.split(',')
                self.data.append((int(line[0]), int(line[1])))

    @timer
    def splitData(self, pivot=0.75):
        # 按照1：3的比例划分测试集和训练集
        # 记录训练集和测试集的数量
        # 划分数据集
        random.seed(random.randint(0, 10000))
        for user, item in self.data:
            if random.random() >= pivot:
                self.test_dct.setdefault(user, set())
                self.test_dct[user].add(item)
            else:
                self.train_dct.setdefault(user, set())
                self.train_dct[user].add(item)

    @timer
    def builtDict(self):
        # 此函数用于建立起user_item_dct和item_user_dct
        # 两个dct格式类似于 { user1:{ a,s,d...} ,user2:{q,w,e...}, ...}

        # 用户物品倒排表就是训练集
        self.user_item_dct = self.train_dct

        for user, item_lst in self.train_dct.items():
            # u 用户，i 物品
            for item in item_lst:
                self.item_user_dct.setdefault(item, set())
                self.item_user_dct[item].add(user)

    @timer
    def UserCF(self):
        # 生成相似矩阵,使用字典存储，格式为 { user:{ user1: 相似度1, user2:相似度2}... ,}
        # 计算步骤
        # 1. 从物品对用户的倒排表中取出所有的用户列表
        # 2. 生成ui,uj用户的同样感兴趣物品数,不需要特别记录共同感兴趣的物品是哪个，所以只取物品数即可
        # 3. 按照公式计算出最后的值
        for user_set in self.item_user_dct.values():
            for ui in user_set:
                self.user_similarity_martix.setdefault(ui, dict())
                for uj in user_set:
                    if ui == uj:
                        continue
                    self.user_similarity_martix[ui].setdefault(uj, 0)
                    # 同样感兴趣的物品数+1
                    self.user_similarity_martix[ui][uj] += 1

        # 得到每个用户之间感兴趣的物品数之后，利用余弦相似度来计算相同的值
        # 对当前ui用户，取得他和有关用户的字典集合
        for ui, related_user in self.user_similarity_martix.items():
            for uj in related_user:
                self.user_similarity_martix[ui][uj] /= sqrt(
                    len(self.user_item_dct[ui]) * len(self.user_item_dct[uj]))

    def UserIIF(self):
        # 改进的userCF算法，只有相似度计算有改变
        for user_set in self.item_user_dct.values():
            for ui in user_set:
                self.user_similarity_martix.setdefault(ui, dict())
                for uj in user_set:
                    if ui == uj:
                        continue
                    self.user_similarity_martix[ui].setdefault(uj, 0)
                    # 同样感兴趣的物品数+1
                    self.user_similarity_martix[ui][uj] += 1 / log(1 + len(user_set))

        # 得到每个用户之间感兴趣的物品数之后，利用改进的算法计算相似度
        # 对当前ui用户，取得他和有关用户的字典集合
        for ui, related_user in self.user_similarity_martix.items():
            for uj in related_user:
                self.user_similarity_martix[ui][uj] /= sqrt(
                    len(self.user_item_dct[ui]) * len(self.user_item_dct[uj]))

    def recommendItem(self, u):
        # 最后的推荐算法，给user推荐物品
        # 首先计算出用户u对于物品j的兴趣
        # 计算方法为：i属于用户喜欢的物品的集合，j是和物品j
        # {物品：兴趣值}
        rank = {}
        user_item_lst = self.user_item_dct[u]

        # v=similar user, wuv=similar factor，sval是用户之间的相似度
        for v, sval in sorted(self.user_similarity_martix[u].items(), key=lambda x: x[1], reverse=True)[0:self.k]:
            for item in self.user_item_dct[v]:
                if item in user_item_lst:
                    continue
                rank.setdefault(item, 0)
                rank[item] += sval
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:self.count]

    @timer
    def evaluateModel(self):
        # 计算精确率，召回率，覆盖率
        hit_item = 0  # 命中的物品
        all_item = len(self.item_user_dct.keys())  # 所有物品
        test_item_num = 0  # 所有测试集中的物品
        recommend_item_set = set()  # 所有推荐物品集合
        recommend_item_num = 0  # 所有推荐物品的数量

        # 遍历测试集，对每个用户,生成推荐列表
        for user, _ in self.train_dct.items():
            # 推荐结果存储在self.recommend_lst中
            recommend_lst = self.recommendItem(user)
            # 获得测试集的物品列表
            test_item_lst = self.test_dct.get(user, {})
            # 验证推荐物品是否在测试集中
            for item, val in recommend_lst:
                if item in test_item_lst:
                    hit_item += 1  # 命中物品数+1
                recommend_item_set.add(item)
            recommend_item_num += self.count
            test_item_num += len(test_item_lst)

        precision = hit_item / (1.0 * recommend_item_num)
        recall = hit_item / (1.0 * test_item_num)
        coverage = len(recommend_item_set) / all_item

        # 计算流行度
        item_popularity = dict()
        for user, items in self.train_dct.items():
            for item in items:
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in self.train_dct.keys():
            rank = self.recommendItem(user)
            for item, pui in rank:
                ret += log(1 + item_popularity[item])
                n += 1
        ret /= n * 1.0

        return precision, recall, coverage, ret


@timer
def execute_model(k, count, times=3):
    # k是算法中涉及到多少人，count是推荐数量
    p, r, c, po = 0, 0, 0, 0
    for i in range(times):
        print('-' * 30)
        s = Solution(k, count)
        s.readData()
        s.splitData()
        s.builtDict()
        s.UserIIF()
        tp, tr, tc, tpo = s.evaluateModel()
        p += tp
        r += tr
        c += tc
        po += tpo
        print('-' * 30)
    p /= times
    r /= times
    c /= times
    po /= times
    # with open('UserIIFresult.txt', 'a+') as fb:
    #     fb.write("{:>2} {:>7} {:>12.4f} {:>9.4f} {:>11.4f} {:>13.4f}\n".format(k, count, p, r, c, po))


if __name__ == '__main__':
    # with open('UserIIFresult.txt','w') as fb:
    #     fb.write('    '.join(['k','count','precision','recall','coverage','popularity'])+'\n')
    for k in [5,10,20,30,50]:
        execute_model(k,5,3)
