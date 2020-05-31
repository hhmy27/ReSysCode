import csv
import random
from tools import timer
from math import sqrt, log


class Solution():
    def __init__(self, k, count):
        # 读入的原始数据集
        self.data = []

        self.test_dct = {}
        self.train_dct = {}
        # 推荐物品的字典，格式是 { item:value, ... }
        # self.recommend_dct = dict()
        # 最后的推荐列表，是按照兴趣值从大到小排序过后的列表,格式为[(item,value),(item,value)...]
        # self.recommend_lst = []
        # 用于记录用户之间的相似度，格式为 { user1:{ user2:value, user3:value...}, ... }
        self.item_similarity_matrix = dict()

        # 相似用户数量
        self.k = k
        # 10个推荐物品
        self.count = count

        # 记录每个用户评分的物体 { 1:{2,4,6...} , ... }
        self.user_item_dct = dict()
        # 记录每个物体评过分的用户 { 1:{1,2,4...} , ... }
        self.item_user_dct = dict()

    @timer
    def readData(self):
        # 读取data.dat文件，文件内容为user,item
        # 本次实验不涉及用户评分，所以不需要存储分数
        with open('../../DataSet/ml-latest-small/ratings.csv', encoding='UTF8') as fb:
            fb.readline()
            for line in fb:
                line = line.strip()
                line = line.split(',')
                # 转换成int，加快比较操作
                self.data.append((int(line[0]), int(line[1]), float(line[2])))

    @timer
    def splitData(self, pivot=0.75):
        # 按照1：3的比例划分测试集和训练集
        random.seed(random.randint(0, 10000))
        # 用户评分
        for user, item, rating in self.data:
            if random.random() >= pivot:
                self.test_dct.setdefault(user, dict())
                self.test_dct[user][item] = rating
            else:
                self.train_dct.setdefault(user, dict())
                self.train_dct[user][item] = rating
        # 节省空间
        del self.data

    @timer
    def builtDict(self):
        # 此函数用于建立起user_item_dct和item_user_dct
        # 两个dct格式类似于 { user1:{ a,s,d...} ,user2:{q,w,e...}, ...}
        # { Item1:{u1,u2..}, Item2:{u3,u6...} }

        # 用户物品倒排表就是训练集
        self.user_item_dct = self.train_dct

        for user, item_dct in self.user_item_dct.items():
            # u 用户，i 物品
            for item in item_dct.keys():
                self.item_user_dct.setdefault(item, set())
                self.item_user_dct[item].add(user)

    @timer
    def ItemCF(self):
        # 生成物品相似矩阵,使用字典存储，格式为 { user:{ user1: 相似度1, user2:相似度2}... ,}
        # 步骤：
        # 1.两两计算物品之间的相似度Wij
        # 2.Wij的计算方法是，喜欢i物品的用户和喜欢j物品的用户交集/根号下喜欢i物品的用户个数*喜欢j物品的用户个数

        for item1, users1 in self.item_user_dct.items():
            self.item_similarity_matrix.setdefault(item1, dict())
            for item2, users2 in self.item_user_dct.items():
                if item1 == item2:
                    continue
                if len(self.item_user_dct[item1]) == 0 or len(self.item_user_dct[item2]) == 0:
                    self.item_similarity_matrix[item1][item2] = 0
                    continue

                self.item_similarity_matrix[item1][item2] = len(
                    self.item_user_dct[item1] & self.item_user_dct[item2]) / sqrt(
                    len(self.item_user_dct[item1]) * len(self.item_user_dct[item2]))

    @timer
    def ItemCF_Norm(self):
        # 生成物品相似矩阵,使用字典存储，格式为 { user:{ user1: 相似度1, user2:相似度2}... ,}
        # 步骤：
        # 1.两两计算物品之间的相似度Wij
        # 2.Wij的计算方法是，喜欢i物品的用户和喜欢j物品的用户交集/根号下喜欢i物品的用户个数*喜欢j物品的用户个数

        for item1, users1 in self.item_user_dct.items():
            self.item_similarity_matrix.setdefault(item1, dict())
            for item2, users2 in self.item_user_dct.items():
                if item1 == item2:
                    continue
                if len(self.item_user_dct[item1]) == 0 or len(self.item_user_dct[item2]) == 0:
                    self.item_similarity_matrix[item1][item2] = 0
                    continue

                self.item_similarity_matrix[item1][item2] = len(
                    self.item_user_dct[item1] & self.item_user_dct[item2]) / sqrt(
                    len(self.item_user_dct[item1]) * len(self.item_user_dct[item2]))

        # ItemCF-Norm额外的一步，对相似矩阵做归一化
        for item1, items in self.item_similarity_matrix.items():
            # 取到相似度最大的值
            ms = max(items)
            for item2 in items:
                if item2 != 0:
                    # 归一化
                    item2 /= ms

    def recommendItem(self, u):
        # 每次找k部相似的电影，最后推荐count部
        rank = dict()
        watched_movies = self.user_item_dct[u]
        for movie, rating in watched_movies.items():
            # 相似电影和权重
            for related_movie, w in sorted(self.item_similarity_matrix[movie].items(), key=lambda x: x[1],
                                           reverse=True)[:self.k]:
                if related_movie in watched_movies.keys():
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:self.count]

    @timer
    def evaluateModel(self):
        # 计算精确率，召回率，覆盖率
        hit_item = 0  # 命中的物品
        all_item = len(self.item_user_dct.keys())  # 所有物品
        test_item_num = 0  # 所有测试集中的物品
        recommend_item_set = set()  # 所有推荐物品集合
        recommend_item_num = 0  # 所有推荐物品的数量

        # 遍历测试集，对每个用户,生成推荐列表
        for user, _ in self.user_item_dct.items():
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
        for user, items in self.user_item_dct.items():
            for item in items:
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in self.user_item_dct.keys():
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
        s.ItemCF_Norm()
        # s.recommendItem()
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
    with open('ItemCF_NormResult.txt', 'a+') as fb:
        fb.write("{:>2} {:>7} {:>12.4f} {:>9.4f} {:>11.4f} {:>13.4f}\n".format(k, count, p, r, c, po))


if __name__ == '__main__':
    with open('ItemCF_NormResult.txt', 'w') as fb:
        fb.write('    '.join(['k', 'count', 'precision', 'recall', 'coverage', 'popularity']) + '\n')
    for k in [5, 10, 20, 30, 50]:
        execute_model(k, 5, 1)
