# ReSysCode

《推荐系统实践》中介绍的算法代码实现

数据集：ml-latest-small https://grouplens.org/datasets/movielens/

算法都封装在Solution里面，最外面有一个函数execute_model用来执行整个过程。

整个过程为：

1. 划分数据集
2. 处理数据集，例如生成倒排表等
3. 关键算法，例如书中介绍了UserCF和UserIIF，两者的用户相似度计算公式不同，所以这里可能会有多个函数供选择
4. 生成推荐列表的函数
5. 评估算法的指标，一共四个指标：precision，recall，coverage，popularity

