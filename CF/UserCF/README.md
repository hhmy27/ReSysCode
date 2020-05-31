# README

UserCF的步骤：

1. 计算用户之间的相似度
2. 根据相似用户喜欢的物品，来给当前用户做推荐

具体做法：

1. 建立倒排表方便计算

   ![image-20200530191033132](MDAssets/README/image-20200530191033132.png)

2. 计算用户相似矩阵

   ![image-20200530191056749](MDAssets/README/image-20200530191056749.png)

   （以UserCF为例）

3. 推荐函数

   ![image-20200530191120470](MDAssets/README/image-20200530191120470.png)

---

UserCF的相似度计算公式：

![image-20200531101554727](MDAssets/README/image-20200531101554727.png)

UserIIF的改进：

![image-20200531101612728](MDAssets/README/image-20200531101612728.png)

---



UserCF的运行结果

![image-20200531101427914](MDAssets/README/image-20200531101427914.png)

UserIIF的运行结果

![image-20200531101436431](MDAssets/README/image-20200531101436431.png)

---





