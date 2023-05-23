import os
from config import *
from utils import *
import math
import time
import numpy as np

class CF_user():
    def __init__(self, train_p, test_p):
        self.if_build = False
        self.if_train = False
        self.if_test = False
        self.rated_num = 0  # 总评分数
        self.user_matrix = []  # 存储用户对物品的评分 [{itemid: score,...},...]
        self.user_ave = []  # 用户对物品的评分准则(对物品评分的平均数)[u1,u2,...]
        self.sim_matrix_user = None  # user 的相似矩阵（稀疏）lil_matrix
        self.item_list = set() # 物品列表，使用set是为了去重
        self.inverted_item_list = dict() # 物品列表的反向索引
        self.r = []  # predicted matirx
        self.train_p = train_p
        self.test_p = test_p
        self.total_sim = 0 # 总相似度数，即相似度矩阵的一半
        self.sim1 = 'sim1.user'
        self.sim2 = 'sim2.user'
        self.mid = 0
        self.if_2 = False
        self.now_sim = 0

    def static_analyse(self):
        # use after build
        print(f"user number: {len(self.user_ave)}")
        print(f"item number: {len(self.item_list)}")
        print(f"rated number: {self.rated_num}")
        print(f"total sim number: {self.total_sim}")

    def build(self, path): # 用于构建user的评分矩阵
        print("Building Rating Matrix...")
        user_item = file_read(path) # 读取训练集
        temp_count = 0
        score_count = 0
        user_id = None
        user_item_num = None
        for i in user_item:
            # 大部分行为评分，因此使用is not None来增加hit概率
            if user_id is not None:
                now_item, now_score = [int(k) for k in i.split()]
                self.item_list.add(now_item)
                score_count += now_score
                self.user_matrix[user_id][now_item] = now_score
                temp_count += 1
                if temp_count == user_item_num:
                    self.user_ave[user_id] = score_count / temp_count
                    user_id = None
            else:
                score_count = 0
                user_id, user_item_num = [int(j) for j in i.split('|')[:2]]
                self.rated_num += user_item_num
                while len(self.user_matrix) < user_id + 1:
                    self.user_matrix.append({})
                    self.user_ave.append(0)
                temp_count = 0

        user_item.close()
        self.item_list = list(self.item_list)
        self.item_list.sort()
        for x in range(len(self.item_list)):
            self.inverted_item_list[self.item_list[x]] = x
        self.if_build = True
        self.total_sim = int((pow(len(self.user_ave), 2)-len(self.user_ave))/2)
        print("Build Rating Matrix Success!")

    def train(self):
        start = time.time()
        print(f"Start train at {time.asctime(time.localtime(start))}")
        self.sim_matrix_user = [{} for _ in range(len(self.user_ave))]
        self.now_size = 0
        self.mid = int(len(self.user_ave) / 2)
        count = 0

        def calculate_similarity(i, j, item, rirj, ri2, rj2):
            m1 = self.user_matrix[i][item] - self.user_ave[i]
            m2 = self.user_matrix[j][item] - self.user_ave[j]
            rirj += m1 * m2
            ri2 += pow(m1, 2)
            rj2 += pow(m2, 2)
            return rirj, ri2, rj2

        # 遍历所有用户对，计算相似度
        for i in range(len(self.user_ave)):
            for j in range(i + 1, len(self.user_ave)):
                rirj, ri2, rj2 = 0
                # 由于用户对物品的评分矩阵是稀疏的，因此使用len来判断哪个用户的评分矩阵更稀疏
                # 而且能够减少计算量
                if len(self.user_matrix[i]) <= len(self.user_matrix[j]):
                    for item in self.user_matrix[i]:
                        if self.user_matrix[j].get(item) is not None:
                            r = calculate_similarity(i, j, item, rirj, ri2, rj2)
                else:
                    for item in self.user_matrix[j]:
                        if self.user_matrix[i].get(item) is not None:
                            r = calculate_similarity(i, j, item, rirj, ri2, rj2)

                if r[1] == 0 or r[2] == 0:
                    self.sim_matrix_user[i][j] = 0
                    self.sim_matrix_user[j][i] = 0
                else:
                    self.sim_matrix_user[i][j] = (r[0] / math.sqrt(r[1] * r[2]))
                    self.sim_matrix_user[j][i] = self.sim_matrix_user[i][j]

                count += 1
                if count % pow(10, print_per) == 0:
                    now_time = time.time()
                    print(f"Now time: {time.asctime(time.localtime(now_time))}, batch time: {now_time - start}, {count}/{self.total_sim}")

        for i in range(len(self.user_ave)):
            # 对相似度进行排序，按照相似度从大到小排序
            self.sim_matrix_user[i] = dict(sorted(self.sim_matrix_user[i].items(), key=lambda x: x[1], reverse=True))

        now_time = time.time()
        print(f"Begin save at {time.asctime(time.localtime(now_time))}")
        save_class(self.sim_matrix_user, Save_path, os.path.join(Save_path, 'sim.user'))
        # temp = []
        # for i in range(self.mid):
        #     temp.append(self.sim_matrix_user[i])
        #     self.sim_matrix_user[i] = {}
        # save_class(temp, Save_path, os.path.join(Save_path, self.sim1))
        # temp = []
        # for i in range(self.mid, len(self.user_ave)):
        #     temp.append(self.sim_matrix_user[i])
        #     self.sim_matrix_user[i] = {}
        # save_class(temp, Save_path, os.path.join(Save_path, self.sim2))
        self.if_train = True
        end = time.time()
        print(f"Now is: {time.asctime(time.localtime(end))}, train time cost is {end - start}.")

    def predict(self, user, item_j):
        x = 0
        y = 0
        count = 0
        item_j = self.inverted_item_list[item_j]
        if self.if_2:
            user = user - self.mid
        for u in self.sim_matrix_user[user]:
            if self.user_matrix[u].get(item_j) is not None and self.sim_matrix_user[user][u] >= Thresh:
                count += 1
                y += self.sim_matrix_user[user][u]
                x += self.sim_matrix_user[user][u] * self.user_matrix[u][item_j]
            if count == topn:
                break

        if y == 0:
            return 0
        else:
            return x / y

    def test(self, path):
        start = time.time()
        print(f"Start test at {time.asctime(time.localtime(start))}")
        data = file_read(path)
        user_id = None
        user_item_num = None
        temp_count = 0
        test_count = 0
        self.if_2 = False
        self.sim_matrix_user = load_class(Save_path, os.path.join(Save_path, self.sim1))
        for i in data:
            if test_count % 1000 == 0 and test_count != 0:
                now_time = time.time()
                print(f"Now is: {time.asctime(time.localtime(now_time))}, train time cost is {now_time - start}, test count: {test_count}.")
            if user_id is None:
                user_id = int(i.split('|')[0], 10)
                if user_id >= self.mid and (not self.if_2):
                    del self.sim_matrix_user[:]
                    self.sim_matrix_user = load_class(Save_path, os.path.join(Save_path, self.sim2))
                    print("load over")
                    self.if_2 = True
                while len(self.r) < user_id + 1:
                    self.r.append([])
                user_item_num = int(i.split('|')[1], 10)
                temp_count = 0
                test_count += 1
            else:
                now_item = int(i.split()[0], 10)
                if self.inverted_item_list.get(now_item) is None:
                    p = self.user_ave[user_id]
                else:
                    p = self.predict(user_id, now_item)
                self.r[user_id].append((now_item, p))
                temp_count += 1
                if temp_count == user_item_num:
                    user_id = None
                test_count += 1
        del self.sim_matrix_user[:]
        end = time.time()
        print('%s Test time cost = %fs' % (time.asctime(time.localtime(end)), end - start))
        self.if_test = True

