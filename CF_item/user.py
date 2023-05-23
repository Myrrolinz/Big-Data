import os
from config import *
from utils import *
import math
import time
import numpy as np
from collections import defaultdict

class CF_user():
    def __init__(self, train_p, test_p):
        self.if_build = False
        self.if_train = False
        self.if_test = False
        self.rating_num = 0  # 总评分数
        self.user_matrix = defaultdict(dict)  # 存储用户对物品的评分 [{itemid: score,...},...]
        self.user_ave = []  # 用户对物品的评分准则(对物品评分的平均数)[u1,u2,...]
        self.sim_matrix_user = None  # user 的 相似矩阵（稀疏）lil_matrix
        self.item_list = set()
        self.change = dict()
        self.r = []  # predicted matirx
        self.train_p = train_p
        self.test_p = test_p
        self.total = 0
        self.sim1 = 'sim1.user'
        self.sim2 = 'sim2.user'
        self.mid = 0
        self.if_2 = False
        self.now_sim = 0

    def static_analyse(self):
        # use after build
        print(f"user number: {len(self.user_ave)}\nrating number: {self.rating_num}\nitem number: {len(self.item_list)}")

    def build(self, path): # 用于构建user的评分矩阵
        print("Building Rating Matrix...")
        user_item = file_read(path) # 读取训练集
        user_id = None
        user_item_num = None
        temp_count = 0
        score_count = 0
        for i in user_item:
            if user_id is None:
                score_count = 0
                user_id, user_item_num = [int(j) for j in i.split('|')[:2]]
                self.rating_num += user_item_num
                while len(self.user_matrix) < user_id + 1: #user矩阵增长
                    self.user_ave.append(0)
                self.rating_num += user_item_num
                temp_count = 0
            else:
                now_item, now_score = [int(k) for k in i.split()]
                self.item_list.add(now_item)
                score_count += now_score
                self.user_matrix[user_id][now_item] = now_score
                temp_count += 1
                if temp_count == user_item_num:
                    self.user_ave[user_id] = score_count / temp_count
                    user_id = None

        user_item.close()
        self.item_list = list(self.item_list)
        self.item_list.sort()
        for x in range(len(self.item_list)):
            self.change[self.item_list[x]] = x

        self.if_build = True
        print("Build Rating Matrix Success!\n")

    def train(self):
        start = time.time()
        print("%s : start train\n" % (time.asctime(time.localtime(start))))
        self.total = int((pow(len(self.user_ave), 2)-len(self.user_ave))/2)
        self.sim_matrix_user = [{} for _ in range(len(self.user_ave))]
        self.now_size = 0
        self.mid = int(len(self.user_ave) / 2)

        count = 0
        for i in range(len(self.user_ave)):
            len_i = len(self.user_matrix[i])
            for j in range(i+1, len(self.user_ave)):
                len_j = len(self.user_matrix[j])
                temp1 = 0
                temp2 = 0
                temp3 = 0
                if len_i <= len_j:
                    for item in self.user_matrix[i]:
                        if self.user_matrix[j].get(item) is not None:
                            m1 = self.user_matrix[i][item] - self.user_ave[i]
                            m2 = self.user_matrix[j][item] - self.user_ave[j]
                            temp1 += m1 * m2
                            temp2 += pow(m1, 2)
                            temp3 += pow(m2, 2)
                        else:
                            continue
                else:
                    for item in self.user_matrix[j]:
                        if self.user_matrix[i].get(item) is not None:
                            m1 = self.user_matrix[i][item] - self.user_ave[i]
                            m2 = self.user_matrix[j][item] - self.user_ave[j]
                            temp1 += m1 * m2
                            temp2 += pow(m1, 2)
                            temp3 += pow(m2, 2)
                        else:
                            continue

                if temp2 == 0 or temp3 == 0:
                    self.sim_matrix_user[i][j] = 0
                    self.sim_matrix_user[j][i] = 0
                else:
                    self.sim_matrix_user[i][j] = (temp1 / math.sqrt(temp2 * temp3))
                    self.sim_matrix_user[j][i] = (temp1 / math.sqrt(temp2 * temp3))

                count += 1
                if count % pow(10, print_per) == 0:
                    now_time = time.time()
                    print("%s Now cost %fs  %d / %d" % (time.asctime(time.localtime(now_time)), now_time - start, count, self.total))

        for i in range(len(self.user_ave)):
            self.sim_matrix_user[i] = dict(sorted(self.sim_matrix_user[i].items(), key=lambda x: x[1], reverse=True))

        now_time = time.time()
        print("%s begin save" % (time.asctime(time.localtime(now_time))))
        save_class(self.sim_matrix_user, Save_path, os.path.join(Save_path, 'sim.user'))
        self.mid = int(len(self.user_ave)/2)
        temp = []
        for i in range(self.mid):
            temp.append(self.sim_matrix_user[i])
            self.sim_matrix_user[i] = {}
        save_class(temp, Save_path, os.path.join(Save_path, self.sim1))
        temp = []
        for i in range(self.mid, len(self.user_ave)):
            temp.append(self.sim_matrix_user[i])
            self.sim_matrix_user[i] = {}
        save_class(temp, Save_path, os.path.join(Save_path, self.sim2))
        self.if_train = True
        end = time.time()
        print('%s Train time cost = %fs' % (time.asctime(time.localtime(end)), end - start))

    def predict(self, user, item_j):
        x = 0
        y = 0
        count = 0
        item_j = self.change[item_j]
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
        print("%s : start test\n" % (time.asctime(time.localtime(start))))
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
                print("%s Now cost %fs  %d " % (time.asctime(time.localtime(now_time)), now_time-start, test_count))
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
                if self.change.get(now_item) is None:
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

