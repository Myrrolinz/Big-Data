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
        self.sec_half = False
        self.now_sim = 0

    def static_analyse(self):
        # use after build
        print(f"user number: {len(self.user_ave)}")
        print(f"item number: {len(self.item_list)}")
        print(f"rated number: {self.rated_num}")
        print(f"total sim number: {self.total_sim}")
        
    def build(self, path):
        print("Building Rating Matrix...")
        user_item = file_read(path)
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
        with open("./Save/user_avg.txt", 'w') as f:
            for i, r in enumerate(self.user_ave):
                f.write(str(i) + " "+ str(r) +'\n')
        self.if_build = True

    def train(self):
        start = time.time()
        print(f"Start train at {time.asctime(time.localtime(start))}")
        self.total = int((math.pow(len(self.user_ave), 2)-len(self.user_ave))/2)
        self.sim_matrix_user = [{} for _ in range(len(self.user_ave))]
        self.now_size = 0
        self.mid = int(len(self.user_ave) / 2)

        count = 0
        for i in range(len(self.user_ave)):
            len_i = len(self.user_matrix[i])
            # print(f"now user {i}")
            # 节省计算开销
            ri2 = np.sum([math.pow(self.user_matrix[i][item] - self.user_ave[i], 2) for item in self.user_matrix[i]])
            for j in range(i+1, len(self.user_ave)):
                len_j = len(self.user_matrix[j])
                rirj = 0
                # ri2 = np.sum([math.pow(self.user_matrix[i][item] - self.user_ave[i], 2) for item in self.user_matrix[i]])
                rj2 = np.sum([math.pow(self.user_matrix[j][item] - self.user_ave[j], 2) for item in self.user_matrix[j]])
                if len_i <= len_j:
                    for item in self.user_matrix[i]:
                        if self.user_matrix[j].get(item) is not None:
                            m1 = self.user_matrix[i][item] - self.user_ave[i]
                            m2 = self.user_matrix[j][item] - self.user_ave[j]
                            rirj += m1 * m2
                else:
                    for item in self.user_matrix[j]:
                        if self.user_matrix[i].get(item) is not None:
                            m1 = self.user_matrix[i][item] - self.user_ave[i]
                            m2 = self.user_matrix[j][item] - self.user_ave[j]
                            rirj += m1 * m2

                if ri2 == 0 or rj2 == 0:
                    self.sim_matrix_user[i][j] = 0
                    self.sim_matrix_user[j][i] = 0
                else:
                    self.sim_matrix_user[i][j] = (rirj / (math.sqrt(ri2)*math.sqrt(rj2)))
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
        # 由于sim_matrix_user有4G，因此使用两个sim_matrix_user分别保存
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
        print(f"Now is: {time.asctime(time.localtime(end))}, train time cost is {end - start}.")

    def predict(self, user, item_j):
        x = 0
        y = 0
        count = 0
        item_j = self.inverted_item_list[item_j]
        if self.sec_half:
            user = user - self.mid
        for u in self.sim_matrix_user[user]:
            if self.user_matrix[u].get(item_j) is not None and self.sim_matrix_user[user][u] >= Thresh:
                count += 1
                y += self.sim_matrix_user[user][u]
                x += self.sim_matrix_user[user][u] * (self.user_matrix[u][item_j] - self.user_ave[u])
            if count == topn:
                break

        if y == 0:
            return self.user_ave[user]
        else:
            return (x / y + self.user_ave[user])

    def test(self, path):
        start = time.time()
        print(f"Start test at {time.asctime(time.localtime(start))}")
        data = file_read(path)
        user_id = None
        user_item_num = None
        temp_count = 0
        test_count = 0
        self.sec_half = False
        self.sim_matrix_user = load_class(Save_path, os.path.join(Save_path, self.sim1))
        for i in data:
            if test_count % 1000 == 0 and test_count != 0:
                now_time = time.time()
                print(f"Now is: {time.asctime(time.localtime(now_time))}, train time cost is {now_time - start}, test count: {test_count}.")
            if user_id is not None: # 增加hit概率
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
            else:
                user_id = int(i.split('|')[0], 10)
                # 由于sim_matrix_user有4G，因此使用两个sim_matrix_user分别加载
                if user_id >= self.mid and (not self.sec_half):
                    del self.sim_matrix_user[:]
                    self.sim_matrix_user = load_class(Save_path, os.path.join(Save_path, self.sim2))
                    print("load over")
                    self.sec_half = True
                while len(self.r) < user_id + 1:
                    self.r.append([])
                user_item_num = int(i.split('|')[1], 10)
                temp_count = 0
                test_count += 1
        del self.sim_matrix_user[:]
        with open("./Save/user_avg_new.txt", 'w') as f:
            for i, r in enumerate(self.user_ave):
                f.write(str(i) + " "+ str(r) +'\n')
        with open("./Save/test_rate.txt", 'w') as f:
            for i in range(len(self.r)):
                f.write(str(i) + "|6\n")
                for j in range(len(self.r[i])):
                    f.write(str(self.r[i][j][0]) + " " + str(rate_modify(self.r[i][j][1])) + "\n")
        end = time.time()
        print('%s Test time cost = %fs' % (time.asctime(time.localtime(end)), end - start))
        self.if_test = True

