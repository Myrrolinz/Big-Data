import os

from config import *
from utils import *
import math
import time
import numpy as np

class CF():
    def __init__(self, train_p, test_p):
        self.if_build = False
        self.if_train = False
        self.if_test = False
        self.rating_num = 0
        self.user_matrix = []  # 存储用户对物品的评分 [{itemid: score,...},...]
        # self.item_matrix = []  # 存储物品的属性 [[at1,at2],...]  now useless
        self.user_ave = []  # 用户对物品的评分准则(对物品评分的平均数)[u1,u2,...]
        # self.user_item_index = []  # 索引矩阵[[item1,item2,...],[],...]
        self.item_user_index = []  # 反向索引[{user: score, ...},...]
        self.sim_matrix = None  # item 的 相似矩阵（稀疏）lil_matrix
        self.item_list = set()
        self.change = dict()
        self.r = []  # predicted matirx
        # self.sim_csr = [[], [], []]  # item 的相似矩阵 使用csr的方式进行压缩 [[row_offset,...], [col,...],[value,...]]
        self.train_p = train_p
        self.test_p = test_p
        self.total = 0
        self.now_sim = 0

    def static_analyse(self):
        # use after build
        print("user number: %d\nrating number: %d\nitem number: %d" %(len(self.user_ave), self.rating_num, len(self.item_list)))

    def build(self, path):
        user_item = file_read(path)
        #item_attribute = file_read(Data_itematr)
        user_id = None
        user_item_num = None
        temp_count = 0
        score_count = 0
        for i in user_item:
            if user_id is None:
                score_count = 0
                user_id = int(i.split('|')[0], 10)
                while len(self.user_matrix) < user_id + 1:
                    self.user_matrix.append({})
                    self.user_ave.append(0)
                user_item_num = int(i.split('|')[1], 10)
                self.rating_num += user_item_num
                temp_count = 0
            else:
                now_item = int(i.split()[0], 10)
                while len(self.item_user_index) < now_item + 1:
                    self.item_user_index.append({})

                now_score = int(i.split()[1], 10)
                self.item_user_index[now_item][user_id] = now_score
                self.item_list.add(now_item)
                score_count += now_score
                self.user_matrix[user_id][now_item]  = now_score
                temp_count += 1
                if temp_count == user_item_num:
                    self.user_ave[user_id] = score_count / temp_count
                    user_id = None

        user_item.close()
        #print(self.user_matrix)

        # for i in item_attribute:
        #     item_id = int(i.split('|')[0], 10)
        #     attribute1 = i.split('|')[1]
        #     attribute2 = i.split('|')[2]
        #     while len(self.item_matrix) < item_id + 1:
        #         self.item_matrix.append([])
        #     self.item_matrix[item_id].append(attribute1)
        #     self.item_matrix[item_id].append(attribute2)
        #
        # item_attribute.close()
        self.item_list = list(self.item_list)
        self.item_list.sort()
        for x in range(len(self.item_list)):
            self.change[self.item_list[x]] = x

        self.if_build = True

    def get_offset(self, i, j):
        ofset = int(i*(len(self.item_list)) + j - (i+1)*(i+2)/2)
        if ofset / pow(10, save_per) == self.now_sim:
            return ofset - self.now_sim * pow(10, save_per)
        else:
            self.now_sim = ofset / pow(10, save_per)
            self.sim_matrix = load_class(Save_path, os.path.join(Save_path, 'sim'+str(self.now_sim)+'.pickle'))
            return ofset - self.now_sim * pow(10, save_per)

    def train(self):
        start = time.time()
        print("%s : start train\n" % (time.asctime(time.localtime(start))))
        self.total = int((pow(len(self.item_list), 2)-len(self.item_list))/2)
        self.sim_matrix = []
        self.now_size = 0

        count = 0
        save_count = 0
        for i in range(len(self.item_list)):
            item_i = self.item_list[i]
            len_i = len(self.item_user_index[item_i])
            for j in range(i + 1, len(self.item_list)):
                temp1 = 0
                temp2 = 0
                temp3 = 0
                item_j = self.item_list[j]
                len_j = len(self.item_user_index[item_j])
                if len_i <= len_j:
                    for user in self.item_user_index[item_i]:
                        if self.item_user_index[item_j].get(user) is not None:
                            m1 = self.item_user_index[item_i][user] - self.user_ave[user]
                            m2 = self.item_user_index[item_j][user] - self.user_ave[user]
                            temp1 += m1 * m2
                            temp2 += pow(m1, 2)
                            temp3 += pow(m2, 2)
                        else:
                            continue
                else:
                    for user in self.item_user_index[item_j]:
                        if self.item_user_index[item_i].get(user) is not None:
                            m1 = self.item_user_index[item_i][user] - self.user_ave[user]
                            m2 = self.item_user_index[item_j][user] - self.user_ave[user]
                            temp1 += m1 * m2
                            temp2 += pow(m1, 2)
                            temp3 += pow(m2, 2)
                        else:
                            continue

                if temp2 == 0 or temp3 == 0:
                    self.sim_matrix.append(0)
                else:
                    self.sim_matrix.append(temp1 / math.sqrt(temp2 * temp3))

                count += 1
                if count % pow(10, print_per) == 0:
                    now_time = time.time()
                    print("%s Now cost %fs  %d / %d" % (time.asctime(time.localtime(now_time)), now_time - start, count, self.total))
                if count % pow(10, save_per) == 0:
                    self.now_size += len(self.sim_matrix)
                    save_class(self.sim_matrix, Save_path, os.path.join(Save_path, 'sim'+str(save_count)+'.pickle'))
                    save_count += 1
                    self.now_sim = save_count
                    self.sim_matrix = []

        save_class(self.sim_matrix, Save_path, os.path.join(Save_path, 'sim' + str(save_count) + '.pickle'))
        self.if_train = True
        end = time.time()
        print('%s Train time cost = %fs' % (time.asctime(time.localtime(end)), end - start))


    def get_sim(self, i, j):
        return self.sim_matrix[self.get_offset(i, j)]

    def predict(self, user, item_j):
        x = 0
        y = 0
        item_j = self.change[item_j]
        for t in self.user_matrix[user]:
            item_i = self.change[t]
            rui = self.user_matrix[user][t]
            if item_i < item_j:
                sim = self.get_sim(item_i, item_j)
            elif item_i == item_j:
                sim = 1
            else:
                sim = self.get_sim(item_j, item_i)
            y += sim
            x += sim*rui
        if y == 0:
            return 0
        else:
            return x/y

    def test(self, path):
        data = file_read(path)
        user_id = None
        user_item_num = None
        temp_count = 0
        for i in data:
            if user_id is None:
                user_id = int(i.split('|')[0], 10)
                while len(self.r) < user_id + 1:
                    self.r.append([])
                user_item_num = int(i.split('|')[1], 10)
                temp_count = 0
            else:
                now_item = int(i.split()[0], 10)
                if self.change.get(now_item) is None:
                    p = -1
                else:
                    p = self.predict(user_id, now_item)
                self.r[user_id].append((now_item, p))
                temp_count += 1
                if temp_count == user_item_num:
                    user_id = None
        self.if_test = True

    def new_predict(self, user_id, item_list):
        p = []
        item_j_list = []
        for item in item_list:
            if self.change.get(item) is None:
                p.append(self.user_ave[user_id])
                item_j_list.append(-1)
            else:
                p.append([])
                item_j_list.append(self.change[item])

        for t in self.user_matrix[user_id]:
            item_i = self.change[t]
            rui = self.user_matrix[user_id][t]
            for j in range(len(item_j_list)):
                if not isinstance(p[j], list):
                    continue
                if item_j_list[j] != -1:
                    item_j = item_j_list[j]
                else:
                    continue

                if item_i == item_j:
                    p[j] = rui
                    continue

                if item_i < item_j:
                    ofs = int(item_i*(len(self.item_list)) + item_j - (item_i+1)*(item_i+2)/2)
                else:
                    ofs = int(item_j * (len(self.item_list)) + item_i - (item_j + 1) * (item_j + 2) / 2)







    def new_test(self, path):
        data = file_read(path)
        item_to_predict = []
        i = data.readline()
        while i:
            user_id = int(i.split('|')[0], 10)
            while len(self.r) < user_id + 1:
                self.r.append([])
            user_item_num = int(i.split('|')[1], 10)
            for x in range(user_item_num):
                i = data.readline()
                now_item = int(i.split()[0], 10)
                item_to_predict.append(now_item)
            item_to_predict.sort()
            p = []
            p = self.new_predict(user_id, item_to_predict)
            for n in range(user_item_num):
                self.r[user_id].append((item_to_predict[n], p[n]))
            i = data.readline()


        data.close()
        self.if_test = True