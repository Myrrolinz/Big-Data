import os
import numpy as np
import math
from config import *
from tqdm import tqdm
import torch
from math import ceil


class LFM(object):
    def __init__(self, factor=5, iter_num=10, alpha=0.002, Lambda=0.04, epsilon=1e-2, test_flag=False, keep_last=2):
        # self.data_folder = data_set_folder
        self.factor = factor  # 分解后的因子个数
        self.iter_num = iter_num  # 整体迭代次数
        self.alpha = alpha
        self.Lambda = Lambda
        self.epsilon = epsilon  # 最终误差要小于epsilon
        self.test_flag = test_flag  # 如果是true就F用小的数据集，如果是false，就用大的数据集
        self.keep_last = keep_last  # 要保存的最新的checkpoint数量

        # 保存数据的稀疏矩阵
        self.train_data = dict()

        # train dataset中的item和user的数量
        self.num_of_items = 0
        self.num_of_users = 0

        # 分解的矩阵
        self.P = None
        self.Q = None

        # 记录读取到的epoch
        self.epoch_of_ckpt = -1

    def load_data(self):
        '''
        加载训练的数据集
        '''
        print("Load data...")

        if self.test_flag:
            curr_data_set_file = test_iter_file
        else:
            curr_data_set_file = train_data_file

        if not os.path.exists(curr_data_set_file):
            print(f"Error: No such {curr_data_set_file}")
            exit(1)

        # 记录所有得分的平均分，用来表示没有打分的
        ave = 0

        with open(curr_data_set_file, 'r', encoding='utf8') as f:
            for line in f:
                # 对于文件中的每一行
                if len(line.split("|")) == 2:
                    # 表示此时这一行是userID|num
                    curr_user = eval(line.split("|")[0])
                    curr_num = eval(line.split("|")[1])
                    self.train_data[curr_user] = dict()
                else:
                    # 表示此时这一行是itemID score
                    # 判断当前这一行对不对
                    if len(self.train_data[curr_user]) >= curr_num:
                        print("Data Error: Form of current line is 'itemID score', but is should be 'userID|num'")
                        exit(1)

                    itemID = eval(line.split()[0])
                    score = eval(line.split()[1]) / 100
                    self.num_of_items = max(self.num_of_items, itemID)
                    # self.train_data[curr_user].append((itemID, score))
                    self.train_data[curr_user][itemID] = score

        self.num_of_users = len(self.train_data.keys())
        self.num_of_items += 1

    def init_latent_matrix(self, init=False):
        '''
        初始化分解矩阵
        '''
        if not os.path.exists(CKPT_FILE_FOLDER):
            os.makedirs(CKPT_FILE_FOLDER)

        if init:
            print("Initialise latent matrix...")
            self.P = np.random.rand(self.num_of_users, self.factor)
            self.Q = np.random.rand(self.factor, self.num_of_items)
            self.epoch_of_ckpt = -1
        else:
            if len(os.listdir(CKPT_FILE_FOLDER)) > 0:
                ckpt_files = os.listdir(CKPT_FILE_FOLDER)
                ckpt_file_number = []
                for f in ckpt_files:
                    ckpt_file_number.append(eval(f.split("_")[0]))
                ckpt_file_number.sort(reverse=True)
                last_number = str(ckpt_file_number[0])

                P_ckpt_file = CKPT_FILE_FOLDER + last_number + "_P.ckpt"
                Q_ckpt_file = CKPT_FILE_FOLDER + last_number + "_Q.ckpt"
                self.epoch_of_ckpt = eval(last_number)

                print(f"Loading last checkpoint: Epoch {self.epoch_of_ckpt}")
                if not os.path.exists(P_ckpt_file) or not os.path.exists(Q_ckpt_file):
                    print(f"Something wrong with ckpt file: No {P_ckpt_file} or {Q_ckpt_file}")
                    exit(1)
                self.P = torch.load(P_ckpt_file)
                self.Q = torch.load(Q_ckpt_file)
            else:
                print("No checkpoint files. Initialise latent matrix...")
                self.P = np.random.rand(self.num_of_users, self.factor)
                self.Q = np.random.rand(self.factor, self.num_of_items)
                self.epoch_of_ckpt = -1

        print(f"Shape: P {self.P.shape} \t Q {self.Q.shape}")

    def read_epoch(self, epoch):
        P_ckpt_file = CKPT_FILE_FOLDER + str(epoch) + "_P.ckpt"
        Q_ckpt_file = CKPT_FILE_FOLDER + str(epoch) + "_Q.ckpt"

        print(f"Loading checkpoint: Epoch {epoch}")
        if not os.path.exists(P_ckpt_file) or not os.path.exists(Q_ckpt_file):
            print(f"Something wrong with ckpt file: No {P_ckpt_file} or {Q_ckpt_file}")
            exit(1)
        self.P = torch.load(P_ckpt_file)
        self.Q = torch.load(Q_ckpt_file)

        print(f"Shape: P {self.P.shape} \t Q {self.Q.shape}")

    def calc_f_norm(self, matrix):
        '''
        计算矩阵的F范数，即所有元素的平方和再开方
        用于计算惩罚项
        '''
        [x, y] = matrix.shape
        f_norm = 0
        for i in range(x):
            for j in range(y):
                f_norm += matrix[i][j] ** 2
        f_norm = math.sqrt(f_norm)

        return f_norm

    def get_test_result(self, test_flag=False):
        '''
        得到本次test.txt中的答案
        '''
        # 获得test.txt中所有的内容
        test = dict()
        print("Reading test.txt...")
        with open(test_data_file) as f:
            for line in f:
                if len(line.split("|")) == 2:
                    # 表示此时这一行是userID|num
                    curr_user = eval(line.split("|")[0])
                    curr_num = eval(line.split("|")[1])
                    test[curr_user] = dict()
                else:
                    # 表示此时这一行是itemID
                    # 判断当前这一行对不对
                    if len(test[curr_user]) >= curr_num:
                        print("Data Error: Form of current line is 'itemID score', but is should be 'userID|num'")
                        exit(1)

                    itemID = eval(line)
                    test[curr_user][itemID] = -1  # 初始化为-1
        print("Computing result of test.txt...")

        if test_flag:
            self.save_result(test_result_file, test)
        else:
            self.save_result(result_file, test)

    def save_result(self, file_name, data):
        print("Saving result file...")
        with open(file_name, "w") as f:
            for user in data.keys():
                f.write(str(user) + "|" + str(len(data[user].keys())) + "\n")
                for item in data[user].keys():
                    f.write(str(item) + " " + str(self.get_pred_result(user, item)) + "\n")

        print(f"Test result file in save as {file_name}.")

    def train(self):
        '''
        开始迭代训练，计算合适的分解矩阵
        '''
        print("Start training...")
        for curr_iter in range(self.epoch_of_ckpt + 1, self.iter_num + self.epoch_of_ckpt + 1):
            print(f"Epoch: {curr_iter}")
            # 在每一次迭代中
            for i in tqdm(range(self.num_of_users), desc='User', ncols=100, leave=False):
                # 对每一个用户
                for j in tqdm(range(self.num_of_items), desc='Item', ncols=100, leave=False):
                    if i in self.train_data.keys() and j in self.train_data[i].keys():
                        eui = np.dot(self.P[i, :], self.Q[:, j]) - self.train_data[i][j]
                        # print(eui)

                        # 梯度下降
                        for k in range(self.factor):
                            self.P[i][k] = self.P[i][k] - self.alpha * (
                                        2 * eui * self.Q[k][j] + 2 * self.Lambda * self.P[i][k])
                            self.Q[k][j] = self.Q[k][j] - self.alpha * (
                                        2 * eui * self.P[i][k] + 2 * self.Lambda * self.Q[k][j])

            self.save_PQ(curr_iter, self.keep_last)

            print("Computing Loss...")
            # 此时用户和电影都已经遍历完毕，计算loss
            loss = 0
            for i in tqdm(range(self.num_of_users), desc='User', ncols=100, leave=False):
                # 对每一个用户
                for j in tqdm(range(self.num_of_items), desc='Item', ncols=100, leave=False):
                    # 对于有评分的，计算评分误差
                    if i in self.train_data.keys() and j in self.train_data[i].keys():
                        loss += (np.dot(self.P[i, :], self.Q[:, j]) - self.train_data[i][j]) ** 2
                        # 加上惩罚项
                        # loss += self.calc_f_norm(self.P[i:]) + self.calc_f_norm(self.Q[:j])
                        for k in range(self.factor):
                            loss += self.Lambda * (self.P[i][k] ** 2 + self.Q[k][j] ** 2)

            print(f"Epoch:{curr_iter}, loss = {loss}")

            # 如果此时已经小于预设的值，就可以提前结束迭代了
            if loss < self.epsilon:
                break

    def save_PQ(self, epoch, keep_last: int):
        if not os.path.exists(CKPT_FILE_FOLDER):
            os.mkdir(CKPT_FILE_FOLDER)

        print(f"Saving checkpoints: Epoch {epoch}")
        torch.save(self.P, CKPT_FILE_FOLDER + f'{epoch}_P.ckpt')
        torch.save(self.Q, CKPT_FILE_FOLDER + f'{epoch}_Q.ckpt')

        ckpt_files = os.listdir(CKPT_FILE_FOLDER)
        ckpt_file_number = []
        for f in ckpt_files:
            ckpt_file_number.append(eval(f.split("_")[0]))
        ckpt_file_number.sort(reverse=True)
        if len(ckpt_files) > 2 * keep_last:
            last_number = str(ckpt_file_number[0])
            second_number = str(ckpt_file_number[2])

            preserve_ckpt_file_list = []
            preserve_ckpt_file_list.append(last_number + "_P.ckpt")
            preserve_ckpt_file_list.append(last_number + "_Q.ckpt")
            preserve_ckpt_file_list.append(second_number + "_P.ckpt")
            preserve_ckpt_file_list.append(second_number + "_Q.ckpt")

            for f in ckpt_files:
                if f not in preserve_ckpt_file_list:
                    os.remove(CKPT_FILE_FOLDER + f)

    def get_pred_result(self, i=-1, j=-1):
        if i == -1 and j == -1:
            return np.dot(self.P, self.Q)
        elif i == -1 or j == -1:
            print("矩阵xy角标错误")
            exit(2)
        else:
            result = np.dot(self.P[i, :], self.Q[:, j]) * 100

            if result >= 100:
                return 100
            elif result < 0:
                return 0
            else:
                if result % 10 >= 5:
                    result = (int(result / 10) + 1) * 10
                else:
                    result = int(result / 10) * 10
                return int(result)

    def calc_train(self, epoch):
        # self.load_data()
        self.read_epoch(epoch)

        for user in self.train_data.keys():
            for item in self.train_data[user].keys():
                # if item == self.num_of_items - 1:
                #     continue
                # else:
                self.train_data[user][item] = self.get_pred_result(user, item)

        file_name = f"./train_result_{epoch}.txt"
        self.save_result(file_name, self.train_data)
