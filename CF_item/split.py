#1.进行数据统计和清洗工作
#2.划分训练集和测试集
import random
from utils import *
from config import *
import numpy as np
import math
import pandas as pd

seed_value = 42  # 设置种子值为42，保证每次划分情况一样，为了可重复性 （可去）
random.seed(seed_value)

def static_analyse(self, u, i, r):
    """_summary_

    Args:
        u (int): user number
        i (int): item number
        r (int): rated number
    """
    # use after build
    print(f"user number: {u}")
    print(f"item number: {i}")
    print(f"rated number: {r}")
        
def train_test_split():
    # 按照 split_size 定义的比例划分成train与validate数据集，同时保证划分后train中包含所有item
    # 目的是后续需要计算item与item之间相似度，如果train中不存在则无法计算，影响效果
    item_user_train_data = {}
    user_item_train_data = {}
    train_train_data = []
    train_test_data = []
    train_data = file_read('./Save/train_data.pickle')
    for item, rates in train_data.items():
        for index, (user, score) in enumerate(rates.items()):
            if index == 0: # 每个item的第一个评分放入train中
                train_train_data.append([user, item, score])
                if item not in item_user_train_data:
                    item_user_train_data[item] = {}
                item_user_train_data[item][user] = score
                if user not in user_item_train_data:
                    user_item_train_data[user] = {}
                user_item_train_data[user][item] = score
                continue
            if np.random.rand() < split_size: # 如果随机数小于split_size，放入test中
                train_test_data.append([user, item, score])
            else: # 放入train
                train_train_data.append([user, item, score])
                if item not in item_user_train_data:
                    item_user_train_data[item] = {}
                item_user_train_data[item][user] = score
                if user not in user_item_train_data:
                    user_item_train_data[user] = {}
                user_item_train_data[user][item] = score
    del train_data

    train_test_data = pd.DataFrame(data=train_test_data, columns=['user', 'item', 'score'])
    train_test_data.to_csv('./Save/train_test.csv')
    train_train_data = pd.DataFrame(data=train_train_data, columns=['user', 'item', 'score'])
    train_train_data.to_csv('./Save/train_train.csv')
    file_save(item_user_train_data, "./Save/item_user_train.pickle")
    file_save(user_item_train_data, "./Save/user_item_train.pickle")
    print("train test data")
    static_analyse(len(train_test_data.index.drop_duplicates()),
                        len(train_test_data['item'].drop_duplicates()),
                        len(train_test_data))
    print("train train data")
    static_analyse(len(train_train_data.index.drop_duplicates()),
                        len(train_train_data['item'].drop_duplicates()),
                        len(train_train_data))

train_test_split()