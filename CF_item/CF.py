# -*- coding:utf-8 -*-
# @Time： 6/4/21 3:12 PM
# @Author: dyf-2316
# @FileName: RecommendationSystem.py
# @Software: PyCharm
# @Project: RecommendationSystem
# @Description:

"""
林铸天 丁一凡 洪一帆 小组
关于使用CF算法进行推荐
"""

import time
import numpy as np
import pandas as pd
import pickle

np.random.seed(1111)


def cal_RMSE(pred_rate, rate):
    return (np.sum((np.array(pred_rate) - np.array(rate)) ** 2) / len(rate)) ** 0.5
    # return (np.linalg.norm(np.array(pred_rate) - np.array(rate), ord=2) / len(rate)) ** 0.5


class RecommendationSystem:

    def __init__(self, train_path, test_path, attribute_path, is_processed=True):
        self.is_processed = is_processed
        self.similarity_map = {}
        self.attribute_similarity = {}
        self.train_path = train_path
        self.test_path = test_path
        self.attribute_path = attribute_path
        self.divide_size = 0.1
        self.train_data = {}
        self.test_data = {}
        self.user_item_train_data = {}
        self.item_user_train_data = {}

        self.item_attributes = []

        self.train_train_data = []
        self.train_test_data = []

        self.bias = {}

    def load_train_data(self):
        # 以 {item:{user: score}} 形式存储，保证在划分数据集的时候，训练集能包含所有item项
        with open(self.train_path, 'r', encoding='utf-8') as f:
            num_of_user = 0
            num_of_item = 0
            num_of_rate = 0
            user = 0
            line = f.readline()
            while line:
                line = line.strip()
                if '|' in line:
                    user, rates = line.split('|')
                    user = int(user)
                    rates = int(rates)
                    num_of_rate += int(rates)
                    num_of_user += 1
                else:
                    item, score = line.split()
                    item = int(item)
                    score = int(score)
                    if item not in self.train_data:
                        self.train_data[item] = {}
                        num_of_item += 1
                    self.train_data[item][user] = score
                line = f.readline()
        print("用户数量：", num_of_user)
        print("商品数量：", num_of_item)
        print("评分数量：", num_of_rate)

    def load_and_process_item_attribute(self):
        with open(self.attribute_path, 'r', encoding='utf-8') as f:
            num_of_item = 0
            line = f.readline()
            while line:
                line = line.strip()
                item, attr1, attr2 = line.split('|')
                item = int(item)
                if item > num_of_item:
                    for i in range(num_of_item, item):
                        self.item_attributes.append([i, None, None])
                num_of_item = item
                attr1 = None if attr1 == "None" else int(attr1)
                attr2 = None if attr2 == "None" else int(attr2)
                num_of_item += 1
                self.item_attributes.append([item, attr1, attr2])
                line = f.readline()

        # 使用DataFrame方便对空值填充
        self.item_attributes = pd.DataFrame(data=self.item_attributes, columns=['item', 'attribute1', 'attribute2'])
        self.train_train_data.set_index('item', inplace=True)
        # 使用 0 对空值进行填充，后续遇到属性全零项，属性相似度为0
        self.item_attributes["attribute1"].fillna(0, inplace=True)
        self.item_attributes["attribute2"].fillna(0, inplace=True)
        # 提前计算模长，减少训练时的计算时间
        self.item_attributes["norm"] = (self.item_attributes["attribute1"] ** 2 + self.item_attributes[
            "attribute2"] ** 2) ** (1 / 2)

        print("商品数量：", num_of_item)
        print("商品属性统计信息：")
        print(self.item_attributes["attribute1"].describe())
        print(self.item_attributes["attribute2"].describe())

        # 将DataFrame转为dict，提高训练时的查询效率
        item_attributes = {}
        for item, row in self.item_attributes.iterrows():
            item_attributes[item] = [int(row['attribute1']), int(row['attribute2']), row['norm']]
        self.item_attributes = item_attributes
        with open('data/item_attributes.pickle', 'wb') as handle:
            pickle.dump(self.item_attributes, handle)

    def divide_train_data(self):
        # 按照 divide_size 定义的比例划分成train与validate数据集，同时保证划分后train中包含所有item
        # 目的是后续需要计算item与item之间相似度，如果train中不存在则无法计算，影响效果
        for item, rates in self.train_data.items():
            for index, (user, score) in enumerate(rates.items()):
                if index == 0:
                    self.train_train_data.append([user, item, score])
                    if item not in self.item_user_train_data:
                        self.item_user_train_data[item] = {}
                    self.item_user_train_data[item][user] = score
                    if user not in self.user_item_train_data:
                        self.user_item_train_data[user] = {}
                    self.user_item_train_data[user][item] = score
                    continue
                if np.random.rand() < self.divide_size:
                    self.train_test_data.append([user, item, score])
                else:
                    self.train_train_data.append([user, item, score])
                    if item not in self.item_user_train_data:
                        self.item_user_train_data[item] = {}
                    self.item_user_train_data[item][user] = score
                    if user not in self.user_item_train_data:
                        self.user_item_train_data[user] = {}
                    self.user_item_train_data[user][item] = score

        del self.train_data

        self.train_test_data = pd.DataFrame(data=self.train_test_data, columns=['user', 'item', 'score'])
        # self.train_test_data.set_index('user', inplace=True)
        self.train_test_data.to_csv('data/train_test.csv')
        self.train_train_data = pd.DataFrame(data=self.train_train_data, columns=['user', 'item', 'score'])
        # self.train_train_data.set_index('user', inplace=True)
        self.train_train_data.to_csv('data/train_train.csv')

        with open("data/item_user_train.pickle", 'wb') as f:
            pickle.dump(self.item_user_train_data, f)
        with open("data/user_item_train.pickle", 'wb') as f:
            pickle.dump(self.user_item_train_data, f)

        print("train test data")
        print("用户数量：", len(self.train_test_data.index.drop_duplicates()))
        print("商品数量：", len(self.train_test_data['item'].drop_duplicates()))
        print("评分数量：", len(self.train_test_data))
        print("train train data")
        print("用户数量：", len(self.train_train_data.index.drop_duplicates()))
        print("商品数量：", len(self.train_train_data['item'].drop_duplicates()))
        print("评分数量：", len(self.train_train_data))

    def load_test_data(self):
        # 载入 test 数据，格式为 {user:{item:score}}，便于后续预测输出
        with open(self.test_path, 'r', encoding='utf-8') as f:
            num_of_user = 0
            items = []
            num_of_rate = 0
            user = 0
            line = f.readline()
            while line:
                line = line.strip()
                if '|' in line:
                    user, rates = line.split('|')
                    user = int(user)
                    rates = int(rates)
                    num_of_rate += int(rates)
                    num_of_user += 1
                    self.test_data[user] = []
                else:
                    item = int(line)
                    if item not in items:
                        items.append(item)
                    self.test_data[user].append(item)
                line = f.readline()
        print("用户数量：", num_of_user)
        print("商品数量：", len(items))
        print("评分数量：", num_of_rate)
        with open("data/test_data.pickle", 'wb') as handle:
            pickle.dump(self.test_data, handle)

    def calculate_data_bias(self):
        # 计算全局bias信息
        overall_mean = self.train_train_data['score'].mean()
        deviation_of_user = self.train_train_data['score'].groupby(
            self.train_train_data['user']).mean() - overall_mean
        deviation_of_item = self.train_train_data['score'].groupby(
            self.train_train_data.index).mean() - overall_mean
        self.bias["overall_mean"] = overall_mean
        self.bias["deviation_of_user"] = dict(deviation_of_user)
        self.bias["deviation_of_item"] = dict(deviation_of_item)
        with open("data/bias.pickle", 'wb') as handle:
            pickle.dump(self.bias, handle)

    def global_model(self):
        # 以总体评分均值预测
        print("Global average")
        predict_rate = np.full(len(self.train_test_data), self.bias["overall_mean"])
        global_average_RMSE = cal_RMSE(predict_rate, self.train_test_data['score'])
        print("RMSE: ", global_average_RMSE)

        # 以 user 评分均值预测
        print("User average")
        predict_rate = []
        for row in self.train_test_data.values:
            user, item, score = row
            predict_rate.append(self.bias["overall_mean"] + self.bias["deviation_of_user"][user])
        predict_rate = np.array(predict_rate)
        user_average_RMSE = cal_RMSE(predict_rate, self.train_test_data['score'])
        print("RMSE: ", user_average_RMSE)

        # 以 item 评分均值
        print("Item average")
        predict_rate = []
        for row in self.train_test_data.values:
            user, item, score = row
            if item not in self.bias["deviation_of_item"]:
                predict_rate.append(self.bias["overall_mean"])
            else:
                predict_rate.append(self.bias["overall_mean"] + self.bias["deviation_of_item"][item])
        predict_rate = np.array(predict_rate)
        item_average_RMSE = cal_RMSE(predict_rate, self.train_test_data['score'])
        print("RMSE: ", item_average_RMSE)

        # 以利用所有全局统计信息修正预测
        print("Global effects")
        predict_rate = []
        for row in self.train_test_data.values:
            user, item, score = row
            if item not in self.bias["deviation_of_item"]:
                predict_rate.append(self.bias["overall_mean"] + self.bias["deviation_of_user"][user])
            else:
                predict_rate.append(
                    self.bias["overall_mean"] + self.bias["deviation_of_user"][user] + self.bias["deviation_of_item"][
                        item])
        predict_rate = np.array(predict_rate)
        global_effects_RMSE = cal_RMSE(predict_rate, self.train_test_data['score'])
        print("RMSE: ", global_effects_RMSE)

    def collaborative_filtering_bias(self):
        # 协同过滤算法
        predict_rate = []
        for index, row in enumerate(self.train_test_data.values):
            user, item_x, pred_score_x = row
            score_x = 0
            b_x = self.bias["deviation_of_item"][item_x] + self.bias["overall_mean"]
            similar_item = {}
            # 计算物品相似度
            for item_y in self.user_item_train_data[user].keys():
                if item_x in self.similarity_map and item_y in self.similarity_map[item_x]:
                    similar_item[item_y] = self.similarity_map[item_x][item_y]
                elif item_y in self.similarity_map and item_x in self.similarity_map[item_y]:
                    similar_item[item_y] = self.similarity_map[item_y][item_x]
                else:
                    b_y = self.bias["deviation_of_item"][item_y] + self.bias["overall_mean"]
                    if self.item_attributes[item_x][2] == 0 or self.item_attributes[item_y][2] == 0:
                        attribute_similarity = 0
                    else:
                        attribute_similarity = (self.item_attributes[item_x][0] * self.item_attributes[item_y][0]
                                                + self.item_attributes[item_x][1] * self.item_attributes[item_y][1]) \
                                               / (self.item_attributes[item_x][2] * self.item_attributes[item_y][2])
                    norm_x = 0
                    norm_y = 0
                    pearson_similarity = 0
                    count = 0
                    for same_user, score in self.item_user_train_data[item_x].items():
                        if same_user not in self.item_user_train_data[item_y]:
                            continue
                        count += 1
                        pearson_similarity += (self.item_user_train_data[item_x][same_user] - b_x) * (
                                self.item_user_train_data[item_y][same_user] - b_y)
                        norm_x += (self.item_user_train_data[item_x][same_user] - b_x) ** 2
                        norm_y += (self.item_user_train_data[item_y][same_user] - b_y) ** 2
                    if count < 20:
                        pearson_similarity = 0
                    if pearson_similarity != 0:
                        pearson_similarity /= (norm_x * norm_y) ** (1 / 2)

                    similarity = (pearson_similarity + attribute_similarity) / 2

                    if item_x not in self.similarity_map:
                        self.similarity_map[item_x] = {}
                    self.similarity_map[item_x][item_y] = similarity

                    similar_item[item_y] = similarity

            similar_item = sorted(similar_item.items(), key=lambda item: item[1], reverse=True)
            b_x = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_x] + self.bias['deviation_of_user'][
                user]
            norm = 0
            for i, (item_y, similarity) in enumerate(similar_item):
                if i > 100:
                    break
                b_y = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_y] + \
                      self.bias['deviation_of_user'][user]
                score_x += similarity * (self.item_user_train_data[item_y][user] - b_y)
                norm += similarity
            if norm == 0:
                score_x = 0
            else:
                score_x /= norm
            score_x += b_x
            score_x = score_x if score_x > 0 else 0
            score_x = score_x if score_x < 100 else 100
            predict_rate.append(score_x)
            if index % 500 == 0 and index != 0:
                print("已预测", index)
            if index % 5000 == 0 and index != 0:
                print("RMSE: ", cal_RMSE(predict_rate, self.train_test_data['score'][:index + 1]))
            if index % 200000 == 0 and index != 0:
                with open("data/similarity_map.pickle", 'wb') as f:
                    pickle.dump(self.similarity_map, f)
        print("RMSE: ", cal_RMSE(predict_rate, self.train_test_data['score']))

    def predict(self):
        index = 0
        buffer = ""
        with open('data/result_CF.txt', 'w') as f:
            for user, items in self.test_data.items():
                buffer += str(user) + "|" + str(len(items)) + '\n'
                for item_x in items:
                    buffer += str(item_x) + " "
                    score_x = 0
                    if item_x in self.bias["deviation_of_item"]:
                        b_x = self.bias["deviation_of_item"][item_x] + self.bias["overall_mean"]
                        similar_item = {}
                        for item_y in self.user_item_train_data[user].keys():
                            if item_x in self.similarity_map and item_y in self.similarity_map[item_x]:
                                similar_item[item_y] = self.similarity_map[item_x][item_y]
                            elif item_y in self.similarity_map and item_x in self.similarity_map[item_y]:
                                similar_item[item_y] = self.similarity_map[item_y][item_x]
                            else:
                                if item_y in self.bias["deviation_of_item"]:
                                    b_y = self.bias["deviation_of_item"][item_y] + self.bias["overall_mean"]
                                else:
                                    b_y = self.bias["overall_mean"]
                                if self.item_attributes[item_x][2] == 0 or self.item_attributes[item_y][2] == 0:
                                    attribute_similarity = 0
                                else:
                                    attribute_similarity = (self.item_attributes[item_x][0] *
                                                            self.item_attributes[item_y][
                                                                0]
                                                            + self.item_attributes[item_x][1] *
                                                            self.item_attributes[item_y][1]) \
                                                           / (self.item_attributes[item_x][2] *
                                                              self.item_attributes[item_y][2])
                                norm_x = 0
                                norm_y = 0
                                pearson_similarity = 0
                                count = 0
                                if item_x in self.item_user_train_data:
                                    for same_user, score in self.item_user_train_data[item_x].items():
                                        if same_user not in self.item_user_train_data[item_y]:
                                            continue
                                        count += 1
                                        pearson_similarity += (self.item_user_train_data[item_x][same_user] - b_x) * (
                                                self.item_user_train_data[item_y][same_user] - b_y)
                                        norm_x += (self.item_user_train_data[item_x][same_user] - b_x) ** 2
                                        norm_y += (self.item_user_train_data[item_y][same_user] - b_y) ** 2
                                    if count < 20:
                                        pearson_similarity = 0
                                    if pearson_similarity != 0:
                                        pearson_similarity /= (norm_x * norm_y) ** (1 / 2)

                                similarity = (pearson_similarity + attribute_similarity) / 2

                                if item_x not in self.similarity_map:
                                    self.similarity_map[item_x] = {}
                                self.similarity_map[item_x][item_y] = similarity

                        similar_item = sorted(similar_item.items(), key=lambda item: item[1], reverse=True)
                        b_x = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_x] + \
                              self.bias['deviation_of_user'][user]
                        norm = 0

                        for i, (item_y, similarity) in enumerate(similar_item):
                            if i > 100:
                                break
                            if item_y in self.bias["deviation_of_item"]:
                                b_y = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_y] + \
                                      self.bias['deviation_of_user'][user]
                            else:
                                b_y = self.bias["overall_mean"] + self.bias['deviation_of_user'][user]
                            score_x += similarity * (self.item_user_train_data[item_y][user] - b_y)
                            norm += similarity

                        if norm == 0:
                            score_x = 0
                        else:
                            score_x /= norm

                        score_x += b_x
                    else:
                        score_x = self.bias['overall_mean'] + self.bias['deviation_of_user'][user]
                    index += 1
                    score_x = score_x if score_x > 0 else 0
                    score_x = score_x if score_x < 100 else 100
                    buffer += str(int(score_x)) + '\n'
                    if index % 1000 == 0 and index != 0:
                        print("已预测", index)
                        f.write(buffer)
                        buffer = ""
        with open('data/result_CF.txt', 'a') as f:
            f.write(buffer)

    def exec(self):
        if not self.is_processed:
            print('Start load train data')
            self.load_train_data()

            print('Start divide train data')
            self.divide_train_data()

            print('load and process item attribute')
            self.load_and_process_item_attribute()

            print('Start calculate data bias')
            self.calculate_data_bias()

            print('Start load test data')
            self.load_test_data()
        else:
            self.train_test_data = pd.read_csv('data/train_test.csv', index_col=0)
            self.train_train_data = pd.read_csv('data/train_train.csv', index_col=0)
            with open("data/item_user_train.pickle", 'rb') as f:
                self.item_user_train_data = pickle.load(f)
            with open("data/user_item_train.pickle", 'rb') as f:
                self.user_item_train_data = pickle.load(f)
            with open("data/item_attributes.pickle", 'rb') as f:
                self.item_attributes = pickle.load(f)
            with open("data/similarity_map.pickle", 'rb') as f:
                self.similarity_map = pickle.load(f)
            with open("data/bias.pickle", 'rb') as f:
                self.bias = pickle.load(f)
            with open("data/test_data.pickle", 'rb') as f:
                self.test_data = pickle.load(f)

        # 训练全局平均模型
        # print('Start global model')
        # start = time.time()
        # self.global_model()
        # end = time.time()
        # print('Running time: %s Seconds' % (end - start))

        # 训练协同过滤模型
        # print('Start collaborative filtering model')
        # start = time.time()
        # self.collaborative_filtering_bias()
        # end = time.time()
        # print('Running time: %s Seconds' % (end - start))

        # 加载预训练参数，直接预测
        print('Start predict')
        start = time.time()
        self.predict()
        end = time.time()
        print('Running time: %s Seconds' % (end - start))


if __name__ == '__main__':
    test = RecommendationSystem(
        test_path='data/test.txt',
        train_path='data/train.txt',
        attribute_path='data/itemAttribute.txt',
        # 是否需要处理初始数据。我们的项目里已经打包了处理好的数据
        # 因此您只需在这里保留true即可
        is_processed=False
    )
    test.exec()
