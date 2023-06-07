import time
import numpy as np
import pandas as pd
import pickle
from config import *
from utils import *
# from split import *
import math
from sklearn.metrics import mean_squared_error
from collections import defaultdict


def cal_RMSE(pred_rate, rate):
    return math.sqrt(mean_squared_error(pred_rate, rate))
    # return (np.sum((np.array(pred_rate) - np.array(rate)) ** 2) / len(rate)) ** 0.5

class CollaborativeFiltering:
    def __init__(self, train_path, test_path, attribute_path, is_processed=True):
        self.is_processed = is_processed
        self.similarity_map = {}
        self.attribute_similarity = {}
        self.train_path = train_path
        self.test_path = test_path
        self.attribute_path = attribute_path
        self.split_size = 0.1
        self.train_data = {}
        self.test_data = {}
        self.user_item_train_data = {}
        self.item_user_train_data = {}
        self.item_attributes = []
        self.train_train_data = []
        self.train_test_data = []
        self.bias = {}
        self.num_of_user = 0
        self.num_of_item = 0
        self.num_of_rate = 0

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
        # print(f"total sim number: {len(self.similarity_map)}")
    
    def load_train_data(self):
        # 以 {item:{user: rate}} 形式存储，保证在划分数据集的时候，训练集能包含所有item项
        user_item = file_read(self.train_path) # 读取训练集
        for i in user_item:
            i = i.strip()
            if '|' in i:
                user, rates_num = [int(j) for j in i.split('|')[:2]]
                self.num_of_rate += rates_num
                self.num_of_user += 1
            else:
                item, rate = [int(k) for k in i.split()]
                if item not in self.train_data:
                    self.train_data[item] = {}
                    self.num_of_item += 1
                self.train_data[item][user] = rate
        self.static_analyse(self.num_of_user, self.num_of_item, self.num_of_rate)
        file_save(self.train_data, "./Save/train_data.pickle")
        del user_item

    def load_test_data(self):
        # 载入 test 数据，格式为 {user:{item:score}}，便于后续预测输出
        test_data = file_read(self.test_path) # 读取测试集
        num_of_user = 0
        num_of_rate = 0
        user = 0
        items = []
        for i in test_data:
            i = i.strip()
            if '|' in i:
                user, rates = [int(j) for j in i.split('|')[:2]]
                num_of_rate += rates
                num_of_user += 1
                self.test_data[user] = []
            else:
                item = int(i)
                if item not in items:
                    items.append(item)
                self.test_data[user].append(item)
        self.static_analyse(num_of_user, len(items), num_of_rate)
        file_save(self.test_data, "./Save/test_data.pickle")
            
    def process_item_attribute(self):
        attr = file_read(self.attribute_path)
        num_of_item = 0
        for i in attr:
            i = i.strip()
            item, attr1, attr2 = i.split('|')
            item = int(item)
            if item > num_of_item:
                for i in range(num_of_item, item):
                    self.item_attributes.append([i, None, None])
            num_of_item = item
            attr1 = None if attr1 == "None" else int(attr1)
            attr2 = None if attr2 == "None" else int(attr2)
            num_of_item += 1
            self.item_attributes.append([item, attr1, attr2])
        del attr
        # 使用DataFrame数据结构，后续处理更加方便
        self.item_attributes = pd.DataFrame(data = self.item_attributes, columns=['item', 'attribute1', 'attribute2'])
        self.train_train_data.set_index('item', inplace = True)   # 将item列设置为索引
        # 使用 0 对空值进行填充，后续遇到属性全零项，属性相似度为0
        self.item_attributes["attribute1"].fillna(0, inplace = True)
        self.item_attributes["attribute2"].fillna(0, inplace = True)
        # 提前计算模长，减少训练时的计算时间
        # 使用math可以提高计算效率
        self.item_attributes["norm"] = self.item_attributes.apply(lambda x: \
            math.sqrt(math.pow(x["attribute1"], 2) + math.pow(x["attribute2"], 2)), axis=1)

        print(f"number of items: {num_of_item}")
        print("items information: \nAttribute1:")
        print(self.item_attributes["attribute1"].describe())
        print("Attribute2:")
        print(self.item_attributes["attribute2"].describe())
        # 将DataFrame转为dict，提高训练时的查询效率
        item_attributes = {}
        for item, row in self.item_attributes.iterrows():
            item_attributes[item] = [int(row['attribute1']), int(row['attribute2']), row['norm']]
        self.item_attributes = item_attributes
        file_save(self.item_attributes, "./Save/item_attributes.pickle")

    def train_test_split(self):
        # 按照 split_size 定义的比例划分成train与validate数据集，同时保证划分后train中包含所有item
        # 目的是后续需要计算item与item之间相似度，如果train中不存在则无法计算，影响效果
        for item, rates in self.train_data.items():
            for index, (user, score) in enumerate(rates.items()):
                if index == 0: # 每个item的第一个评分放入train中
                    self.train_train_data.append([user, item, score])
                    if item not in self.item_user_train_data:
                        self.item_user_train_data[item] = {}
                    self.item_user_train_data[item][user] = score
                    if user not in self.user_item_train_data:
                        self.user_item_train_data[user] = {}
                    self.user_item_train_data[user][item] = score
                    continue
                if np.random.rand() < self.split_size: # 如果随机数小于split_size，放入test中
                    self.train_test_data.append([user, item, score])
                else: # 放入train
                    self.train_train_data.append([user, item, score])
                    if item not in self.item_user_train_data:
                        self.item_user_train_data[item] = {}
                    self.item_user_train_data[item][user] = score
                    if user not in self.user_item_train_data:
                        self.user_item_train_data[user] = {}
                    self.user_item_train_data[user][item] = score
        del self.train_data

        self.train_test_data = pd.DataFrame(data=self.train_test_data, columns=['user', 'item', 'score'])
        self.train_test_data.to_csv('./Save/train_test.csv')
        self.train_train_data = pd.DataFrame(data=self.train_train_data, columns=['user', 'item', 'score'])
        self.train_train_data.to_csv('./Save/train_train.csv')
        file_save(self.item_user_train_data, "./Save/item_user_train.pickle")
        file_save(self.user_item_train_data, "./Save/user_item_train.pickle")
        print("train test data")
        self.static_analyse(len(self.train_test_data.index.drop_duplicates()),
                            len(self.train_test_data['item'].drop_duplicates()),
                            len(self.train_test_data))
        print("train train data")
        self.static_analyse(len(self.train_train_data.index.drop_duplicates()),
                            len(self.train_train_data['item'].drop_duplicates()),
                            len(self.train_train_data))

    def calculate_data_bias(self):
        # 计算全局bias信息
        overall_mean = self.train_train_data['score'].mean()
        # 使用groupby计算每个用户和每个商品的bias
        deviation_of_user = self.train_train_data['score'].groupby(
            self.train_train_data['user']).mean() - overall_mean
        deviation_of_item = self.train_train_data['score'].groupby(
            self.train_train_data.index).mean() - overall_mean
        self.bias["overall_mean"] = overall_mean
        self.bias["deviation_of_user"] = dict(deviation_of_user)
        self.bias["deviation_of_item"] = dict(deviation_of_item)
        file_save(self.bias, "./Save/bias.pickle")

    def fetch_similarity(self, item_i, item_j):
        similar_item = None
        if item_i in self.similarity_map and item_j in self.similarity_map[item_i]:
            similar_item = self.similarity_map[item_i][item_j]
        elif item_j in self.similarity_map and item_i in self.similarity_map[item_j]:
            similar_item = self.similarity_map[item_j][item_i]
        else:
            similar_item = None
        return similar_item
        
    def calc_similar_item(self, user, item_i):
        bias_i = self.bias["deviation_of_item"][item_i] + self.bias["overall_mean"]
        similar_item = {}
        # 计算物品相似度
        for item_j in self.user_item_train_data[user].keys():
            # 如果有在similarity_map中，直接取出
            similar_item[item_j] = self.fetch_similarity(item_i, item_j)
            # 如果不在similarity_map中，计算相似度
            if similar_item[item_j] is None:
                if item_j in self.bias["deviation_of_item"]:
                    bias_j = self.bias["deviation_of_item"][item_j] + self.bias["overall_mean"]
                else:
                    bias_j = self.bias["overall_mean"]
                if self.item_attributes[item_i][2] == 0 or self.item_attributes[item_j][2] == 0:
                    attribute_similarity = 0
                else:
                    # attribute 1, attribute 2 相乘后除以范式，以计算属性相似度
                    attribute_similarity = (self.item_attributes[item_i][0] * self.item_attributes[item_j][0]
                                            + self.item_attributes[item_i][1] * self.item_attributes[item_j][1]) \
                                            / (self.item_attributes[item_i][2] * self.item_attributes[item_j][2])
                norm_i = 0
                norm_j = 0
                sim_ = 0
                count = 0
                # 想办法提升性能，不然三个循环太慢了
                # norm_i = np.sum([math.pow(self.item_user_train_data[item_i][user] - bias_i, 2) for user, item in self.item_user_train_data[item_i].items()])
                norm_j = np.sum([math.pow(self.item_user_train_data[item_j][user] - bias_j, 2) for user, item in self.item_user_train_data[item_j].items()])
                if item_i in self.item_user_train_data:
                    for same_user, score in self.item_user_train_data[item_i].items():
                        norm_i += math.pow(self.item_user_train_data[item_i][same_user] - bias_i, 2)
                        if same_user not in self.item_user_train_data[item_j]:
                            continue
                        count += 1
                        sim_ += (self.item_user_train_data[item_i][same_user] - bias_i) \
                            * (self.item_user_train_data[item_j][same_user] - bias_j)
                    if count < 20:
                        sim_ = 0
                    if sim_ != 0:
                        sim_ /= math.sqrt(norm_i * norm_j)
                similarity = (sim_ + attribute_similarity) / 2
                # 对称填充item_x
                if item_i not in self.similarity_map:
                    self.similarity_map[item_i] = {}
                self.similarity_map[item_i][item_j] = similarity
                similar_item[item_j] = similarity
        return similar_item
    
    def collaborative_filtering_bias(self):
        # 协同过滤算法，利用train_test_split的数据计算RMSE
        predict_rate = []
        for index, row in enumerate(self.train_test_data.values):
            user, item_i, pred_score_x = row
            rating = 0
            similar_item = self.calc_similar_item(user, item_i)
            similar_item = sorted(similar_item.items(), key = lambda item: item[1], reverse = True) # 按相似度降序排列
            # 更新bias
            bias_i = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_i] + self.bias['deviation_of_user'][user]
            norm = 0
            for i, (item_j, similarity) in enumerate(similar_item):
                if i > topn: # 可调
                    break
                bias_j = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_j] + self.bias['deviation_of_user'][user]
                rating += similarity * (self.item_user_train_data[item_j][user] - bias_j)
                norm += similarity
            if norm == 0:
                rating = 0
            else:
                rating /= norm
            rating += bias_i
            # rating = valid_rate(rating)
            predict_rate.append(valid_rate(rating))
            # 输出与保存
            if index % 500 == 0 and index != 0:
                print("已预测", index)
                print("RMSE: ", math.sqrt(mean_squared_error(predict_rate, self.train_test_data['score'][:index + 1])))
            if index == len(self.train_test_data.values) - 1:
                file_save(self.similarity_map, "./Save/similarity_map.pickle")
        print("RMSE: ", cal_RMSE(predict_rate, self.train_test_data['score']))

    def predict(self):
        index = 0
        pred_dict = ""
        pred_dict = defaultdict(dict)
        for user, items in self.test_data.items():
            pred_dict[user] = {}
            for item_i in items:
                pred_dict[user][item_i] = 0
                rating = 0
                if item_i in self.bias["deviation_of_item"]:
                    similar_item = self.calc_similar_item(user, item_i)
                    similar_item = sorted(similar_item.items(), key=lambda item: item[1], reverse=True)
                    bias_i = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_i] + self.bias['deviation_of_user'][user]
                    norm = 0
                    for i, (item_j, similarity) in enumerate(similar_item):
                        if i > topn:
                            break
                        if item_j in self.bias["deviation_of_item"]:
                            bias_j = self.bias['overall_mean'] + self.bias['deviation_of_item'][item_j] + self.bias['deviation_of_user'][user]
                        else:
                            bias_j = self.bias["overall_mean"] + self.bias['deviation_of_user'][user]
                        rating += similarity * (self.item_user_train_data[item_j][user] - bias_j)
                        norm += similarity
                    if norm == 0:
                        rating = 0
                    else:
                        rating /= norm 
                    rating += bias_i
                else:
                    rating = self.bias['overall_mean'] + self.bias['deviation_of_user'][user]
                index += 1
                pred_dict[user][item_i] = rate_modify(rating)
                if index % 1000 == 0 and index != 0:
                    print("predicted", index)
        # file_save(pred_dict, "./Save/result_CF_bias.pickle")
        with open('../Results/result_CF_bias.txt', 'w') as f:
            for user, _ in pred_dict.items():
                f.write(str(user)+'|6\n')
                for item, score in pred_dict[user].items():
                    f.write(str(item)+' '+str(score)+'\n')
        # file_save(pred_dict, "./Save/result_CF_bias.pickle")

    def exec(self):
        if not self.is_processed:
            print('Start loading train data')
            self.load_train_data()
            print('Start dividing train data')
            self.train_test_split()
            print('load and process item attribute')
            self.process_item_attribute()
            print('Start calculating data bias')
            self.calculate_data_bias()
            print('Start loading test data')
            self.load_test_data()
        else:
            self.train_test_data = pd.read_csv('./Save/train_test.csv', index_col=0)
            self.train_train_data = pd.read_csv('./Save/train_train.csv', index_col=0)
            with open("./Save/item_user_train.pickle", 'rb') as f:
                self.item_user_train_data = pickle.load(f)
            with open("./Save/user_item_train.pickle", 'rb') as f:
                self.user_item_train_data = pickle.load(f)
            with open("./Save/item_attributes.pickle", 'rb') as f:
                self.item_attributes = pickle.load(f)
            # with open("data/similarity_map.pickle", 'rb') as f:
            #     self.similarity_map = pickle.load(f)
            with open("./Save/bias.pickle", 'rb') as f:
                self.bias = pickle.load(f)
            with open("./Save/test_data.pickle", 'rb') as f:
                self.test_data = pickle.load(f)

        # train collaborative filtering model
        # print('Start collaborative filtering model')
        # start = time.time()
        # self.collaborative_filtering_bias()
        # end = time.time()
        # print('Running time: %s Seconds' % (end - start))

        # load pre-trained model
        print('Start predict')
        start = time.time()
        self.predict()
        end = time.time()
        print('Running time: %s Seconds' % (end - start))


if __name__ == '__main__':
    test = CollaborativeFiltering(
        test_path = '../Data/test.txt',
        train_path = '../Data/train.txt',
        attribute_path='../Data/itemAttribute.txt',
        is_processed = False
    )
    test.exec()
    os.system('pause')
