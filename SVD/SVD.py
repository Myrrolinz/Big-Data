import struct
import math
import time
import random
from collections import defaultdict
from scipy import optimize


import numpy as np

from configAndUtils import *

class SVDModel(object):
    def __init__(self):
        #超参数设置
        self.factors = FACTORS  # 分解后的隐含因子个数
        self.epochs = EPOCHS  # 整体迭代次数
        self.learning_rate = LR
        self.LambdaUB = LAMBDAUB
        self.LambdaIB = LAMBDAIB
        self.LambdaP = LAMBDAP
        self.LambdaQ = LAMBDAQ

        self.user_num=0 #用户总人数（实际人数<=usr_id）
        self.item_num=0 #同理
        self.rating_scale = [0, 100]  # 用户打分范围为0：100

        #从统计好的文件中读取数据
        self.user_dict = defaultdict()
        self.item_dict = defaultdict()
        self.item_attr = defaultdict()  # 储存商品属性 {商品实际id: (商品实际id, 属性1, 属性2)......}
        with open(user_dictFile,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()  # 去除行末尾的换行符和空格
                key, value = line.split(':')  # 按冒号进行分割
                key = key.strip()  # 去除键中的空格
                value = value.strip()  # 去除值中的空格
                self.user_dict[int(key)]=int(value)
                # print(key, value)  # 打印每行的键和值
            f.close()

        with open(item_dictFile,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()  # 去除行末尾的换行符和空格
                key, value = line.split(':')  # 按冒号进行分割
                key = key.strip()  # 去除键中的空格
                value = value.strip()  # 去除值中的空格
                self.item_dict[int(key)]=int(value)
                # print(key, value)  # 打印每行的键和值
            f.close()

        with open(item_attrFile,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()  # 去除行末尾的换行符和空格
                values = line.split(',')  # 按逗号进行分割
                key = int(values[0].strip())  # 去除键中的空格并转换为整数类型
                value = tuple(int(x.strip()) for x in values[1:])  # 去除值中的空格，并将值转换为整数类型
                self.item_attr[key] = value
            f.close()

        with open(item_numFile,'r') as f:
            line = f.readline()
            self.item_num=int(line)
            f.close()

        with open(user_numFile,'r') as f:
            line = f.readline()
            self.user_num=int(line)
            f.close()

        '''
        初始化P/Q矩阵、初始化偏置项
        :return:
        '''
        # Calculate the square root of the number of factors
        sqrt_factors = math.sqrt(self.factors)
        #随机初始化P Q矩阵，随机数与隐因子平方成反比
        #P矩阵的大小是user_num × factors；Q大小item_num × factors
        self.P = [[random.random() / sqrt_factors for i in range(0, self.factors)] for i in range(0, self.user_num)]
        self.Q = [[random.random() / sqrt_factors for i in range(0, self.factors)] for i in range(0, self.item_num)]

        self.user_bias = [0.0 for i in range(0, self.user_num)]
        self.item_bias = [0.0 for i in range(0, self.item_num)]


    def loadTrainSet(self):
        # 读取训练集
        with open(training_set, 'r') as f:
            trainSet = f.readlines()
            f.close()

        self.lil_matrix=[] #稀疏矩阵 用于存数据
        self.overall_train_mean_rating=0.0
        overall_rating_sum=0.0
        self.num_trainingData = 0
        for train_line in trainSet:
            line = train_line.strip()
            if line == "":
                continue
            user_id, item_id, rate_str = line.split(',')#CSV文件以逗号分割
            uid = self.user_dict[int(user_id)]  # Find the user's Id in this program based on the actual user id
            iid = self.item_dict[int(item_id)]  # Find the item's Id in this program based on the actual item id
            user_rate = float(rate_str)
            self.lil_matrix.append((uid, iid, user_rate))  # Store in sprase_matrix as (program's user id, program's item id, user rating)
            overall_rating_sum += user_rate
            self.num_trainingData += 1

        self.overall_train_mean_rating =overall_rating_sum/ self.num_trainingData


    def dot(self,u,i):
        sum = 0.0
        for k in range(0, self.factors):
            sum += (self.P[u][k] * self.Q[i][k])
        return sum

    # 对rp进行边界检查（须在0到100之间）
    def correctRank(self,rp):
        if rp < 0:
            rp = float(0)
        if rp > 100:
            rp = float(100)
        return rp


    def train(self):
        print("================ 开始训练 ================")
        print("training_set:",training_set)
        print("test_set:",test_set)
        print("LAMBDAUB:",LAMBDAUB)
        print("LAMBDAIB:",LAMBDAIB)
        print("LAMBDAP:",LAMBDAP)
        print("LAMBDAQ:",LAMBDAQ)
        print("FACTORS:",FACTORS)
        print("learning_rate:",LR)
        print("decay:",decay_factor)


        # 迭代训练epochs次
        start=time.time()
        for epoch in range(0, self.epochs):
            rmse = 0.0
            self.lil_matrix_diff = [] #打分预测值与真实值之间的差距！

            for u, i, r in self.lil_matrix:
                # 计算预测分值
                rp = self.overall_train_mean_rating + self.user_bias[u] + self.item_bias[i] + self.dot(u, i)
                rp = self.correctRank(rp)

                #计算真实值和预测值之间的差值
                diff = r - rp
                # 修正偏置参数：使用四个Lambda值LambdaUB、LambdaIB、LambdaP、LambdaQ
                self.user_bias[u] += self.learning_rate * (diff - self.LambdaUB * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (diff - self.LambdaIB * self.item_bias[i])
                for k in range(0, self.factors):
                    # 每次迭代计算矩阵之前需要保存原值，否则越迭代值越大，最终导致超过浮点数范围
                    q_i_k = self.Q[i][k]
                    p_u_k = self.P[u][k]

                    self.P[u][k] += self.learning_rate * (diff * q_i_k - self.LambdaP * p_u_k)
                    self.Q[i][k] += self.learning_rate* (diff * p_u_k - self.LambdaQ * q_i_k)
                rmse += diff ** 2
                self.lil_matrix_diff.append((u, i, r, diff))

            rmse /= self.num_trainingData
            rmse = math.sqrt(rmse)

            # 每次迭代之后降低学习率，提高正确率，防止因为学习率过高错过正确的值
            self.learning_rate *= decay_factor
            print('RMSE in epoch %d: %f' % (epoch+1, rmse))
            self.evaluateWithTrain()

        end=time.time()
        duration=end-start
        print("训练总用时：", "%.6f" % duration, "秒")

        # 将参数保存在文件中，节省后续计算时间
        print('正在保存各个参数...',end='')

        # 1. 保存bu
        f1 = open(USER_BIAS_VEC, 'wb')
        f1.write(struct.pack('i', self.user_num))
        for i in self.user_bias:
            f1.write(struct.pack('d', i))
        f1.close()

        # 2. 保存bi
        f2 = open(ITEM_BIAS_VEC, 'wb')
        f2.write(struct.pack('i', self.item_num))
        for bi in self.item_bias:
            f2.write(struct.pack('d', bi))
        f2.close()

        # 保存P矩阵
        f3 = open(P_MATRIX, 'wb')
        f3.write(struct.pack('i', self.user_num))
        f3.write(struct.pack('i', self.factors))
        for p in self.P:
            for pi in p:
                f3.write(struct.pack('d', pi))
        f3.close()

        # 保存Q矩阵
        f4 = open(Q_MATRIX, 'wb')
        f4.write(struct.pack('i', self.item_num))
        f4.write(struct.pack('i', self.factors))
        for q in self.Q:
            for qi in q:
                f4.write(struct.pack('d', qi))
        f4.close()

        # 保存lil_matrix_diff和用户打分平均值
        f5 = open(LIL_MATRIX, 'wb')
        f5.write(struct.pack('d', self.overall_train_mean_rating))
        f5.write(struct.pack('i', self.num_trainingData))
        for u, i, r, diff in self.lil_matrix_diff:
            f5.write(struct.pack('i', u))
            f5.write(struct.pack('i', i))
            f5.write(struct.pack('d', r))
            f5.write(struct.pack('d', diff))
        f5.close()
        print("保存完成")


    def loadItemAttributeDataset(self):
        with open(NewItemAttributeDataset, 'r') as f:
            item_lines = f.readlines()

        self.item_attr={}
        self.user_item_attrs=defaultdict(list)
        for item_line in item_lines:
            line = item_line.strip()
            if line == "":
                continue
                
            item_id, attr1, attr2 = line.split(',')
            item_id = int(item_id)
            if item_id not in self.item_dict:
                continue
                
            item_name = self.item_dict[item_id]
            attr1, attr2 = float(attr1), float(attr2)
            self.item_attr[item_name] = (attr1, attr2)

        for user, item, rate, diff in self.lil_matrix_diff:
            # {userid:(item_id, 残差, attr1, attr2) ......}
            self.user_item_attrs[user].append((item, diff, self.item_attr[item][0], self.item_attr[item][1]))

    def basic_linear(self, user):
        # equation: Residuals = a * attr1 + b * attr2 + c
        def regression(x, y, p):  # 回归函数
            a, b, c = p
            return a * x + b * y + c
        # 残差函数
        def residuals(p, z, x, y):
            return z - regression(x, y, p)

        l = len(self.user_item_attrs[user])
        # 存当前用户打分的所有商品的attr1
        x = np.array([self.user_item_attrs[user][i][2] for i in range(0, l)])
        # 存当前用户打分的所有商品的attr2
        y = np.array([self.user_item_attrs[user][i][3] for i in range(0, l)])
        # 存商品真实值和预测值之间的残差
        z = np.array([self.user_item_attrs[user][i][1] for i in range(0, l)])
        # 最小二乘法拟合
        plsq = optimize.leastsq(residuals, [0, 0, 0], args=(z, x, y))
        # 获得拟合结果
        a, b, c = plsq[0]

        return a, b, c


    def linear(self):
        #1.从已经存储的.dat文件中读取参数
        print("正在读取保存的参数...")
        self.user_bias = []
        self.item_bias = []
        self.P = []
        self.Q = []
        #按照存储写出的方式读入参数

        #1.1 读取UB_VECTOR.dat
        with open(USER_BIAS_VEC, 'rb') as f:
            byte_str = f.read(4)
            user_len = struct.unpack('i', byte_str)[0]
            for i in range(user_len):
                byte_str = f.read(8)
                x = struct.unpack('d', byte_str)[0]
                self.user_bias.append(x)
        f.close()
        
        #1.2 读取IB_VECTOR.dat
        with open(ITEM_BIAS_VEC, 'rb') as f:
            byte_str = f.read(4)
            item_len = struct.unpack('i', byte_str)[0]
            for i in range(item_len):
                byte_str = f.read(8)
                l = struct.unpack('d', byte_str)[0]
                self.item_bias.append(l)
        f.close()

        #1.3 读取P_MATRIX.dat
        with open(P_MATRIX, 'rb') as f:
            byte_str = f.read(4)
            user_len = struct.unpack('i', byte_str)[0]
            byte_str = f.read(4)
            factor_len = struct.unpack('i', byte_str)[0]
            for i in range(user_len):
                new_list = []
                for j in range(factor_len):
                    byte_str = f.read(8)
                    l = struct.unpack('d', byte_str)[0]
                    new_list.append(l)
                self.P.append(new_list)
        f.close()

        #1.4 读取Q_MATRIX.dat
        with open(Q_MATRIX, 'rb') as f:
            byte_str = f.read(4)
            item_len = struct.unpack('i', byte_str)[0]
            byte_str = f.read(4)
            factor_len = struct.unpack('i', byte_str)[0]
            for i in range(item_len):
                new_list = []
                for j in range(factor_len):
                    byte_str = f.read(8)
                    l = struct.unpack('d', byte_str)[0]
                    new_list.append(l)
                self.Q.append(new_list)
        f.close()

        #1.5 读取LIL_MATRIX.dat
        with open(LIL_MATRIX, 'rb') as f:
            byte_str = f.read(8)
            self.overall_train_mean_rating = struct.unpack('d', byte_str)[0]
            byte_str = f.read(4)
            self.num_trainingData = struct.unpack('i', byte_str)[0]
            for ii in range(self.num_trainingData):
                byte_str = f.read(4)
                u = struct.unpack('i', byte_str)[0]
                byte_str = f.read(4)
                i = struct.unpack('i', byte_str)[0]
                byte_str = f.read(8)
                r = struct.unpack('d', byte_str)[0]
                byte_str = f.read(8)
                err = struct.unpack('d', byte_str)[0]
                self.lil_matrix_diff.append((u, i, r, err))
        f.close()

        #2.读取itemAttribute.csv的数据
        self.loadItemAttributeDataset()

        #3.对所有用户，用最小二乘法拟合残差值
        self.user_para={}
        # self.user_para = defaultdict(list)
        for k, v in self.user_item_attrs.items():
            self.user_para[k] = self.basic_linear(k)


    def linear_predict(self, u, i):
        item_attr = self.item_attr.get(i)
        user_para = self.user_para.get(u)

        if item_attr is None or user_para is None:
            return 0.0

        attr1, attr2 = item_attr
        a, b, c = user_para
        return a * attr1 + b * attr2 + c


    def evaluateWithTrain(self):
        print('通过验证集计算RMSE...',end='')
        num_test_data = 0
        rmse = 0.0
        with open(test_set, 'r') as f:
            for line in f.readlines():
                if line == "":
                        continue
                user_id, item_id, rate_str = line.split(',')
                u = self.user_dict[int(user_id)]
                i = self.item_dict[int(item_id)]
                r = float(rate_str)
                rp1 = self.overall_train_mean_rating + self.user_bias[u] + self.item_bias[i] + self.dot(u, i)
                rp1 = self.correctRank(rp1)
                err1 = r - rp1
                rmse += err1 ** 2
                num_test_data += 1
        rmse = math.sqrt(rmse / num_test_data)
        print(f'验证集上的RMSE为: {rmse}')


    def evaluate(self):
        print("================  通过验证集计算RMSE   ================")
        self.linear()
        num_test_data = 0
        rmse = 0.0
        rmse_with_IA = 0.0

        with open(test_set, 'r') as f, open(RESULT_FOLDER+'eval_result.csv', 'w') as resultf:
            for line in f.readlines():
                if line == "":
                        continue
                user_id, item_id, rate_str = line.split(',')
                u = self.user_dict[int(user_id)]
                i = self.item_dict[int(item_id)]
                r = float(rate_str)
                rp1 = self.overall_train_mean_rating + self.user_bias[u] + self.item_bias[i] + self.dot(u, i)
                rp2 = rp1 + self.linear_predict(u, i)
                rp1 = self.correctRank(rp1)
                rp2 = self.correctRank(rp2)
                
                err1, err2 = r - rp1, r - rp2
                rmse += err1 ** 2
                rmse_with_IA += err2 ** 2

                #写出测试结果，方便实验观察训练效果：用户ID,物品ID，预测值，预测值2(带物品属性），真实打分
                resultf.write(f'{int(user_id)},{int(item_id)},{rp1},{rp2},{r}\n')
                num_test_data += 1
        rmse, rmse_with_IA = math.sqrt(rmse / num_test_data), math.sqrt(rmse_with_IA / num_test_data)
        print(f'验证集上的RMSE为: {rmse}')
        print(f'利用itemattribute的RMSE为: {rmse_with_IA}')


    def predictOnTestDataset(self):#在test.txt上进行预测，得到本次实验结果
        print('开始对test.txt的预测...')
        test_file = []
        with open(testDataset, 'r') as f:
            test_file = f.readlines()
            f.close()

        res1 = open(RESULT_FOLDER+'result.txt', 'w')  # 优化前
        res2 = open(RESULT_FOLDER+'result_with_item_attribute.txt', 'w')  # 优化后
 
        for test in test_file:
            line = test.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
                user_id = int(user_id)
                res1.write(line + '\n')
                res2.write(line + '\n')
            else:
                if line == "":
                    continue
                item_id = int(line)
                u = self.user_dict[user_id]
                i = self.item_dict[item_id]
                rp1 = self.overall_train_mean_rating + self.user_bias[u] + self.item_bias[i] + self.dot(u, i)
                rp2 = rp1 + self.linear_predict(u, i)
                #将预测打分scale在0到100的区间内
                rp1 = self.correctRank(rp1)
                rp2 = self.correctRank(rp2)
                res1.write('%d %f \n' % (item_id, rp1))
                res2.write('%d %f \n' % (item_id, rp2))
        res1.close()
        res2.close()