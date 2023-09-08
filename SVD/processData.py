#1.进行数据统计和清洗工作
#2.划分训练集和测试集
import random
import time
from configAndUtils import *
import psutil
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt

def splitDataset(filename, testPortion):#分割train.txt
    print(f'读入{filename}')
    with open(filename, 'r') as f:
        dataset = f.readlines()
        f.close()

    print(f'将{filename}按照{1 - testPortion}:{testPortion}的比例分为训练集、测试集')
    trainset = open(training_set, 'w')  # 打开trainset文件
    testset = open(test_set, 'w')  # 打开testset文件

    seed_value = 333  # 设置种子值为42，保证每次划分情况一样，为了可重复性 （可去）
    random.seed(seed_value)
    for line in dataset:
        data = line.strip()  #删掉每行数据后的换行符
        if data.find('|') != -1:
            user_id, num_of_rating_items = data.split('|')
        else:
            if data== "":
                continue
            item_id, rating = data.split()
            if random.random() >= testPortion:#按照比例将数据集划分
                trainset.write('%s,%s,%s\n' % (user_id, item_id, rating))
            else:
                testset.write('%s,%s,%s\n' % (user_id, item_id, rating))

    trainset.close()
    testset.close()
    print("划分完成")

#用于统计train.txt、test.txt、itemAttribute.txt信息
def analyseWhloeDataset():
    print("正在统计itemAttribute.txt信息... ")
    # 开始计时
    start_time = time.time()
    with open(itemAttributeDataset, 'r') as f:
        item_attr_data= f.readlines()
        f.close()

    item_num=0
    user_num=0
    user_dict = {}
    item_dict = {}
    item_attr = {}  # 储存商品属性 {商品实际id: (商品实际id, 属性1, 属性2)......}
    whloe_attr_data = []  # 存储itemattribute.txt中的所有行（商品实际id, 属性1, 属性2）
    attr1_mean = 0.0  # 属性1的平均值
    attr2_mean = 0.0  # 属性2的平均值
    attr_data_num=0

    min_user_id = 0
    max_user_id = 0
    min_item_id = 0
    max_item_id = 0

    #TODO 处理真的蛮重要：如果属性值为None，则用0表示（看这样是否合适）！！！
    for attr_data in item_attr_data:
        line = attr_data.strip()
        if line == "":
            continue
        item_id, attr1, attr2 = line.split('|')
        if attr1 == 'None':
            attr1 = 0
        else:
            attr1 = int(attr1)
        if attr2 == 'None':
            attr2 = 0
        else:
            attr2 = int(attr2)
        item_id_int = int(item_id)
        #获取商品最大最小id
        max_item_id =max(item_id_int, max_item_id)
        min_item_id = min(item_id_int, min_item_id)
        # 建立商品的实际id和程序中id的映射
        item_dict[item_id_int] = item_num
        # 将itemattribute.txt中的每行转换为（商品实际id, attr1, attr2）
        item_attr[item_id_int] = (item_id_int, attr1, attr2)
        whloe_attr_data.append((item_id_int, attr1, attr2))
        item_num += 1
        attr_data_num += 1

    # 统计所有属性的平均值
    for item_id, attr1, attr2 in whloe_attr_data:
        attr1_mean += attr1
        attr2_mean += attr2
    attr1_mean /= len(whloe_attr_data)
    attr2_mean /= len(whloe_attr_data)

    #从min_item_id到max_item_id，不一定每个id的商品属性都被提供
    #缺失id的属性用平均值代替
    #TODO 处理商品缺失信息： 暂时使用平均值来弄...有点不对劲
    attr_list = []#商品id从min_item_id到max_item_id，是连续的
    item_no_attrs = 0 #记录没有属性记录的商品数量
    for i in range(min_item_id, max_item_id + 1):
        try:
            # 在attr_list中找是否存在对应的商品id
            #获取到商品ID为 i 的属性信息元组 (item_id, attr1, attr2)，得到属性元组 attr1 和 attr2
            attr_list.append([item_attr[i][1], item_attr[i][2]])
        except KeyError:
            # 没找到则该商品缺失属性信息，用平均值补上
            attr_list.append([attr1_mean, attr2_mean])
            item_no_attrs += 1

    print("正在统计train.txt信息...")
    # 开始计时
    start_time = time.time()
    #统计用户id的最大最小值 建立数据集用户id（可能不连续，同PR）和在程序中id的映射
    with open(trainDataset, 'r') as f:
        wholeDataset = f.readlines()
        f.close()

    user_ratings = {}  # 用于储存用户评分：{ 用户1: [评分1, 评分2, ...], ... }
    item_ratings = {}  # 用于储存商品评分: { 商品1: [评分1, 评分2, ...], ... }
    train_data_num = 0 #记录train.txt数据条数
    Rate = 0.0
    rating_scale=[0, 100] #用户打分范围为0：100
    #记录该分数的打分人数：例如100分的有x人打
    num_of_this_rating = [0 for _ in range(0, rating_scale[1] - rating_scale[0] + 1)]
    whole_set_mean_rating = 0.0  # 训练集中所有评分的平均数



    for data in wholeDataset:
        line = data.strip()
        if line.find('|') != -1:
            user_id, user_item_count = line.split('|')
            user_id_int = int(user_id) #将str转化成int
            try:
                u = user_dict[user_id_int] #TODO see it ：在字典中无法被找到（包含去重功能）
            except KeyError:
                user_ratings[user_id_int] = [] #初始化评分字典
                #建立实际id和程序id的映射
                user_dict[user_id_int] = user_num
                #寻找用户最大最小id
                max_user_id=max(user_id_int, max_user_id)
                min_user_id=min(user_id_int, min_user_id)
                user_num += 1
        else:
            if line == "":
                continue
            item_id, rating_str = line.split()
            train_data_num += 1
            item_id_int = int(item_id)
            Rate = float(rating_str)
            num_of_this_rating[math.floor(Rate)] += 1
            whole_set_mean_rating += Rate
            user_ratings[user_id_int].append(Rate)
            try:
                i =item_dict[item_id_int]
            except KeyError:
                item_dict[item_id_int] = item_num
                max_item_id=max(item_id_int, max_item_id)
                min_item_id =min(item_id_int, min_item_id)
                item_num += 1
            try:
                item_ratings[item_id_int].append(Rate)
            except KeyError:
                item_ratings[item_id_int] = []
                item_ratings[item_id_int].append(Rate)

    whole_set_mean_rating /= train_data_num
    # 结束计时
    end_time = time.time()
    print("处理完毕")
    # 计算时长
    duration = end_time - start_time
    # 打印时长
    print("统计时长：", "%.6f" % duration, "秒")
    print(f'train.txt中最大用户ID为{max_user_id},最小用户为ID为{min_user_id}')
    print(f'train.txt中最大物品ID为{max_item_id},最小物品为ID为{min_item_id}\n')


    print("正在统计test.txt信息...")
    # 开始计时
    start_time = time.time()
    with open(testDataset, 'r') as f:
        test_data = f.readlines()
        f.close()

    test_data_num = 0  # 测试集中需要测试的数据条数
    for test in test_data:
        line = test.strip()
        if line.find('|') != -1:
            user_id, user_item_count = line.split('|')
            user_id_int = int(user_id)
            try:
                u =user_dict[user_id_int]
            except KeyError:
                user_ratings[user_id_int] = []
                user_dict[user_id_int] =user_num
                max_user_id = max(user_id_int, max_user_id)
                min_user_id = min(user_id_int, min_user_id)
                user_num += 1
        else:
            if line == "":
                continue
            test_data_num += 1
            item_id_int = int(line)
            try:
                i =item_dict[item_id_int]
            except KeyError:
                item_dict[item_id_int] =item_num
                max_item_id = max(item_id_int, max_item_id)
                min_item_id = min(item_id_int, min_item_id)
                item_num += 1

    # 结束计时
    end_time = time.time()
    print("处理完毕")
    # 计算时长
    duration = end_time - start_time
    # 打印时长
    print("统计时长：", "%.6f" % duration, "秒")
    # print(f'Train.txt、test.txt中最大用户ID为{max_user_id},最小用户为ID为{min_user_id}')
    # print(f'Train.txt、test.txt中最大物品ID为{max_item_id},最小物品为ID为{min_item_id}\n')



     #储存“物品实际 id，attr1，attr2”的形式（id从：0到max_itemid连续）
    print("正在写出到processed_itemAttribute.csv...")
    with open(NewItemAttributeDataset, 'w') as f:
        for i in range(len(attr_list)):
            f.write('%d,%d,%d\n' % (i, attr_list[i][0], attr_list[i][1]))
        f.close()
    # 结束计时
    end_time = time.time()
    print("处理完毕")
    # 计算时长
    duration = end_time - start_time
    # 打印时长
    print("统计时长：", "%.6f" % duration, "秒\n")

    print("正在生成统计信息文件...")
    with open(DATA_STATISTICS_FOLDER+'dataStatistics.txt', 'w',encoding='utf-8') as statf:
        statf.write('数据集中最小用户id: %d\n' % min_user_id)
        statf.write('数据集中最大的用户id: %d\n' % max_user_id)
        statf.write('数据集中用户总数: %d\n' % user_num)
        statf.write('数据集中最小商品id: %d\n' % min_item_id)
        statf.write('数据集中最大商品id: %d\n' % max_item_id)
        statf.write('数据集中商品总数: %d\n' % item_num)
        # statf.write('没有被提供属性的商品数: %d\n' % item_no_attrs)
        statf.write('train.txt评分总数: %d\n' % train_data_num)
        statf.write('train.txt所有得分的平均值: %f\n' %whole_set_mean_rating)
        statf.write('属性集数据条数: %d\n' % attr_data_num)
        statf.write('属性1的平均值: %f\n' % attr1_mean)
        statf.write('属性2的平均值: %f\n' % attr2_mean)
        statf.write('没有被提供属性的商品数: %d\n' % item_no_attrs)
        statf.close()
    print("已写出dataStatistics.txt")

    with open(DATA_STATISTICS_FOLDER+'userAvgScore.csv', 'w') as f:
        s = sorted(user_ratings.items(), key=lambda x: x[0])
        for u, r in s:
            f.write(str(u))#usrid(注意不一定所有人都打分了）
            f.write(',')
            total_sum = 0.0
            for rate in r:
                total_sum += rate
            if len(r) != 0:
                mean = total_sum / len(r)
            else:
                mean = 0.0
            f.write(str(mean)) #该用户打出的平均分
            f.write(',')
            f.write(str(len(r))) #该用户给多少个商品进行了打分
            f.write('\n')
        f.close()
    print("已写出userAvgScore.csv; 格式：usrid:avgScore:ratingTimes")

    item_avg_list=[]#用于画出商品信息
    item_being_rated_times=[]#商品被打分次数
    with open(DATA_STATISTICS_FOLDER+'itemAgScore.csv', 'w') as f:
        s = sorted(item_ratings.items(), key=lambda x: x[0])
        for i, r in s:
            f.write(str(i)) #itemID
            f.write(',')
            total_sum = 0.0
            for rate in r:
                total_sum += rate
            if len(r) != 0:
                mean = total_sum / len(r)
            else:
                mean = 0.0
            format(float(mean), '.2f')
            item_avg_list.append(mean)
            f.write(str(mean)) #该商品获得的平均分
            f.write(',')
            f.write(str(len(r))) #有多少个用户给这个商品打分
            item_being_rated_times.append(len(r))
            f.write('\n')
        f.close()



    print("已写出itemAgScore.csv; 格式：item_id:avgScore:ratedTimes")
    #写出打分分布情况
    with open(DATA_STATISTICS_FOLDER+'ratingDistribution.csv', 'w') as rate:
        for i, j in enumerate(num_of_this_rating):
            rate.write('%d,%d\n' % (i + rating_scale[0], j))
        rate.close()
    print("已写出ratingDistribution.csv")
    #写出usr_dict/item_dict/item_attr/item_num/usr_num：方便后续使用
    print("正在写出usr_dict/item_dict/item_attr/item_num/usr_num信息")
    with open(user_dictFile,'w') as f:
        for key, value in user_dict.items():
            f.write(f"{key}: {value}\n")
        f.close()

    with open(item_dictFile,'w') as f:
        for key,value in item_dict.items():
            f.write(f"{key}: {value}\n")
        f.close()

    with open(item_attrFile,'w') as f:
        for key,value in item_attr.items():
            f.write(f"{key},{value[0]},{value[1]},{value[2]}\n")
        f.close()
    with open(user_numFile,'w') as f:
        f.write(str(user_num))
        f.close()
    with open(item_numFile,'w') as f:
        f.write(str(item_num))
        f.close()

    print('写出完毕')
    #画几个必要的图
    # 设置字体
    # plt.rcParams['font.family'] = 'SimHei'
    # hist, bins, _=plt.hist(item_avg_list, bins=10,rwidth=0.9)  # 可以根据需要调整bins的数量
    # plt.xticks(bins)  # x轴刻度设置为箱子边界
    #
    # plt.xlabel('平均分/分')
    # plt.ylabel('商品数')
    # plt.title('商品得分均值分布')
    # plt.show()


    # plt.rcParams['font.family'] = 'SimHei'
    # hist, bins, _ = plt.hist(item_being_rated_times, bins=8,range=(0, 300), rwidth=0.9)  # 可以根据需要调整bins的数量
    # plt.xticks(bins)  # x轴刻度设置为箱子边界
    #
    # plt.xlabel('被打分次数/次')
    # plt.ylabel('频数')
    # plt.title('商品被打分次数分布')
    # plt.show()


#划分数据集
splitDataset(trainDataset, TEST_PORTION)

memory_usage1 =getProcessMemory()

#统计所有数据集情况
analyseWhloeDataset()
memory_usage2 = getProcessMemory()
memory_usage=memory_usage2+memory_usage1
print(f"Process Data Memory usage(it indicates the maximum memory that a process used): {memory_usage} MB")

