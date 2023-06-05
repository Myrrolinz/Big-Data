# !/usr/bin/env python
# -*- coding: utf-8 -*-
# cython:language_level=3
# @Time    : 2023/5/27 15:39
# @File    : main.py
import gc
import os
import csv
import time
import pandas as pd
import psutil as psutil
# import dask.dataframe as dd



def main(df, path):
    # 检测目录是否存在
    # if not os.path.exists(path):
    #     # 不存在则创建
    #     os.mkdir(path)
    # 获取user不重复值
    users_list = df['user'].unique()
    with open('Data/train_train.txt', 'w', encoding='utf-8') as f:
        for user in users_list:
            # 获取每个user的数据
            user_df = df[df['user'] == user]
            # 获取行数
            user_df_len = len(user_df)
            f.write(f'{user}|{user_df_len}\n')
            # 遍历user_df的每一行，获取item和score
            for index, row in user_df.iterrows():
                item = row['item']
                score = row['score']
                # 在文件最后一行写入item和score
                f.write(f'{item} {score}\n')
            # # 新建txt文件
            # with open(f'{path}/{user}.txt', 'w', encoding='utf-8') as f:
            #     # 在第一行写入userid|行数
            #     f.write(f'{user}|{user_df_len}\n')
            #     # 遍历user_df的每一行，获取item和score
            #     for index, row in user_df.iterrows():
            #         item = row['item']
            #         score = row['score']
            #         # 在文件最后一行写入item和score
            #         f.write(f'{item}  {score}\n')


def read_csv(filename):
    data = {}
    with open(filename, 'r', encoding='utf-8') as file:
        reader: csv.DictReader = csv.DictReader(file)
        for row in reader:
            user = row['user']
            item = row['item']
            score = row['score']
            if user in data:
                data[user].append({'item': item, 'score': score})
            else:
                data[user] = [{'item': item, 'score': score}]
    return data


def main_dict(df, path):
    # 检测目录是否存在
    # if not os.path.exists(path):
    #     # 不存在则创建
    #     os.mkdir(path)

    # 读取CSV文件数据并转换为字典
    data = read_csv(df)

    with open(f'Data/{path}.txt', 'w', encoding='utf-8') as f:
        for user, user_data in data.items():
            user_df_len = len(user_data)
            
            f.write(f'{user}|{user_df_len}\n')
            for item_data in user_data:
                item = item_data['item']
                score = item_data['score']
                f.write(f'{item} {score}\n')
    # 遍历用户数据字典，将数据写入对应的txt文件
    # for user, user_data in data.items():
    #     user_df_len = len(user_data)
    #     with open(f'{path}/{user}.txt', 'w', encoding='utf-8') as f:
    #         f.write(f'{user}|{user_df_len}\n')
    #         for item_data in user_data:
    #             item = item_data['item']
    #             score = item_data['score']
    #             f.write(f'{item}  {score}\n')


# def main_dd(filename, path):
#     # 检测目录是否存在
#     if not os.path.exists(path):
#         # 不存在则创建
#         os.mkdir(path)

#     # 使用Dask从CSV文件创建延迟计算的DataFrame
#     dask_df = dd.read_csv(filename, encoding='utf-8')

#     # 获取user不重复值
#     users_list = dask_df['user'].unique().compute()  # 计算唯一值

#     for user in users_list:
#         # 获取每个user的数据
#         user_df = dask_df[dask_df['user'] == user]

#         # 获取行数
#         user_df_len = len(user_df)

#         # 新建txt文件
#         with open(f'{path}/{user}.txt', 'w', encoding='utf-8') as f:
#             # 在第一行写入userid|行数
#             f.write(f'{user}|{user_df_len}\n')

#             # 遍历user_df的每一行，获取item和score
#             for _, row in user_df.iterrows():
#                 item = row['item']
#                 score = row['score']

#                 # 在文件最后一行写入item和score
#                 f.write(f'{item}  {score}\n')


# os.mkdir('pandas')
# os.mkdir('dict')
# os.mkdir('Dask')

# print("======pandas======")
# start_memory = psutil.Process().memory_info().rss / 1024 / 1024
# start_time = time.perf_counter()
# # 读取csv
# train_train_df = pd.read_csv('train_train.csv', encoding='utf-8')
# train_test_df = pd.read_csv('train_test.csv', encoding='utf-8')
# main(train_test_df, 'pandas/test')
# main(train_train_df, 'pandas/train')
# end_memory = psutil.Process().memory_info().rss / 1024 / 1024
# end_time = time.perf_counter()
# execution_time = end_time - start_time
# memory_usage = end_memory - start_memory
# print(f"Execution Time: {execution_time:.4f} seconds")
# print(f"Memory Usage: {memory_usage:.2f} MB")

# gc.collect()  # 手动gc

print("======dict======")
start_memory = psutil.Process().memory_info().rss / 1024 / 1024
start_time = time.perf_counter()
# main_dict('Data/train_train.csv', 'train_train')
main_dict('Data/train_test.csv', 'train_test')
end_memory = psutil.Process().memory_info().rss / 1024 / 1024
end_time = time.perf_counter()
execution_time = end_time - start_time
memory_usage = end_memory - start_memory
print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Memory Usage: {memory_usage:.2f} MB")

# gc.collect()  # 手动gc

# print("======Dask======")
# start_memory = psutil.Process().memory_info().rss / 1024 / 1024
# start_time = time.perf_counter()
# main_dict('train_train.csv', 'Dask/train_train')
# main_dict('train_test.csv', 'Dask/train_test')
# end_memory = psutil.Process().memory_info().rss / 1024 / 1024
# end_time = time.perf_counter()
# execution_time = end_time - start_time
# memory_usage = end_memory - start_memory
# print(f"Execution Time: {execution_time:.4f} seconds")
# print(f"Memory Usage: {memory_usage:.2f} MB")