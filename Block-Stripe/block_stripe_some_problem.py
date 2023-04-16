import json
import os
import numpy as np
import math
import pickle
from collections import defaultdict

SAVE_DATA_FOLDER = './Data/Block_data/'
Directly_REVERSE_MATRIX_FOLDER = './Data/Directly_reverse_marix/'
RESULT_FOLDER = '../Results/'

data_txt = '../Data/data.txt'
pages_txt = './Data/pages.txt'
result_data_file = './Data/block_stripe_result.data'
top100_result_file = RESULT_FOLDER + 'Block_stripe_result.txt'


block_size = 1000
compute_size = 50
b = 0.85

# flags, which are used to control the flow of program
flag_pre_block_test = True
flag_test = True

# mkdirs
if not os.path.exists(SAVE_DATA_FOLDER):
    os.makedirs(SAVE_DATA_FOLDER)
# 如果路径已存在，清空所有文件
else:
    if len(os.listdir(SAVE_DATA_FOLDER))!=0:
        for items in os.listdir(SAVE_DATA_FOLDER): 
            os.remove(SAVE_DATA_FOLDER + items)

if not os.path.exists(Directly_REVERSE_MATRIX_FOLDER):
    os.makedirs(Directly_REVERSE_MATRIX_FOLDER)
else:
    if len(os.listdir(Directly_REVERSE_MATRIX_FOLDER))!=0:
        for items in os.listdir(Directly_REVERSE_MATRIX_FOLDER): 
            os.remove(Directly_REVERSE_MATRIX_FOLDER + items)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


def save_data_for_RI(s_matrix, name_count):
    # 排序
    s_matrix = dict(sorted(s_matrix.items(), key=lambda d: d[0], reverse=False))
    for item_des in s_matrix.keys():
        s_matrix[item_des].sort()

    # 保存当前list里的所有内容到磁盘中
    reverse_file_name = Directly_REVERSE_MATRIX_FOLDER + 'Directly_reverse_' + str(name_count) + '.txt'
    with open(reverse_file_name, 'w', encoding='utf-8') as save_f:
        save_f.write(json.dumps(s_matrix))

def process_block_data(pages, compute_size, files_of_des, out_of_nodes):
    # save file according to the value of size
    k = math.ceil(pages / compute_size)

    for num in range(k):
        save_file_name = os.path.join(SAVE_DATA_FOLDER, f'{num}.txt')
        temp_save_dict = {}
        for des in range(num * compute_size + 1, (num + 1) * compute_size + 1):
            if des not in files_of_des: 
                continue
            des_src_list = []
            for file_number in files_of_des[des]:
                des_reverse_file = os.path.join(Directly_REVERSE_MATRIX_FOLDER, f'Directly_reverse_{file_number}.txt')
                with open(des_reverse_file, 'r', encoding='utf-8') as f:
                    reverse_data = json.load(f)
                    des_src_list.extend(reverse_data.get(str(des), []))

            for src in des_src_list:
                if src not in temp_save_dict:
                    d = out_of_nodes[src]
                    temp_save_dict[src] = [d, [des]]
                else:
                    temp_save_dict[src][1].append(des)

        temp_save_dict = dict(sorted(temp_save_dict.items(), key=lambda x: x[0]))
        for key in temp_save_dict:
            temp_save_dict[key][1].sort()

        with open(save_file_name, 'w', encoding='utf-8') as out:
            json.dump(temp_save_dict, out)

        if des >= pages:
            break
    
    print("Block Data Saved")
    
            
def pre_block(data):
    """
       :param data: path of data.txt
       :return: None
       for each element in matrix: src:[degree, [list_of_out]]
       """
    print("Preprocessing Blocks")

    pages = -1
    name_count = 0
    # 使用defaultdict替换字典的setdefault方法，可以更简洁地初始化字典中的值
    s_matrix = defaultdict(list)
    out_of_nodes = defaultdict(int)
    files_of_des = defaultdict(list)
    current_num_des_nodes = []

    # 建立倒排索引
    with open(data_txt, 'r', encoding='utf-16') as f:
        for line in f:
            src = int(line.split()[0].split()[0], 10)
            des = int(line.split()[1].split()[0], 10)
            pages = max(src, pages, des)

            # 记录出度
            out_of_nodes[src] += 1

            # 记录des所在的文件
            if name_count not in files_of_des[des]:
                files_of_des[des].append(name_count)

            if des not in current_num_des_nodes:
                if len(current_num_des_nodes) % block_size == 0:
                    if len(current_num_des_nodes) != 0:
                        # 如果此时长度达到要求，需要保存内容
                        # 保存当前list里的所有内容到磁盘中
                        save_data_for_RI(s_matrix, name_count)
                        name_count += 1

                        # 清空内存
                        s_matrix = defaultdict(list)
                        current_num_des_nodes = []

                        # 保存新的内容
                        s_matrix[des].append(src)
                        current_num_des_nodes.append(des)

                    else:
                        # curr为0，表示没有内容，直接保存新的内容到内存中即可
                        s_matrix[des].append(src)
                        current_num_des_nodes.append(des)
                else:
                    # des不在列表里，此时长度没有到达要求，直接保存到内存中
                    s_matrix[des].append(src)
                    current_num_des_nodes.append(des)
            else:
                # 已有des，保存即可
                s_matrix[des].append(src)
         # 处理最后一个块
        save_data_for_RI(s_matrix, name_count)

        with open(pages_txt, 'w', encoding='utf-8') as pages_file:
            pages_file.write(str(pages))
        
        # 清空内存
        s_matrix = defaultdict(list)
        current_num_des_nodes = []
    
    print("RevertedIndex Data Saved")
    
    process_block_data(pages, compute_size, files_of_des, out_of_nodes)
    
    
def sparse_matrix_multiply(size, N, M):
    """
    compute one block each time, using one block of M and r_old
    :param size: size of each block
    :param N: number of dot
    :param M: the folder of M
    :param r_old: the folder of r_old
    :return: r_new, S
    r_new: a list of N/K elements, which K is the number of block
    S: current S
    """
    with open('./r_new.txt', 'w', encoding='utf-8') as new:
        new.truncate()
    with open('./r_new_temp.txt', 'w', encoding='utf-8') as new:
        new.truncate()
    with open('./r_old.txt', 'w', encoding='utf-8') as old:
        old.truncate()
    # 初始化 r_old
    temp = np.full(N, 1/N)
    with open('./r_old.txt', 'w', encoding='utf-8') as t:
        for i in temp:
            t.write(str(i) + '\n')
    if_end = 0
    m_list = os.listdir(M)
    K = N // size
    if N % size != 0:
        K += 1
    print(K)
    E = pow(10, -9)

    while not if_end:
        r_new = np.zeros(size)
        start = 1  # 每一块r_new的起始索引
        end = start + size - 1  # 每一块r_new的结束索引
        S = 0

        for matrixs in range(len(m_list)):
            m_now = M + str(matrixs) + '.txt'
            with open(m_now, 'r', encoding='utf-8') as f1:
                matrix = json.loads(f1.read())
            if matrix == {}:
                continue
            else:
                with open('./r_old.txt', 'r', encoding='utf-8') as f2:
                    count = 1
                    i = f2.readline().rstrip()
                    for src in matrix:
                        while int(src, 10) != count:
                            count += 1
                            i = f2.readline().rstrip()
                        if int(src, 10) == count:
                            d = matrix[src][0]
                            i = float(i)
                            for des in matrix[src][1]:
                                t = des
                                if t > end:
                                    c = t - start
                                    c = c // size * size
                                    with open('./r_new_temp.txt', 'a', encoding='utf-8') as f3:
                                        for x in range(c):
                                            f3.write('0.0\n')
                                    start += c
                                    end = start + size - 1
                                r_new[des - start] += b * i / d
                            count += 1
                            i = f2.readline().rstrip()
                        else:
                            count += 1

            start += size
            end = start + size - 1

            if end > N:
                end = N
            with open('./r_new_temp.txt', 'a', encoding='utf-8') as f3:
                for j in r_new:
                    f3.write(str(j) + '\n')
                    S += j
            if start <= end:
                r_new = np.zeros(end - start + 1)

        e = 0.0

        with open('./r_new_temp.txt', 'r', encoding='utf-8') as t:
            with open('./r_new.txt', 'w', encoding='utf-8') as new:
                for i in t:
                    i = float(i)
                    new.write(str((i + (1 - S) / N)) + '\n')

        with open('./r_new.txt', 'r', encoding='utf-8') as new:
            with open('./r_old.txt', 'r', encoding='utf-8') as old:
                for i, j in zip(new, old):
                    e += abs(float(i)-float(j))

        with open('./r_new.txt', 'r', encoding='utf-8') as new:
            with open('./r_old.txt', 'w', encoding='utf-8') as old:
                old.truncate()
                for i in new:
                    i = float(i)
                    old.write(str(i) + '\n')

        print(e)
        if abs(e) <= E:
            if_end = 1
        else:
            with open('./r_new.txt', 'w', encoding='utf-8') as new:
                new.truncate()
            with open('./r_new_temp.txt', 'w', encoding='utf-8') as new:
                new.truncate()

if __name__ == '__main__':
    print("This is %s" % 'Page Rank')

    pre_block(data_txt) # 预处理
    if flag_test:
        with open(pages_txt, 'r', encoding='utf-8') as f:
            n = f.read()
        n = int(n, 10)
        print(n)
        sparse_matrix_multiply(compute_size, n, SAVE_DATA_FOLDER)

        r = dict()
        count = 1
        with open('./r_new.txt', 'r', encoding='utf-8') as f:
            for i in f:
                r[count] = float(i.rstrip())
                print(r[count])
                count += 1

        result = dict()
        result = dict(sorted(r.items(), key=lambda d: d[1], reverse=True))  # 按照rank排序
        with open(top100_result_file, 'w', encoding='utf-8') as f:
            count = 0
            for i in result:
                if count<100:
                    f.write(str(i) + '\t\t' + str(result[i]) + '\n')
                    count+=1
                else:
                    break
        
        # 写data文件
        with open(result_data_file, 'wb') as f:
            pickle.dump(result, f)

        os.remove('./r_new.txt')
        os.remove('./r_new_temp.txt')
        os.remove('./r_old.txt')
        os.remove(pages_txt)


    os.system("pause")