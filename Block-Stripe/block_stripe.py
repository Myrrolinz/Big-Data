import json
import os
import numpy as np
import math
import pickle
from collections import defaultdict

SAVE_DATA_FOLDER = './Results/Block_data/'
Directly_REVERSE_MATRIX_FOLDER = './Results/Directly_reverse_marix/'
RESULT_FOLDER = './Results/'

data_txt = './Data/data.txt'
new_data_txt = './Data/new_data.txt'
pages_txt = './Data/pages.txt'
number_txt = './Data/number.txt'
result_data_file = './Results/block_stripe_result.data'
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

def preprocessing(data_txt, new_data_txt):
    # 读取文件并建立节点编号映射关系
    edges = []
    with open(data_txt, 'r') as f:
        for line in f:
            src, dest = map(int, line.strip().split())
            if (src, dest) not in edges:
                edges.append((src, dest))
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])

    node_list = sorted(list(nodes))  # 得到排序后的节点列表

    # 建立节点映射字典，将旧id映射为新id
    node_dict = {}
    for i, node_id in enumerate(node_list):
        node_dict[node_id] = i + 1

    # 将边列表中的节点id映射为新id
    new_edges = []
    for edge in edges:
        new_src = node_dict[edge[0]]
        new_dest = node_dict[edge[1]]
        new_edges.append((new_src, new_dest))
    with open(new_data_txt, 'w') as f:
        for edge in new_edges:
            new_src = edge[0]
            new_dest = edge[1]
            f.write(str(new_src) + " " + str(new_dest) + "\n")
    return node_dict

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
        temp_save_dict = dict()
        temp_count = 1
        while (temp_count % compute_size != 1 or temp_count == 1):
            des = temp_count + num * compute_size
            temp_count += 1

            # if reverse_matrix.get(des) is None : continue
            if des not in list(files_of_des.keys()): continue

            # 定位在哪一个reverse文件里
            des_src_list = []
            for file_number in files_of_des[des]:
                des_reverse_file = Directly_REVERSE_MATRIX_FOLDER + 'Directly_reverse_' + str(file_number) + '.txt'
            
                with open(des_reverse_file, 'r', encoding='utf-8') as f:
                    reverse_data = json.loads(f.read())
                    if not reverse_data.get(str(des)) is None:
                        des_src_list.extend(reverse_data[str(des)])

            for src in des_src_list:
                if temp_save_dict.get(src) is None:
                    # 定位src的d
                    d = out_of_nodes[src]
                    temp_save_dict[src] = [d, [des]]
                else:
                    temp_save_dict[src][1].append(des)

        # 排序并保存
        temp_save_dict = dict(sorted(temp_save_dict.items(), key=lambda d: d[0], reverse=False))

        for key in temp_save_dict:
            temp_save_dict[key][1].sort()

        with open(save_file_name, 'w', encoding='utf-8') as out:
            out.write(json.dumps(temp_save_dict))

        if (temp_count + num * compute_size) > pages: break
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
    with open(data, 'r', encoding='utf-8') as f:
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
    temp = np.full(N, 1/N, dtype=np.float64)
    with open('./r_old.txt', 'w', encoding='utf-8') as t:
        for i in temp:
            t.write(str(i) + '\n')
    if_end = 0
    m_list = os.listdir(M)
    K = N // size
    if N % size != 0:
        K += 1
    print(K)
    E = 0.0001 #TODO originally E = 1e-9, maybe still needs modify

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
                            i = np.float64(i)
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
                    i = np.float64(i)
                    new.write(str((i + (1 - S) / N)) + '\n')

        with open('./r_new.txt', 'r', encoding='utf-8') as new:
            with open('./r_old.txt', 'r', encoding='utf-8') as old:
                for i, j in zip(new, old):
                    # e += abs(np.float64(i)-np.float64(j))
                    tmp = abs(np.float64(i)-np.float64(j))
                    if e < tmp:
                        e = tmp

        with open('./r_new.txt', 'r', encoding='utf-8') as new:
            with open('./r_old.txt', 'w', encoding='utf-8') as old:
                old.truncate()
                for i in new:
                    i = np.float64(i)
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
    node_dict = {}
    node_dict = preprocessing(data_txt, new_data_txt)
    print("PageRank Block Stripe Begin")
    pre_block(new_data_txt) # 预处理
    if flag_test:
        with open(pages_txt, 'r', encoding='utf-8') as f:
            N = f.read()
        N = int(N, 10)
        print("now max_number is: ",N)
        sparse_matrix_multiply(compute_size, N, SAVE_DATA_FOLDER)

        r = dict()
        count = 1
        with open('./r_new.txt', 'r', encoding='utf-8') as f:
            for i in f:
                r[count] = np.float64(i.rstrip())
                count += 1

        result = dict()
        result = dict(sorted(r.items(), key=lambda d: d[1], reverse=True))  # 按照rank排序
        
        # 将映射字典键值反转
        reverse_mapping = {v: k for k, v in node_dict.items()}
        result_original = {}
        for mapped_node, pagerank in result.items():
            original_node = reverse_mapping[mapped_node]
            result_original[original_node] = pagerank
            
        with open(top100_result_file, 'w', encoding='utf-8') as f:
            count = 0
            for i in result_original:
                if count<100:
                    f.write('[' + str(i) + ']    [' + str(result_original[i]) + ']\n')
                    count+=1
                else:
                    break
        
        # 写data文件
        with open(result_data_file, 'wb') as f:
            pickle.dump(result_original, f)
    os.system("pause")