from contextlib import AsyncExitStack
import time
import os
import pickle
from typing import Callable
import numpy as np
import shutil
from config import *

# 测试时间
def time_test(desc:str, handler:Callable, *args, **kwargs):
    start = time.time()
    rtn = handler(*args, **kwargs)
    print(f"{desc}: {time.time() - start: .2f}s")
    return rtn

# 读入文件
def read_data(filename:str, batch_size:int=-1):
    """
    read graph data from `filename`
    :param filename: N lines. Each line contains two positive integers representing the start and end points of an edge.
    :batch_size: The number of lines read into memory at a time. Read All at once if `batch_size` <= 0
    :return: edges and number of points
    """
    edges = []
    maxn = -1
    count = 0
    with open(filename, 'r', encoding=DATA_ENCODING) as f:
        for line in f:
            src, des = map(int, line.split())
            maxn = max(maxn, src, des)
            edges.append((src, des))
            count += 1
            if (count == batch_size):
                yield edges, maxn # 返回边和最大节点
                edges = []
                count = 0
    yield edges, maxn
    
    

def save_result(result:dict, filename:str, topn=100):
    """
    save the whole `result` and save `topn` result in `filename`
    """
    _, dumpfile = os.path.split(filename)
    dumpfile = os.path.join(MIDDLE_PATH, os.path.splitext(dumpfile)[0] + ".data")
    with open(dumpfile, 'wb') as f:
        pickle.dump(result, f)
    result = sorted(result.items(), key=lambda x:x[1], reverse=True)
    if topn > 0:
        result = result[:topn] 
    with open(filename, 'w', encoding='utf-8') as f:
        for line in result:
            f.write(f"{line[0]:<10}  {line[1]}\n")
    print("file saved")
        
def load_result(filename:str):
    _, dumpfile = os.path.split(filename)
    dumpfile = os.path.join(MIDDLE_PATH, os.path.splitext(dumpfile)[0] + ".data")
    with open(dumpfile, 'rb') as f:
        rtn = pickle.load(f)
    return rtn

def compare_results(result1:dict, result2:dict, norm):
    """
    compare L1 or L2 metric between 2 results
    """
    assert(len(result1) == len(result2)), f"{len(result1)} != {len(result2)}"
    length = len(result1)
    result1 = sorted(result1.items(), key=lambda x:x[0])
    result2 = sorted(result2.items(), key=lambda x:x[0])
    v1 = np.array(list(zip(*result1))[1])
    v2 = np.array(list(zip(*result2))[1])
    dist = np.linalg.norm(v1 - v2, norm) / length
    print(f"average L{norm} dist: {dist}")

def int2bytes(a:int):
    return a.to_bytes(SIZE_INT, 'little', signed=True)

def bytes2int(bytes):
    return int.from_bytes(bytes, 'little', signed=True)

# def get_index_map(node_list):
#     """
#     map: node_list -> [0, len(node_list))
#     """
#     m = {}
#     for i, e in enumerate(node_list):
#         m[e] = i
#     return m

def create_path(path:str, remain=True):
    if os.path.exists(path):
        if remain:
            print(f"{path} is exist.")
        else:
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"everything in {path} are removed.")
    else:
        os.makedirs(path)
        print(f"{path} is created.")


def setup(stripe:bool=False, remain_blocks:bool=False) -> int:
    if not os.path.exists(DATA_IN):
        print(f"Please put datas in {DATA_IN}.")
        return -1
    create_path(RESULT_PATH)
    create_path(MIDDLE_PATH)

    if(stripe):
        create_path(SGRAPH_PATH)
        create_path(BLOCK_PATH, remain_blocks)
        create_path(RANK_PATH, False)
    return 0


if __name__ == "__main__":
    # a = {1:2, 3:3, 2:4}  # 2 4 3
    # b = {1:3, 2:2, 3:5}  # 3 2 5
    # compare_results(a, b, 2)

    # for data in read_data(DATA_IN, 1):
    #     print(data)
    #     print("----")

    print(bytes2int(int2bytes(22)))





