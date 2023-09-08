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

def create_path(path:str, remain=True):
    if os.path.exists(path):
        if remain:
            print(f"{path} already exists.")
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
    return 0


if __name__ == "__main__":

    print(bytes2int(int2bytes(22)))





