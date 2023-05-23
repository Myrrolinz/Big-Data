import time
from typing import Callable
import torch
import numpy as np
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def time_test(desc: str, handler: Callable, *args, **kwargs):
    start = time.time()
    rtn = handler(*args, **kwargs)
    print(f"{desc}: {time.time() - start: .2f}s")
    return rtn


def save_ckpt(path, obj):
    # torch.save(obj, path)
    time_test(f"save_ckpt {path} ", torch.save, obj, path)


def load_ckpt(path):
    # return torch.load(path)
    return time_test(f"load_ckpt {path}", torch.load, path)
