from os.path import join
import os
# input and out path


# DATA_IN = join(DATA_PATH, "data.txt")
DATA_IN = "./Data/Data.txt"

DATA_ENCODING = "utf-8"

RESULT_PATH = join(".", "Results")

STANDARD_OUT = join(RESULT_PATH, "networkx.txt")

BASIC_OUT = join(RESULT_PATH, "basic.txt")


# paramter for basic pagerank

TELEPORT = 0.85
 
MAX_ITER = 100

#收敛值
EPSILON = 0.0001

NORM = 1

# paramter for block, no more than CHUNK_SIZE * SIZE_INT Bytes data structure in Memory
CHUNK_SIZE = 4000

SIZE_INT = 4
