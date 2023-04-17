from os.path import join

# input and out path

DATA_PATH = join("..", "Data")

# DATA_IN = join(DATA_PATH, "data.txt")
DATA_IN = "D:\LessonProjects\Big-Data\Data\data.txt"

DATA_ENCODING = "utf-16"

RESULT_PATH = join("..", "Results")

MIDDLE_PATH = "Middle"

STANDARD_OUT = join(RESULT_PATH, "networkx.txt")

BASIC_OUT = join(RESULT_PATH, "basic.txt")

STRIPE_OUT = join(RESULT_PATH, "stripes.txt")

FU_STRIPE_OUT = join(RESULT_PATH, "fu-stripes.txt")

SGRAPH_PATH = join(MIDDLE_PATH, "sgraphs")

BLOCK_PATH = join(MIDDLE_PATH, "blocks") 

RANK_PATH = join(MIDDLE_PATH, "ranks")




# paramter for basic pagerank

TELEPORT = 0.85
 
MAX_ITER = 100

EPSILON = 1e-6

NORM = 1

# paramter for block, no more than CHUNK_SIZE * SIZE_INT Bytes data structure in Memory
CHUNK_SIZE = 4000

SIZE_INT = 4
