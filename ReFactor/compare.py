from config import *
from utils import compare_results, load_result

def get_baseline(length):
    return {k + 1 : 1/length for k in range(length)}

if __name__ == "__main__":
    # r1 = load_result(BASIC_OUT)
    # r2 = load_result(STRIPE_OUT)
    r1 = {}
    r2 = {}
    with open('Results\output.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            r1[key] = float(value)
    with open('D:\LessonProjects\Big-Data\Results\\basic.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split('        ')
            r2[key] = float(value)
    compare_results(r1, r2, NORM)