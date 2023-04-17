import sys
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
from utils import time_test, save_result, read_data
from config import *

id_pagerank_DICT_PATH = "D:\LessonProjects\Big-Data\Data\data.txt"
PRINT_NUM = 100
PGRANK_ADJUST_PARAM = 1e4


def gen_pagerank():
    f = open(id_pagerank_DICT_PATH, 'rb')
    p = f.readlines()

    G = nx.DiGraph()
    for i in range(0, len(p)):
        nodes_split = str(p[i], "utf-8").split()
        fm = nodes_split[0]
        to = nodes_split[1]
        G.add_edge(fm, to)
    print("add finish..")

    pr = nx.pagerank(G, alpha=0.85)
    # pr = pagerank(G, alpha=0.85)
    prdic = {}
    for node, pageRankValue in pr.items():
        prdic[node] = pageRankValue
    print("calc pagerank finish..")

    data_list = [{k: v} for k, v in prdic.items()]
    f = lambda x: list(x.values())[0]
    sorted_dic = sorted(data_list, key=f, reverse=True)
    # print(sorted_dic)

    # for i in range(0, PRINT_NUM):
    #     print(sorted_dic[i])
    sorted_dic=sorted_dic[:100]
    with open('output.txt', 'a') as f:
        for d in sorted_dic:
            for k, v in d.items():
                f.write(f'{k}\t{v}\n')

if __name__ == "__main__":
    gen_pagerank()