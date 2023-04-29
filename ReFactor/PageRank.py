import numpy as np
import copy
from collections import defaultdict
from config import *
from utils import save_result, time_test, read_data, setup
import os

class Graph:
    """
    Using lil matrix: src, in-edges, out-degree
    """
    def __init__(self, nnodes, edges) -> None:
        self.nnodes = nnodes  # number of nodes
        self.in_edges = defaultdict(list) # in-edge list
        self.out_degrees = [0] * (nnodes + 1)
        for i, (src, dst) in enumerate(edges):
            self.out_degrees[src] += 1 # update out-degree
            self.in_edges[dst].append(src) # add in-edge

def load_graph() -> Graph:
    # Read data from dataset
    links = []
    with open("Data\data.txt", "r", encoding="utf-8") as file:
        for line in file:
            src, dst = map(int, line.split())
            links.append((src, dst))
        
        # Get unique list of nodes
        nodes=[]
        for row in links:
            if row not in nodes:
                nodes.append(row)
        print("num of total links: {}".format(len(links)))
        print("num of unique links: {}".format(len(nodes)))

        # Get unique list of nodes
        max_both = -1 # max number of nodes (maybe != total number of nodes)
        for i in links:
            max_both = max(max_both, i[0], i[1])

        print("max node number: {}".format(max_both))
        return Graph(max_both, nodes)


def pagerank2(graph: Graph):
    N = graph.nnodes
    r_old = np.full(N + 1, 1 / N)  # 初始化向量
    # print(r_old)
    r_old[0] = 0
    iter = 0
    i=0
    while True:
        r_new = np.zeros(N + 1)
        for dst in range(1, N + 1):  # dst只是个变量哈：表示从结点1：N+1
            for src in graph.in_edges[dst]:  # 指向dst的结点
                r_new[dst] += r_old[src] / graph.out_degrees[src]

        r_new *= TELEPORT
        s = float(np.sum(r_new))
        r_new += (1 - s) / N  # Now re-insert the leaked PageRank：修正
        #TODO:范式、收敛值e可改 看能否提高准确度
        e = np.linalg.norm(r_new - r_old, ord=NORM)  # 第一范式距离表示收敛
        iter += 1
        if e < EPSILON or iter >= MAX_ITER:
            print(f"absolute error: {e}, iter: {iter}")
            break
        r_old = np.copy(r_new)

    result = {}
    for i in range(1, N + 1):
        result[i] = r_new[i]

    return result


def pagerank2_revise(graph: Graph):
    # Preprocessing
    # While calculating N, in-degree = 0 is not considered.
    # However, the lil matrix is still the original graph.
    i = 0
    count = 0
    is_isolate = np.full(graph.nnodes+1, True)
    while i < (graph.nnodes+1):
        if not (graph.out_degrees[i] == 0 and \
                len(graph.in_edges[i])==0): # in- or out-degree = 0
            count += 1
            is_isolate[i] = False
        i += 1
    N = count
    print("number of unique nodes:{} ".format(count))

    # Initializing matrix
    r_old = np.full(graph.nnodes+1, np.float64(1 / N))
    # Note: isolate node is not considered while calculating r_old and r_new
    r_old[is_isolate] = 0

    iter = 0
    while True:
        # Note: idx=0 is useless, we set it for convinience
        r_new = np.zeros(graph.nnodes + 1, dtype=np.float64)
        for dst in range(graph.nnodes+ 1):
            if len(graph.in_edges[dst]) == 0:  # in-degree = 0
                r_new[dst] = 0
            else:
                for src in graph.in_edges[dst]:
                    # $$ r*{new}_j = \sum_{i\rightarrow j}\beta * 
                    # frac{r*{old}_i}{d_i} $$
                    # Note: \beta = TELEPORT. TELEPORT = 0.85
                    r_new[dst] += (r_old[src] \
                        / graph.out_degrees[src])*TELEPORT
        s = np.float64(np.sum(r_new))
        print("Iter: {}, Before adjusting, S value: {:.17f}".format(iter, s))
        r_new += (1 - s) / N # Now re-insert the leaked PageRank
        r_new[is_isolate] = 0
        s = np.float64(np.sum(r_new))
        print("Iter: {}, After adjusting, S value: {:.17f}".format(iter, s))
        
        iter += 1

        # Find max error
        error = 0.
        for i in range(graph.nnodes + 1):
            if(error < abs(r_new[i] - r_old[i])):
                error = abs(r_new[i] - r_old[i])

        if (error < EPSILON or iter >= MAX_ITER):
            print(f"absolute error: {error}, iter: {iter}")
            break
        r_old = copy.deepcopy(r_new)

    result = {}
    for i in range(1, graph.nnodes+1):
        result[i] = r_new[i]
        
    return result


if __name__ == '__main__':
    print("This is Basic Page Rank")

    if setup() == 0:
        graph = time_test("read_graph", load_graph)
        result = time_test("pagerank", pagerank2_revise, graph)
        save_result(result, BASIC_OUT)
        os.system("pause")