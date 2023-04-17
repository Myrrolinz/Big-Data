import numpy as np
from config import *
import os
from utils import save_result, time_test, read_data, setup
from collections import defaultdict

class Graph:
    """
    用稀疏矩阵来存储数据：src、指向src的边、出度
    此结构方便计算
    """
    def __init__(self, nnodes, edges) -> None:
        self.nnodes = nnodes # 图中节点的数量
        self.in_edges = defaultdict(list) # 存储每个节点的入边列表
        self.out_degrees = [0] * nnodes # 存储每个节点的出度，初始化为0
        
        # 添加每条边到相应的节点的入边列表中，并更新每个节点的出度
        for i, (src, dst) in enumerate(edges):
            self.out_degrees[src] += 1 # 更新起始节点的出度
            self.in_edges[dst].append(src) # 将边添加到终止节点的入边列表中

# class Graph:
#     """
#     用稀疏矩阵来存储数据：src、指向src的边、出度
#     此结构方便计算
#     """
#     def __init__(self, nnodes, edges) -> None:
#         self.nnodes = nnodes # 图中节点的数量
#         self.in_edges = [] # 存储每个节点的入边列表
#         self.out_degrees = [0] * nnodes # 存储每个节点的出度，初始化为0
        
#         # 初始化每个节点的入度列表
#         for i in range(nnodes + 1):
#             self.in_edges.append([])

#         # 添加每条边到相应的节点的入边列表中，并更新每个节点的出度
#         for edge in edges:
#             self.out_degrees[edge[0]] += 1 # 更新起始节点的出度
#             self.in_edges[edge[1]].append(edge[0]) # 将边添加到终止节点的入边列表中
            
#         """ 
#         最终Graph类be like: 
#         Node 1: in_edges = [3], out_degree = 1
#         Node 2: in_edges = [1, 4], out_degree = 1
#         Node 3: in_edges = [2], out_degree = 2
#         Node 4: in_edges = [5], out_degree = 1
#         """
        


# def read_graph() -> Graph:
#     edges, nnodes = next(read_data(DATA_IN))
#     return Graph(nnodes, edges)

def load_graph() -> Graph:
    # 读入数据
    # Read data from dataset
    links = []
    # 【注意改路径！】
    print(DATA_IN)
    with open("D:\LessonProjects\Big-Data\Data\data.txt", "r", encoding="utf-8") as file:
        for line in file:
            src, dst = map(int, line.split())
            # src, dst = line.strip().split()
            links.append((src, dst))

        # Get unique list of nodes
        max_both = -1 # 用于记录最大的节点编号
        for i in links:
            max_both = max(max_both, i[0], i[1])

        print(max_both)
        return Graph(max_both, links)
    
from scipy.sparse import lil_matrix

def pagerank(graph:Graph, d=0.85, eps=1e-8, max_iter=100):
    N = graph.nnodes
    M = lil_matrix((N, N)) # 创建一个N x N的稀疏矩阵
    for dst in range(1, N+1):
        for src in graph.in_edges[dst]:
            M[dst-1, src-1] = 1 / graph.out_degrees[src]

    # 根据PageRank算法初始化r
    r = np.full(N, 1/N)

    for i in range(max_iter):
        r_new = d * M.dot(r) + (1-d)/N # 计算新的PageRank向量
        if np.abs(r - r_new).sum() < eps: # 判断是否达到收敛
            print(f"absolute error: {np.abs(r - r_new).sum()}, iter: {i+1}")
            break
        r = r_new

    result = {}
    for i in range(1, N+1):
        result[i] = r[i-1]
    return result

# def pagerank(graph: Graph, alpha=0.85, eps=1e-8, max_iter=100):
#     N = graph.nnodes
#     r_old = np.full(N, 1 / N)
#     dangling_weights = np.zeros(N)
#     is_dangling = np.zeros(N, dtype=bool)
#     for i in range(N):
#         if graph.out_degrees[i] == 0:
#             is_dangling[i] = True
#             dangling_weights[i] = 1 / N

#     iter = 0
#     while True:
#         r_new = np.zeros(N)
#         # 计算PageRank值的主要部分
#         for dst in range(N):
#             for src in graph.in_edges[dst]:
#                 r_new[dst] += r_old[src] / graph.out_degrees[src]

#         # 添加“随机跳转”的部分
#         r_new *= alpha
#         r_new += (1 - alpha) * (dangling_weights.sum() / N + alpha * r_old[is_dangling].sum())

#         # 计算误差
#         err = np.abs(r_new - r_old).sum()
#         if err < eps or iter >= max_iter:
#             break

#         r_old = r_new
#         iter += 1

#     result = {}
#     for i in range(N):
#         result[i + 1] = r_new[i]

#     return result

    
# # PageRank算法实现
# def pagerank(graph:Graph):
#     N = graph.nnodes # 获取节点数目
#     r_old = np.full(N + 1, 1 / N) # 初始化 PageRank 向量
#     # 在 PageRank 算法中，网页排名通常是从 1 开始计数，而不是从 0 开始。
#     # 因此，对于代码中的r_old数组，r_old[0]的值需要设置为0，以避免计算的结果出现偏移。
#     # 具体来说，r_old[0]是不参与 PageRank 计算的，但需要在后续输出结果时被排除在外。
#     r_old[0] = 0
#     iter = 0 # 迭代计数器
#     while True:
#         r_new = np.zeros(N + 1) # 初始化新的 PageRank 向量
#         for dst in range(1, N + 1):
#             # 遍历连接到该节点的所有节点
#             for src in graph.in_edges[dst]:
#                 # 根据 PageRank 算法计算每个节点的 PageRank 值
#                 r_new[dst] += r_old[src] / graph.out_degrees[src]

#         # 根据公式更新新的 PageRank 向量
#         r_new *= TELEPORT
#         r_new += (1 - np.sum(r_new)) / N # 表示“随机跳转”部分的 PageRank 值。
#         # 计算当前迭代的误差
#         # TODO:范式、收敛值e可改 看能否提高准确度
#         e = np.linalg.norm(r_new - r_old, ord=NORM)
#         iter += 1
#         # 如果误差小于阈值，或者达到了最大迭代次数，就结束迭代
#         if e < EPSILON or iter >= MAX_ITER:
#             print(f"absolute error: {e}, iter: {iter}")
#             break
#         r_old = np.copy(r_new)

#     result = {}
#     for i in range(1, N + 1):
#         result[i] = r_new[i]
#     return result


if __name__ == '__main__':
    print("This is Basic Page Rank")
    
    if setup() == 0:
        graph = time_test("read_graph", load_graph)

        result = time_test("pagerank", pagerank, graph)

        save_result(result, BASIC_OUT)

        os.system("pause")
