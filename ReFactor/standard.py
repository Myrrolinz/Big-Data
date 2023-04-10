import networkx as nx
from utils import time_test, save_result, read_data
from config import *

def pagerank_with_nx():  # 定义 PageRank 函数
    G = nx.Graph()  # 初始化一个空的无向图
    edges, npoints = next(read_data(DATA_IN))  # 读取数据文件
    G.add_nodes_from(range(1, npoints + 1))  # 添加节点
    G.add_edges_from(edges)  # 添加边
    print(G)  # 输出图形结构
    pr = time_test("pagerank", nx.pagerank, G, alpha=TELEPORT, tol=EPSILON, max_iter=MAX_ITER)  # 计算 PageRank
    save_result(pr, STANDARD_OUT)  # 将结果保存到文件中

if __name__ == "__main__":
    pagerank_with_nx()


            

