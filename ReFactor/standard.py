import networkx as nx
from utils import time_test, save_result, read_data
from config import *

def pagerank_with_nx():
    G = nx.Graph()
    edges, npoints = next(read_data(DATA_IN))
    G.add_nodes_from(range(1, npoints + 1))
    G.add_edges_from(edges)
    print(G)
    pr = time_test("pagerank", nx.pagerank, G, alpha=TELEPORT, tol=EPSILON, max_iter=MAX_ITER)
    save_result(pr, STANDARD_OUT)

if __name__ == "__main__":
    pagerank_with_nx()


            

