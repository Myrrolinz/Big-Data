import numpy as np
from config import *
import os
from utils import save_result, time_test, read_data, setup


class Graph:
    """
    Adjacency List
    node begins from 1
    """
    def __init__(self, nnodes, edges) -> None:
        self.nnodes = nnodes
        self.in_edges = []
        # self.out_edges = []
        self.out_degrees = [0] * nnodes
        for i in range(nnodes + 1):
            self.in_edges.append([])
            # self.out_edges.append([])

        # self.starters = []
        for edge in edges:
            # self.out_edges[edge[0]].append(edge[1])
            self.out_degrees[edge[0]] += 1
            self.in_edges[edge[1]].append(edge[0])

        # for i, e in enumerate(self.in_edges):
        #     if len(e) == 0:
        #         self.starters.append(i)
        # for edge in self.out_edges:
        #     edge.sort()
        


def read_graph() -> Graph:
    edges, nnodes = next(read_data(DATA_IN))
    return Graph(nnodes, edges)


def pagerank(graph:Graph):
        N = graph.nnodes
        r_old = np.full(N + 1, 1 / N)
        r_old[0] = 0
        iter = 0
        while True:
            r_new = np.zeros(N + 1)
            for dst in range(1, N + 1):
                for src in graph.in_edges[dst]:
                    # r_new[dst] += r_old[src] / len(graph.out_edges[src])
                    r_new[dst] += r_old[src] / graph.out_degrees[src]

            # no need to do this
            # for starter in graph.starters:
            #     r_new[starter] = 0

            r_new *= TELEPORT
            r_new += (1 - np.sum(r_new)) / N
            e = np.linalg.norm(r_new - r_old, ord=NORM)
            iter += 1
            # print(f"absolute error: {e}, iter: {iter}")
            if e < EPSILON or iter >= MAX_ITER:
                print(f"absolute error: {e}, iter: {iter}")
                break
            r_old = np.copy(r_new)

        result = {}
        for i in range(1, N + 1):
            result[i] = r_new[i]
        return result


if __name__ == '__main__':
    print("This is Basic Page Rank")
    
    if setup() == 0:
        graph = time_test("read_graph", read_graph)

        result = time_test("pagerank", pagerank, graph)

        save_result(result, BASIC_OUT)

        os.system("pause")
