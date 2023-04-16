from cgi import test
from basic import Graph, load_graph
from stripes import SubGraph, load_outds_by_range, load_ginfo
from utils import read_data
from config import *

def merge_dict(dict1:dict, dict2:dict):
    l1 = len(dict1)
    l2 = len(dict2)
    dict1.update(dict2)
    assert(len(dict1) == l1 + l2)

def compare_dict(dict1:dict, dict2:dict):
    assert(len(dict1) == len(dict2))
    # if(len(dict1) != len(dict2)):
    #     k1 = set(dict1.keys())
    #     k2 = set(dict2.keys())
    #     for k in k1:
    #         if k not in k2:
    #             print(f"{k} in k1 not in k2")
    #     for k in k2:
    #         if k not in k1:
    #             print(f"{k} in k2 not in k1")
    for k in dict1.keys():
        assert(dict1[k] == dict2[k])

def test_sgraphs():
    for i, data in enumerate(read_data(DATA_IN, CHUNK_SIZE)):
        edges, _ = data
        sg1 = SubGraph()
        sg1.add_edges(edges)
        sg2 = SubGraph()
        sg2.load(i)
        # assert(len(sg1.in_edges) == len(sg2.in_edges))
        # assert(len(sg1.out_degrees) == len(sg2.out_degrees))
        # for k in sg1.in_edges.keys():
        #     assert(sg1.in_edges[k] == sg2.in_edges[k])
        # for k in sg1.out_degrees.keys():
        #     assert(sg1.out_degrees[k] == sg2.out_degrees[k])
        compare_dict(sg1.in_edges, sg2.in_edges)
        compare_dict(sg1.out_degrees, sg2.out_degrees)
        print(f"sgraph_{i} is correct")
            
def test_outd():
    """
    test whether outd.data is correct.
    Case cannot be found: outd.data is longer.
    """
    graph = load_graph()
    outds = load_outds_by_range(1, graph.nnodes)
    n = 0
    for i, p in enumerate(list(zip(graph.out_degrees[1:], outds))):  # graph' saving begins from 0
        d1, d2 = p
        if d1 != d2:
            print(f"Node {i + 1} out degree dismatch! expect {d1} but get {d2}")
            n += 1
            if n > 100:
                break
    assert(n == 0)
    print("outd.data is correct")


def test_blocks():
    """
    test whether block_xx.data is correct.
    """
    ginfo = load_ginfo()
    nblock = ginfo["nblock"]
    sgraph = SubGraph()
    in_edges = {}
    for i in range(nblock):
        sgraph.load_from_block(i)
        merge_dict(in_edges, sgraph.in_edges)
        dsts = list(sgraph.in_edges.keys())
        print(f"block_{i} has dsts from {dsts[0]} to {dsts[-1]} (not successive)")
    graph = load_graph()
    in_edges_c = {}
    for i, d in enumerate(graph.in_edges[1:]):
        if len(d) > 0:
            in_edges_c[i + 1] = sorted(d)
    outds_c = {}
    for i, d in enumerate(graph.out_degrees[1:]):
        outds_c[i + 1] = d
    compare_dict(in_edges, in_edges_c)
    print("blocks_xx.data are correct.")



if __name__ == "__main__":
    # test_sgraphs()  # ok
    # test_outd()  # ok
    test_blocks()  # ok
    



    