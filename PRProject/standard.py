import numpy as np
import networkx as nx
import os

RESULT_FOLDER = './Results/'
filename=RESULT_FOLDER+"standard.txt"
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load data from file
data = np.loadtxt('./Data/Data.txt', dtype=int,encoding="utf-8")

# Build graph from edges
G = nx.DiGraph()
for row in data:
    G.add_edge(row[0], row[1])

# Calculate PageRank scores for nodes
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)

# Sort nodes by PageRank score and print top 100
sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
if len(sorted_nodes) == 0:
    print("Error: There are no nodes in the graph.")
else:
    '''
    display in consle
    '''
    for k in range(min(100, len(sorted_nodes))):
        node_id, pr_score = sorted_nodes[k]
        print("NodeID: {} PageRank: {: }".format(node_id, pr_score))
        sorted_dic = [node_id, pr_score]

    topn = 100
    if topn > 0:
        sorted_nodes = sorted_nodes[:topn]
        # 一次性 写出
        with open(filename, 'w', encoding='utf-8') as f:
            for line in sorted_nodes:
                f.write(f"{line[0]:<10} {line[1]}\n")

        print("finished")