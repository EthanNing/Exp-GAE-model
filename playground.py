# [[236 122  24 ...  85  98 291]
#  [186 285 346 ...  75 332 339]]
import nocd
import numpy as np
import math
import math
import torch.distributions as td
import torch


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def edge_prob(e1, e2):
    num_edges = 5038
    num_nodes = 348
    logits = emb[e1 - 1].dot(emb[e2 - 1]) - np.log(1 - num_edges / (num_nodes ** 2 - num_nodes))
    probs = 1 - math.exp(-logits)
    return probs


def find_comm(node):
    comm_list = list()
    for i in range(len(emb[node])):
        if emb[node][i] != 0.:
            comm_list.append(i)
    print(comm_list)


emb = np.load("z.npy")
print(emb.shape)
adj = np.empty([347, 347], dtype=float)

"""
for i in range(347):
    for j in range(347):
        adj[i][j] = edge_prob(i, j)

print(adj[121][284])
find_comm(98)
find_comm(284)

"""


