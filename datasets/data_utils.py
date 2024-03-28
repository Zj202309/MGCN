import os
from utils.misc import *
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import *

def load_graph_adj(path, data_path):
    data = np.loadtxt(data_path, dtype=float)
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def load_dataset(args):
    dataset_name = args.dataset
    if dataset_name in ['acm']:
        path = os.path.join(os.getcwd(), 'datasets', 'data', dataset_name)
        x_path = path + '/{}.txt'.format(dataset_name)
        y_path = path + '/{}_label.txt'.format(dataset_name)
        graph_path = path + '/{}_graph.txt'.format(dataset_name)
        x = np.loadtxt(x_path, dtype=float)
        y = np.loadtxt(y_path, dtype=int)  
        adj = load_graph_adj(graph_path, x_path)
        args.num_nodes = x.shape[0]

    return x, y, adj


