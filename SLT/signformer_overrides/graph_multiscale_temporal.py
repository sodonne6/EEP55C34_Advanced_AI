#import tensorflow as tf
import torch
import numpy as np
from IPython.core.debugger import set_trace


class Graph():
    def __init__(self, num_node):
        if num_node != 33:
            raise ValueError(f"Graph only defines the 33-node MediaPipe pose topology, got {num_node}")
        self.num_node = num_node
        self.AD, self.AD2, self.bias_mat_1, self.bias_mat_2 = self.normalize_adjacency()

    def normalize_adjacency(self):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_1base = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)]
        neighbor_link = neighbor_1base
        edge = self_link + neighbor_link
        A = np.zeros((self.num_node, self.num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        A2 = np.zeros((self.num_node, self.num_node))
        for root in range(A.shape[1]):
            for neighbour in range(A.shape[0]):
                if A[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A.shape[0]):
                        if A[neighbour, neighbour_of_neigbour] == 1:
                            A2[root, neighbour_of_neigbour] = 1

        bias_mat_1 = np.zeros(A.shape)
        bias_mat_2 = np.zeros(A2.shape)
        bias_mat_1 = np.where(A != 0, bias_mat_1, -1e9)
        bias_mat_2 = np.where(A2 != 0, A2, -1e9)
        
        bias_mat_1 = bias_mat_1.astype('float32')
        bias_mat_2 = bias_mat_2.astype('float32')
        
        AD = self.normalize(A).float()
        AD2 = self.normalize(A2).float()
        
        return AD, AD2, bias_mat_1, bias_mat_2

    def normalize(self, adjacency):
        rowsum = np.array(adjacency.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = np.diag(r_inv)
        normalize_adj = r_mat_inv.dot(adjacency)
        normalize_adj = normalize_adj.astype('float32')
        normalize_adj = torch.tensor(normalize_adj)
        return normalize_adj
