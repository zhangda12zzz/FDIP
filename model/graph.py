import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 定义图结构基类 - A 类型图（针对24节点的关节连接）
class Graph_A():
    def __init__(self, layout='lstm', strategy='uniform', max_hop=1, dilation=1):
        """

        Parameters
        ----------
        layout    数据输入的格式，'lstm'表示输入为24个节点的LSTM输出，'ntu'表示输入为16个节点的NTU-RGBD数据
        strategy  图的连接策略，'uniform'表示所有节点间都有边，'distance'表示根据距离远近有不同的权重，'spatial'表示根据空间位置有不同的权重
        max_hop   图的最大跳数
        dilation  图的膨胀率
        """
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A_a

    def get_edge(self, layout):
        """
        根据布局定义图边和节点的连接关系
        Parameters
        ----------
        layout

        Returns
        -------

        """
        if layout == 'lstm':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_ = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
                              (7,10),(8,11),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),
                              (16,18),(18,20),(20,22),(17,19),(19,21),(21,23)]
            neighbor_link = [(i,j) for (i,j) in neighbor_link_]
            self.edge = self_link + neighbor_link
            self.center = 1-1
            self.upLevelList = [0,1,2,5,6,9,10,11,12,14,15,16,19,20,21]

    def get_adjacency(self, strategy):
        """
        根据连接策略定义图的邻接矩阵
        Parameters
        ----------
        strategy

        Returns
        -------

        """

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A_a = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A_a = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A_a = A

# 16节点的图结构类
class Graph_J():
    def __init__(self, layout='lstm', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A_j

    def get_edge(self, layout):
        if layout == 'lstm':
            self.num_node = 16
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_ = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(6,7),(7,8),(7,9),
                              (7,10),(8,11),(9,12),(10,13),(12,14),(13,15)]
            neighbor_link = [(i,j) for (i,j) in neighbor_link_]
            self.edge = self_link + neighbor_link
            self.center = 1-1
            self.upLevelList = [0,2,4,6,7,9,11,12,14,15]

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A_j = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A_j = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A_j = A


# 10节点的图结构类
class Graph_P():
    def __init__(self, layout='lstm', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A_p

    def get_edge(self, layout):
        if layout == 'lstm':
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_ = [(1,2),(1,3),(1,4),(4,5),(5,6),(5,7),(7,8),(5,9),(9,10)]
            neighbor_link = [(i-1,j-1) for (i,j) in neighbor_link_]
            self.edge = self_link + neighbor_link
            self.center = 1-1
            self.upLevelList = [0,1,2,5,7,9]

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A_p = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A_p = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A_p = A

# 6节点的图结构类
class Graph_B():
    def __init__(self, layout='lstm', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A_b

    def get_edge(self, layout):
        if layout == 'lstm':
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_ = [(1,2), (1,3), (1,4), (1,5), (1,6)]
            neighbor_link = [(i-1,j-1) for (i,j) in neighbor_link_]
            self.edge = self_link + neighbor_link
            self.center = 1-1
            self.upLevelList = []

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A_b = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A_b = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A_b = A
            
def get_hop_distance(num_node, edge, max_hop=1):
    """
    计算图节点的最短距离
    Parameters
    ----------
    num_node
    edge
    max_hop

    Returns   hop_dis
    -------

    """
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    """
    对有向图进行归一化处理（行标准化）
    Parameters
    ----------
    A    ：邻接矩阵

    Returns  ：    归一化之后的邻接矩阵
    -------

    """
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

class Unpool(nn.Module):
    """
        图上采样层（用于图卷积的反向操作）
    Args:
        after_graph: 上采样后的图结构
        before_graph: 上采样前的图结构
        idx: 节点映射索引
    """
    def __init__(self, after_graph, before_graph, idx):
        super(Unpool, self).__init__()
        self.graph_after_unpool = after_graph
        self.graph_before_unpool = before_graph
        self.before_in_after_idx = idx

    def forward(self, feature):
        new_h = feature.new_zeros([self.graph_after_unpool.shape[1], feature.shape[1], feature.shape[2], feature.shape[3]])
        new_h[self.before_in_after_idx] = feature
        return new_h

if __name__ == '__main__':
    graph_b = Graph_B()
    A_b = torch.tensor(graph_b.A_b, dtype=torch.float32, requires_grad=False)
    print(A_b)
