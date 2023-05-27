import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

import articulate as art
import config as conf

from model.t_graph import Graph_B, Graph_J, Graph_P, Graph_A, Unpool

class s_gcn(nn.Module):
    r'''
        用于输入数据x的gcn
        输入：动态信息x:[n, d(in_channels), t, v]； 邻接矩阵A:[k, v, v(w)]
    '''
    def __init__(self, in_channels, out_channels, k_num):
        super().__init__()

        self.k_num = k_num
        self.lin = nn.Linear(in_channels, out_channels*(k_num))

    def forward(self, x, A_skl):        # x:[n, d(in_channels), t, v]; A:[k, v, v(w)]
        x = x.permute(0,2,3,1)  #[n,t,v,d]
        x = self.lin(x)
        x = x.permute(0,3,1,2)
        
        n, kc, t, v = x.size()                                             # n = 64(batchsize), kc = 128, t = 49, v = 21
        x = x.view(n, self.k_num,  kc//(self.k_num), t, v)             # [64, 4, 32, 49, 21]
        A_all = A_skl
        x = torch.einsum('nkctv, kvw->nctw', (x, A_all))    # [n,c,t,v]
        
        return x

class AGGRU(nn.Module):
    r'''
        GCN+GRU网络，输入图时序信息，输出预测结果
    '''
    
    def __init__(self, n_in_dec, n_hid_dec, n_out_dec, strategy='uniform', edge_weighting=True):
        super().__init__()

        self.graphA = Graph_A(strategy=strategy)
        self.graphJ = Graph_J(strategy=strategy)
        graph_a = torch.tensor(self.graphA.A_a, dtype=torch.float32, requires_grad=False)
        graph_j = torch.tensor(self.graphJ.A_j, dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('graph_pos', graph_a)   # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        self.register_buffer('graph_imu', graph_j)   # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        
        k_num_pos, j_24 = self.graph_pos.size(0), self.graph_pos.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        k_num_imu, j_15 = self.graph_imu.size(0), self.graph_imu.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        if edge_weighting:
            self.emul_in = nn.Parameter(torch.ones(self.graph_pos.size()))   # [k_num, j_num, j_num]
            self.eadd_in = nn.Parameter(torch.ones(self.graph_pos.size()))   # [k_num, j_num, j_num]
            self.emul_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
            self.eadd_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
        else:
            self.emul_in = 1
            self.eadd_in = nn.Parameter(torch.ones(self.A_graph_in.size()))
            self.emul_out = 1
            self.eadd_out = nn.Parameter(torch.ones(self.A_graph_out.size()))
            
        self.pos_gcn = s_gcn(3, 3, k_num_pos)     # 针对位移的gcn
        self.imu_gcn = s_gcn(12, 12, k_num_imu)
        
        self.in_fc = torch.nn.Linear(n_in_dec, n_hid_dec)
        self.in_dropout = nn.Dropout(0.2)

        self.gru = nn.GRU(n_hid_dec, n_hid_dec, num_layers=2, bidirectional=True, batch_first=True)        
        self.out_fc = nn.Linear(2 * n_hid_dec, n_hid_dec)
        self.out_reg = nn.Linear(n_hid_dec, n_out_dec)
        

    def forward(self, x, hidden=None):                     
        n, t, d = x.size()
        pos = x[:,:,:24*3].view(n,t,24,3)  #[n,t,24*3]
        pos = pos.permute(0,3,1,2)                      # [n,3,t,24]
        imu = x[:,:,24*3:]  #[n,t,3*15+9*15] 需要填充ori和acc
        acc = imu[:,:,:16*3].view(n,t,16,3)
        ori = imu[:,:,16*3:].view(n,t,16,9)
        imu_data = torch.concat((acc,ori), dim=-1)
        imu_data = imu_data.permute(0,3,1,2)            # [n,12,t,15]
        
        pos_res = pos + self.pos_gcn(pos, self.graph_pos * self.emul_in + self.eadd_in)
        imu_res = imu_data + self.imu_gcn(imu_data, self.graph_imu * self.emul_out + self.eadd_out)
        pos_res = pos_res.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        imu_res = imu_res.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        
        input = torch.concat((pos_res, imu_res), dim=-1)
        input = self.in_dropout(input)
        input = relu(self.in_fc(input))
        
        result, _ = self.gru(input, hidden) 
        result = input + self.out_fc(result)
        
        output = self.out_reg(result)
        
        return output

