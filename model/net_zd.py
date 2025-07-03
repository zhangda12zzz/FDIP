import os
import sys

from model.DFTFPE import DSTFPE
from model.MSFKE import NodeAwareMSFKE
import torch


@torch.no_grad()
def predictPose_single(self, acc, ori, preprocess=False):
    r'''
        acc: [t,6,3]
        ori: [t,6,3,3]
        顺序为：根、左右脚、头、左右手（有预处理） /  右手左手、右脚左脚、头、根（无预处理）
    '''
    if not preprocess:
        order = [2, 3, 4, 0, 1, 5]
        acc_cal = acc[:, order]
        ori_cal = ori[:, order]

        acc_tmp = torch.cat((acc_cal[:, 5:], acc_cal[:, :5] - acc_cal[:, 5:]), dim=1).bmm(
            ori_cal[:, -1])  # / conf.acc_scale
        ori_tmp = torch.cat((ori_cal[:, 5:], ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5])), dim=1)
    else:
        acc_tmp = acc
        ori_tmp = ori

    t = acc_tmp.shape[0]
    root = ori_tmp[:, 0].view(t, 3, 3)

    acc_tmp = acc_tmp.view(t, -1)  # [t,18]
    ori_tmp = ori_tmp.view(t, -1)  # [t,54]
    imu = torch.cat((acc_tmp, ori_tmp), dim=-1).unsqueeze(0)  # [1,t,72]

    leaf_pos, all_pos, glo_pose = self.forward(imu)
    # pose = self._reduced_glb_euler_to_full_local_mat(root, glo_pose)     # euler版本
    # pose = self._reduced_glb_axis_to_full_local_mat(root, glo_pose)     # axis版本
    pose = self._reduced_glb_6d_to_full_local_mat(root, glo_pose)  # r6d版本
    # pose = self._reduced_glb_mat_to_full_local_mat(root, glo_pose)    # 矩阵版本
    return pose.squeeze(0)

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

import articulate as art
import config as conf
from model.graph import Graph_B, Graph_J, Graph_P, Graph_A, Unpool


# 定义欧拉角到旋转矩阵的转换函数
def euler2mat(euler):
    r'''
        euler: [n,t,v,3] => return[n,t,v,3,3]
    '''
    n, t, _ = euler.shape
    euler = euler.view(n, t, 15, 3)
    cos = torch.cos(euler)  # [n,t,v,3]
    sin = torch.sin(euler)  # [n,t,v,3]
    mat = cos.new_zeros(n, t, 15, 9)

    mat[:, :, :, 0] = cos[:, :, :, 1] * cos[:, :, :, 2]
    mat[:, :, :, 1] = -cos[:, :, :, 1] * sin[:, :, :, 2]
    mat[:, :, :, 2] = sin[:, :, :, 1]
    mat[:, :, :, 3] = cos[:, :, :, 0] * sin[:, :, :, 2] + cos[:, :, :, 2] * sin[:, :, :, 0] * sin[:, :, :, 1]
    mat[:, :, :, 4] = cos[:, :, :, 0] * cos[:, :, :, 2] - sin[:, :, :, 0] * sin[:, :, :, 1] * sin[:, :, :, 2]
    mat[:, :, :, 5] = -cos[:, :, :, 1] * sin[:, :, :, 0]
    mat[:, :, :, 6] = sin[:, :, :, 0] * sin[:, :, :, 2] - cos[:, :, :, 0] * cos[:, :, :, 2] * sin[:, :, :, 1]
    mat[:, :, :, 7] = cos[:, :, :, 2] * sin[:, :, :, 0] + cos[:, :, :, 0] * sin[:, :, :, 1] * sin[:, :, :, 2]
    mat[:, :, :, 8] = cos[:, :, :, 0] * cos[:, :, :, 1]

    return mat.contiguous()

"""
没用到
"""
class s_gcn(nn.Module):
    r'''
        用于输入数据x的gcn
        输入：动态信息x:[n, d(in_channels), t, v]； 邻接矩阵A:[k, v, v(w)]
    '''

    def __init__(self, in_channels, out_channels, k_num):
        super().__init__()

        self.k_num = k_num  # 多个邻接矩阵个数/卷积核个数
        self.lin = nn.Linear(in_channels, out_channels * (k_num))

    def forward(self, x, A_skl):  # x:[n, d(in_channels), t, v]; A:[k, v, v(w)]
        x = x.permute(0, 2, 3, 1)  # [n,t,v,d]
        x = self.lin(x)
        x = x.permute(0, 3, 1, 2)

        n, kc, t, v = x.size()  # n = 64(batchsize), kc = 128, t = 49, v = 21
        x = x.view(n, self.k_num, kc // (self.k_num), t, v)  # [64, 4, 32, 49, 21]
        A_all = A_skl
        x = torch.einsum('nkctv, kvw->nctw', (x, A_all))  # 对每个邻接矩阵实现卷积操作-[n,c,t,v]

        return x


class FDIP_1(nn.Module):
    def __init__(self, input_dim=54, output_dim=15):
        """
        FDIP_1: 叶关节位置回归网络

        参数:
        - input_dim: IMU输入维度(54 = 6IMU × (3维加速度 + 6维旋转))
        - output_dim: 输出维度(15 = 5个叶关节 × 3维坐标)
        """
        super().__init__()

        # 特征维度设置6*9*2
        feature_dim = 108
        num_nodes = 6

        self.msfke = NodeAwareMSFKE(
            input_channels=input_dim//num_nodes,  # 每个节点的输入特征维度(54/6=9)
            num_nodes=num_nodes,  # 节点数量(共6个节点)
            base_channels=128,  # 与原始模型相同
            num_scales=5,  # 频率尺度数量
            node_feature_dim=18,  # 每个节点的输出特征维度(108/6=18)
            stage='early',
        )

        # 双流时空融合姿态估计器
        self.dstfpe = DSTFPE(
            trunk_dim=feature_dim,
            num_nodes=num_nodes,  # 节点数量(共6个节点)
            limb_dim=feature_dim,
            hidden_dim=256,
            output_dim=output_dim,
            num_heads=8,
            stage='early',
        )

    def forward(self, x):
        """
        输入: [batch, seq_len, input_dim] IMU数据
        输出: [batch, seq_len, output_dim] 叶关节位置
        """
        # 多尺度频域特征提取
        msfke_out = self.msfke(x)
        trunk_features = msfke_out['trunk_features']
        limb_features = msfke_out['limb_features_combined']

        # 双流时空融合姿态估计
        joint_positions = self.dstfpe(trunk_features, limb_features)
        return joint_positions


class FDIP_2(nn.Module):
    def __init__(self, input_dim=72, output_dim=72):
        """
        FDIP_2: 全关节位置回归网络

        参数:
        - input_dim: IMU输入维度(18+54 = 6IMU × (6+3+3)    残差连接后的
        - output_dim: 输出维度(72 = 24个全关节 × 3维坐标)
        """
        super().__init__()

        # 特征维度设置6*9*2
        feature_dim = 144
        num_nodes = 6

        # 多尺度频域运动学编码器
        self.msfke = NodeAwareMSFKE(
            input_channels=input_dim//num_nodes,  # 每个节点的输入特征维度(18/6=3)
            num_nodes=num_nodes,  # 节点数量(共6个节点)
            node_feature_dim=feature_dim//num_nodes,
            base_channels=128,  # 每个频段的通道数
            num_scales=5,  # 频率尺度数量
            stage='mid',
        )

        # 双流时空融合姿态估计器
        self.dstfpe = DSTFPE(
            num_nodes=num_nodes,  # 节点数量(共6个节点)
            trunk_dim=feature_dim,
            limb_dim=feature_dim,
            hidden_dim=128,
            output_dim=output_dim,  #24*3位置
            num_heads=8,
            stage='mid',
        )

    def forward(self, x):
        """
        输入: [batch, seq_len, input_dim] IMU数据
        输出: [batch, seq_len, output_dim] 叶关节位置
        """
        # 多尺度频域特征提取
        msfke_out = self.msfke(x)
        trunk_features = msfke_out['trunk_features']
        limb_features = msfke_out['limb_features_combined']

        # 双流时空融合姿态估计
        joint_positions = self.dstfpe(trunk_features, limb_features)

        return joint_positions


class FDIP_3(nn.Module):
    def __init__(self, input_dim=288, output_dim=144):
        """
        FDIP_1: 叶关节位置回归网络

        参数:
        - input_dim: IMU输入维度(54 = 6IMU × (3维加速度 + 6维旋转))
        - output_dim: 输出维度(15 = 5个叶关节 × 3维坐标)
        """
        super().__init__()

        self.indices = [0,7,8,12,20,21]
        self.num_indices = len(self.indices)
        self.imu_features = 9  # 每个叶节点传感器特征数
        self.pos_features = 3  # 每个节点位置特征数
        self.num_leaf = 6  # 叶节点数量

        self.feature_dim = (self.imu_features + self.pos_features) * 2
        self.num_nodes = 24

        self.msfke = NodeAwareMSFKE(
            input_channels=input_dim//self.num_nodes,
            num_nodes=self.num_nodes,
            base_channels=128,
            num_scales=5,  # 频率尺度数量
            node_feature_dim=self.feature_dim ,
            stage='late',
        )

        # 双流时空融合姿态估计器
        self.dstfpe = DSTFPE(
            trunk_dim=self.feature_dim*self.num_nodes,
            num_nodes=self.num_nodes,
            limb_dim=self.feature_dim*self.num_nodes,
            hidden_dim=256,
            output_dim=output_dim,
            num_heads=8,
            stage='late',
        )

    def _build_sensor_matrix(self, imu_data, device):
        """智能构建传感器数据矩阵，仅叶子节点有数据"""
        batch_size, seq_len, _ = imu_data.shape

        # 创建全零的24节点传感器矩阵
        sensor_matrix = torch.zeros(
            batch_size, seq_len, self.num_nodes, self.imu_features,
            dtype=imu_data.dtype, device=device
        )

        # 将6个叶节点的传感器数据分配到正确位置
        leaf_sensor_data = imu_data.reshape(
            batch_size, seq_len, self.num_leaf, self.imu_features
        )

        # 将每个叶节点数据分配到指定位置
        for idx, leaf_idx in enumerate(self.indices):
            sensor_matrix[:, :, leaf_idx, :] = leaf_sensor_data[:, :, idx, :]

        return sensor_matrix




    def forward(self, x , body_position):
        """
        输入: [batch, seq_len, input_dim] IMU数据
        输出: [batch, seq_len, output_dim] 叶关节位置
        """
        batch_size, seq_len, _ = x.shape

        position_data = body_position.reshape(
            batch_size, seq_len, self.num_nodes, self.pos_features
        )

        # 2. 构建完整的传感器特征矩阵
        sensor_data = self._build_sensor_matrix(x, position_data.device)

        # 3. 合并节点特征: 传感器数据 + 位置数据
        combined_features = torch.cat([sensor_data, position_data], dim=-1)
        combined_features = combined_features.reshape(batch_size, seq_len, -1)

        # 多尺度频域特征提取
        msfke_out = self.msfke(combined_features)
        trunk_features = msfke_out['trunk_features']

        limb_features = msfke_out['limb_features_combined']


        # 双流时空融合姿态估计
        pose = self.dstfpe(trunk_features, limb_features)

        return pose


class FDIP_Masked(nn.Module):
    def __init__(self, input_dim=288,output_dim=144,):
        """
        最终版FDIP - 带有计算掩码:

        核心逻辑:
        1. 接收稀疏IMU和完整位置数据。
        2. 拼装成 (B, S, 24, 12) 的完整特征张量。
        3. **额外创建一个 (B, S, 24) 的 `imu_validity_mask` 张量**。
        4. 将数据和掩码一起传递给下游模块。
        """
        super().__init__()

        self.num_nodes = 24
        self.num_leaf_nodes = 6
        self.imu_features = 9
        self.pos_features = 3
        self.node_feature_dim = self.imu_features + self.pos_features

        leaf_indices = torch.tensor([0, 7, 8, 12, 20, 21], dtype=torch.long)
        self.register_buffer('leaf_indices', leaf_indices)

        # --- 使用能感知掩码的下游模块 ---
        self.msfke = MSFKE_MaskAware(
            input_channels=self.node_feature_dim,
            num_nodes=self.num_nodes,
            base_channels=128, num_scales=5, node_feature_dim=self.node_feature_dim * 2, stage='late',
        )

        self.dstfpe = DSTFPE(
            trunk_dim=self.node_feature_dim*2,
            num_nodes=self.num_nodes,
            limb_dim=self.node_feature_dim,
            output_dim=output_dim,
            hidden_dim=256, num_heads=8, stage='late',
        )

    def forward(self, x, body_position):
        """
        x (Tensor): 叶节点IMU数据 (B, S, 54)
        body_position (Tensor): 所有节点位置数据 (B, S, 72)
        """
        batch_size, seq_len, _ = body_position.shape
        device = x.device

        # 1. 准备输入数据 (与之前相同)
        leaf_imu_data = x.view(batch_size, seq_len, self.num_leaf_nodes, self.imu_features)
        pos_data = body_position.view(batch_size, seq_len, self.num_nodes, self.pos_features)

        full_imu_data = torch.zeros(
            batch_size, seq_len, self.num_nodes, self.imu_features,
            device=device, dtype=x.dtype
        )
        full_imu_data[:, :, self.leaf_indices, :] = leaf_imu_data

        combined_features = torch.cat([full_imu_data, pos_data], dim=-1)
        network_input = combined_features.view(batch_size, seq_len, -1)

        # --- 2. 核心改动：创建并传递计算掩码 ---
        # 创建一个掩码，标记哪些节点的IMU数据是有效的
        # 形状: (B, S, 24)
        imu_validity_mask = torch.zeros(batch_size, seq_len, self.num_nodes, device=device, dtype=torch.float32)
        # 将叶节点对应位置设为1
        imu_validity_mask[:, :, self.leaf_indices] = 1.0

        # 3. 将数据和掩码一同送入下游网络
        msfke_out = self.msfke(network_input, mask=imu_validity_mask)
        trunk_features = msfke_out['trunk_features']
        limb_features = msfke_out['limb_features_combined']

        pose = self.dstfpe(trunk_features, limb_features)

        return pose

"""
测试
"""
# 测试FDIP_1模型

# def test_fdip_1():
#     print("=" * 70)
#     print("FDIP_1模型测试 - 叶关节位置回归")
#     print("=" * 70)
#
#     # 创建模型
#     model = FDIP_1(input_dim=54, output_dim=15)
#
#     # 模拟数据
#     batch_size, seq_len = 2, 10
#     imu_data = torch.randn(batch_size, seq_len, 54)
#
#     # 前向传播
#     output = model(imu_data)
#
#     # 输出信息
#     print(f"输入IMU数据: {imu_data.shape}")
#     print(f"输出叶关节位置: {output.shape}")
#
#     imu_data1 = torch.randn(batch_size, seq_len, 3)
#     imu_data_cat = torch.cat([imu_data, output, imu_data1], dim=-1)
#     print(imu_data_cat.shape)
#     model1 = FDIP_2(input_dim=72, output_dim=72)
#     output1 = model1(imu_data_cat)
#
#     # # 参数统计
#     # total_params = sum(p.numel() for p in model.parameters())
#     # print(f"模型总参数量: {total_params:,}")
#     print(f"输入IMU数据: {imu_data_cat.shape}")
#     print(f"输出叶关节位置: {output1.shape}")
#
#     model2 = FDIP_3(input_dim=288, output_dim=144)
#     output2 = model2(imu_data, output1)
#
#     print(f"输入IMU数据: {imu_data_cat.shape}")
#     print(f"输出叶关节位置: {output2.shape}")
#
#
#     return model2, output2
#
#
# if __name__ == "__main__":
#     model, output = test_fdip_1()
#
# sys.exit()




class GIP_3(nn.Module):  # GAIP的rnn3，输出结果为矩阵24*6
    r'''
        GCN+GRU网络，输入图时序信息，输出预测结果
    '''

    def __init__(self, n_in_dec, n_hid_dec, n_out_dec, strategy='uniform', edge_weighting=True):
        super().__init__()

        self.graphB = Graph_B(strategy=strategy)
        graph_b = torch.tensor(self.graphB.A_b, dtype=torch.float32, requires_grad=False)
        self.register_buffer('graph_imu', graph_b)  # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        k_num_imu, j_6 = self.graph_imu.size(0), self.graph_imu.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）

        self.graphA = Graph_A(strategy=strategy)
        # self.graphJ = Graph_J(strategy=strategy)
        graph_a = torch.tensor(self.graphA.A_a, dtype=torch.float32, requires_grad=False)  # 24节点
        # graph_j = torch.tensor(self.graphJ.A_j, dtype=torch.float32, requires_grad=False)   #16节点

        self.register_buffer('graph_pos', graph_a)  # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        # self.register_buffer('graph_imu', graph_j)   # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”

        k_num_pos, j_24 = self.graph_pos.size(0), self.graph_pos.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        # k_num_imu, j_15 = self.graph_imu.size(0), self.graph_imu.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        if edge_weighting:  # 边权重，生成可用的权重参数
            self.emul_in = nn.Parameter(torch.ones(self.graph_pos.size()))  # [k_num, j_num, j_num]
            self.eadd_in = nn.Parameter(torch.ones(self.graph_pos.size()))  # [k_num, j_num, j_num]
            self.emul_out = nn.Parameter(torch.ones(self.graph_imu.size()))  # [k_num, j_num, j_num]
            self.eadd_out = nn.Parameter(torch.ones(self.graph_imu.size()))  # [k_num, j_num, j_num]
        else:
            self.emul_in = 1
            self.eadd_in = nn.Parameter(torch.ones(self.A_graph_in.size()))
            self.emul_out = 1
            self.eadd_out = nn.Parameter(torch.ones(self.A_graph_out.size()))

        self.pos_gcn = s_gcn(3, 3, k_num_pos)  # 针对位移的gcn
        self.imu_gcn = s_gcn(9, 9, k_num_imu)

        self.in_fc = torch.nn.Linear(n_in_dec, n_hid_dec)
        self.in_dropout = nn.Dropout(0.2)

        self.gru = nn.GRU(n_hid_dec, n_hid_dec, num_layers=2, bidirectional=True, batch_first=True)
        self.out_fc = nn.Linear(2 * n_hid_dec, n_hid_dec)
        self.out_reg = nn.Linear(n_hid_dec, n_out_dec)

        # 添加软约束层
        num_joints = n_out_dec // 6
        self.soft_limits = LearnableSoftLimits6DLayer(num_joints=num_joints)

    def forward(self, x, hidden=None):
        n, t, d = x.size()
        pos = x[:, :, 6 * 9:].view(n, t, 24, 3)  # [n,t,24*3]
        pos = pos.permute(0, 3, 1, 2)  # [n,3,t,24]
        imu = x[:, :, :6 * 9].view(n, t, 6, 9)
        acc = imu[:, :, :, :3].view(n, t, 6, 3)
        ori = imu[:, :, :, 3:].view(n, t, 6, 6)
        imu_data = imu.permute(0, 3, 1, 2)

        # 使用s-GC模块
        pos_res = pos + self.pos_gcn(pos, self.graph_pos * self.emul_in + self.eadd_in)
        imu_res = imu_data + self.imu_gcn(imu_data, self.graph_imu * self.emul_out + self.eadd_out)
        pos_res = pos_res.permute(0, 2, 1, 3).contiguous().view(n, t, -1)
        imu_res = imu_res.permute(0, 2, 1, 3).contiguous().view(n, t, -1)
        # 对比消融实验：没有sGC模块
        # pos_res = pos.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        # imu_res = imu_data.permute(0, 2, 1, 3).contiguous().view(n,t,-1)

        input = torch.cat((imu_res, pos_res), dim=-1)
        input = self.in_dropout(input)
        input = relu(self.in_fc(input))

        result, _ = self.gru(input, hidden)
        result = input + self.out_fc(result)

        output = self.out_reg(result)

        output = self.soft_limits(output)

        return output


class GGIP(nn.Module):
    def __init__(self, n_hid_dec=256, strategy='uniform', edge_weighting=True):
        super().__init__()
        self.name = 'GGIP'

        self.gip1 = FDIP_1(6 * 12, n_hid_dec, 5 * 3)
        self.gip2 = FDIP_2(6 * 15, n_hid_dec, 23 * 3)
        self.gip3 = FDIP_3(24 * 3 + 16 * 12, n_hid_dec, 15 * 6, strategy=strategy)  # uniform / spatial
        # self.gip3 = AGGRU_3(24*3+16*12, n_hid_dec, 15*6)
        # self.gip3 = AGGRU_3(24*3+16*12, n_hid_dec, 15*9)

        self.smpl_model_func = art.ParametricModel(conf.paths.smpl_file)
        self.global_to_local_pose = self.smpl_model_func.inverse_kinematics_R  # 全局到局部坐标的转换函数
        self.loadPretrain()  # 加载预训练模型
        self.eval()  # 评估、不训练

    def forward(self, x, saperateTrain=True):
        r'''
            要求输入imu的顺序：根、左右脚、头、左右手。SMPL joint order: [0,7,8,12,20,21]
        '''
        n, t, _ = x.shape  # x:[n,t,acc(18)+ori(54)]
        acc = x[:, :, :18].view(n, t, 6, 3)
        ori = x[:, :, 18:].view(n, t, 6, 9)  # order: 根、左右脚、头、左右手

        input1 = torch.cat((acc, ori), -1).view(n, t, -1)  # [n,t,6*12]
        output1 = self.gip1(input1)  # [n,t,15]

        p_leaf = output1.view(n, t, 5, 3)
        p_leaf = torch.cat((p_leaf.new_zeros(n, t, 1, 3), p_leaf), -2)  # [n,t,6,3]

        input2 = torch.cat((acc, ori, p_leaf), -1).view(n, t, -1)  # [n,t,6*15]
        if saperateTrain:
            input2_ = input2.detach()
        else:
            input2_ = input2
        output2 = self.gip2(input2_)

        p_all = output2.view(n, t, 23, 3)
        p_all = torch.cat((p_all.new_zeros(n, t, 1, 3), p_all), -2).view(n, t, 72)
        full_acc = acc.new_zeros(n, t, 16, 3)
        full_ori = ori.new_zeros(n, t, 16, 9)
        imu_pos = [0, 4, 5, 11, 14, 15]  # [14,15,4,5,11,0]左手右手，左腿右腿，头，根节点
        full_acc[:, :, imu_pos] = acc
        full_ori[:, :, imu_pos] = ori
        full_acc = full_acc.view(n, t, -1)
        full_ori = full_ori.view(n, t, -1)

        # input3 = torch.concat((p_all, full_acc, full_ori), dim=-1)  # 默认6d模型ggip3的输入顺序
        input3 = torch.concat((full_acc, full_ori, p_all), dim=-1)  # [n,t,24*3+16*12]
        if saperateTrain:
            input3_ = input3.detach()
        else:
            input3_ = input3
        output3 = self.gip3(input3_)  # [n,t,90]

        return output1, output2, output3

    # 输入减少的全局6d姿态，输出局部坐标系下的旋转矩阵  15->24
    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        batch = glb_reduced_pose.shape[0]
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(batch, -1, conf.joint_set.n_reduced,
                                                                                  3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(batch, glb_reduced_pose.shape[1], 24, 1,
                                                                               1)
        global_full_pose[:, :, conf.joint_set.reduced] = glb_reduced_pose

        pose = global_full_pose.clone().detach()
        for i in range(global_full_pose.shape[0]):
            pose[i] = self.global_to_local_pose(global_full_pose[i]).view(-1, 24, 3, 3)  # 到这一步变成了相对父节点的相对坐标
        pose[:, :, conf.joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, :, 0:1] = root_rotation.view(batch, -1, 1, 3, 3)  # 第一个是全局根节点方向
        return pose.contiguous()

    # 根据是否分离开训练加载相应的预训练权重
    def loadPretrain(self, seperate=False):
        if seperate:
            path1 = 'model/weight/seperateTri/Rl_192epoch.pkl'
            path2 = 'model/weight/seperateTri/Ra_80epoch.pkl'
            path3 = 'model/weight/seperateTri/Rp_280epoch.pkl'
            self.gip1.load_state_dict(torch.load(path1)['model_state_dict'])
            self.gip2.load_state_dict(torch.load(path2)['model_state_dict'])
            self.gip3.load_state_dict(torch.load(path3)['model_state_dict'])
        else:
            pathWight = 'model/weight/ggip_all_6d_optloss_spatial.pt'
            self.load_state_dict(torch.load(pathWight))

    def forwardRaw(self, imu):
        r'''
            要求输入imu的顺序：[n,t,72]
            acc和ori分开输入，acc在前(:18)，ori在后(18:)
            顺序为：关节点顺序为右手左手、右脚左脚、头、根，acc（18）+ori（54），已经经过标准化。(加速度没有除以缩放因子)
        '''
        n, t, _ = imu.shape
        acc = imu[:, :, :18].view(n, t, 6, 3)
        ori = imu[:, :, 18:].view(n, t, 6, 9)

        order = [5, 2, 3, 4, 0, 1]
        acc = acc[:, :, order]
        ori = ori[:, :, order]
        input = torch.cat((acc.view(n, t, -1), ori.view(n, t, -1)), dim=-1)
        leaf_pos, all_pos, r6dpose = self.forward(input)
        return leaf_pos, all_pos, r6dpose

    # 加入gip3 模型的前向计算，允许添加噪声以增强模型鲁棒性。
    def ggip3ForwardRaw(self, imu, joint_all):
        r'''
            标准输入：
                acc和ori分开输入，acc在前(:18)，ori在后(18:)
                顺序为：关节点顺序为右手左手、右脚左脚、头、根，acc（18）+ori（54），已经经过标准化。(加速度没有除以缩放因子)
                joint_all就是
        '''
        n, t, _ = imu.shape
        acc = imu[:, :, :18].view(n, t, 6, 3)
        ori = imu[:, :, 18:].view(n, t, 6, 9)

        order = [5, 2, 3, 4, 0, 1]
        acc = acc[:, :, order]
        ori = ori[:, :, order]

        full_acc = acc.new_zeros(n, t, 16, 3)
        full_ori = ori.new_zeros(n, t, 16, 9)
        imu_pos = [0, 4, 5, 11, 14, 15]  # [14,15,4,5,11,0]左手右手，左腿右腿，头，根节点
        full_acc[:, :, imu_pos] = acc
        full_ori[:, :, imu_pos] = ori
        full_acc = full_acc.view(n, t, -1)
        full_ori = full_ori.view(n, t, -1)

        p_all_modify = joint_all.view(n, t, 23 * 3)
        noise = 0.025 * torch.randn(p_all_modify.shape).to(p_all_modify.device).float()  # 为了鲁棒添加的高斯噪声，标准差为0.4
        p_all_noise = p_all_modify + noise
        p_all = p_all_noise.view(p_all_noise.shape[0], p_all_noise.shape[1], 23, 3)
        p_all = torch.cat((p_all.new_zeros(n, t, 1, 3), p_all), -2).view(n, t, 72)

        input = torch.concat((full_acc, full_ori, p_all), dim=-1)

        pose_6d = self.gip3.forward(input)
        # # pose_mat = self.gip3.forward(input)   # mat(9d) version
        # pose_euler = self.gip3.forward(input)   # euler(3d) version
        # pose_mat = euler2mat(pose_euler)

        return pose_6d  # []

    def calSMPLpose(self, imu):
        r'''
            要求输入imu的顺序：根、左右脚、头、左右手。SMPL joint order: [0,7,8,12,20,21]
            要求acc和ori分开输入，acc在前(:18)，ori在后(18:)
        '''
        _, _, global_pose = self.forward(imu)  # [n,t,15*6=90]
        return global_pose

    def calFullJointPos(self, imu):
        r'''
            要求输入imu的顺序：根、左右脚、头、左右手。SMPL joint order: [0,7,8,12,20,21]
            要求acc和ori分开输入，acc在前(:18)，ori在后(18:)
        '''
        _, full_joint_position, _ = self.forward(imu)  # [n,t,23*3]
        return full_joint_position

    # 在不进行额外预处理的情况下，根据单帧的加速度和方向数据预测姿态。
    @torch.no_grad()
    def predictPose_single(self, acc, ori, preprocess=False):
        r'''
            acc: [t,6,3]
            ori: [t,6,3,3]
            顺序为：根、左右脚、头、左右手（有预处理） /  右手左手、右脚左脚、头、根（无预处理）
        '''
        if not preprocess:
            order = [2, 3, 4, 0, 1, 5]
            acc_cal = acc[:, order]
            ori_cal = ori[:, order]

            acc_tmp = torch.cat((acc_cal[:, 5:], acc_cal[:, :5] - acc_cal[:, 5:]), dim=1).bmm(
                ori_cal[:, -1])  # / conf.acc_scale
            ori_tmp = torch.cat((ori_cal[:, 5:], ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5])), dim=1)
        else:
            acc_tmp = acc
            ori_tmp = ori

        t = acc_tmp.shape[0]
        root = ori_tmp[:, 0].view(t, 3, 3)

        acc_tmp = acc_tmp.view(t, -1)  # [t,18]
        ori_tmp = ori_tmp.view(t, -1)  # [t,54]
        imu = torch.cat((acc_tmp, ori_tmp), dim=-1).unsqueeze(0)  # [1,t,72]

        leaf_pos, all_pos, glo_pose = self.forward(imu)
        # pose = self._reduced_glb_euler_to_full_local_mat(root, glo_pose)     # euler版本
        # pose = self._reduced_glb_axis_to_full_local_mat(root, glo_pose)     # axis版本
        pose = self._reduced_glb_6d_to_full_local_mat(root, glo_pose)  # r6d版本
        # pose = self._reduced_glb_mat_to_full_local_mat(root, glo_pose)    # 矩阵版本
        return pose.squeeze(0)




# 可学习软约束层
class LearnableSoftLimits6DLayer(nn.Module):
    def __init__(self, num_joints):
        super(LearnableSoftLimits6DLayer, self).__init__()
        # 初始化每个关节的最小角度和最大角度，单位是弧度。可根据经验设定初值。
        self.min_angles = nn.Parameter(torch.full((num_joints,), -3.14))
        self.max_angles = nn.Parameter(torch.full((num_joints,), 3.14))

    def forward(self, pose_6d):
        """
        pose_6d: Tensor of shape [batch_size, num_joints*6]
        返回：
            Tensor of shape [batch_size, num_joints*6]（经过软约束的6D旋转参数）
        """
        n, t, d = pose_6d.size()
        pose_6d = pose_6d[:, :, :].view(n, t, 24, 6)

        # 1. 6D -> Rotation Matrix
        # (需要保证该函数是可微分的)
        rot_matrices = art.math.r6d_to_rotation_matrix(pose_6d)

        # 2. Rotation Matrix -> Axis-Angle 表示
        axis_angles = art.math.rotation_matrix_to_axis_angle(rot_matrices)  # shape: [batch_size, num_joints, 3]
        axis_angles = axis_angles.view(n, t, 24, 3)

        # 3. 提取角度（向量模长）和旋转轴
        angles = torch.norm(axis_angles, dim=-1)  # shape: [batch_size, num_joints]

        # 防止除零，加上一个小常数
        axis = axis_angles / (angles.unsqueeze(-1) + 1e-6)  # 保持数值稳定

        # 4. 对角度施加可学习软约束
        # 扩展可学习参数的维度以便于广播
        min_angles = self.min_angles.unsqueeze(0).unsqueeze(0)  # 拓展维度
        max_angles = torch.max(min_angles, self.max_angles.unsqueeze(0).unsqueeze(0))  # 拓展维度

        # 扩展 min_angles 和 max_angles 以便与 angles 广播
        min_angles = min_angles.expand(n, t, -1)  # shape: [batch_size, num_joints]
        max_angles = max_angles.expand(n, t, -1)  # shape: [batch_size, num_joints]

        # 对每个关节的旋转角度做 clamp
        clamped_angles = torch.clamp(angles, min=min_angles, max=max_angles)  # shape: [batch_size, num_joints]

        # 5. 重构新的 axis-angle 表示
        new_axis_angles = axis * clamped_angles.unsqueeze(-1)  # shape: [batch_size, num_joints, 3]

        # 6. Axis-Angle -> Rotation Matrix
        new_rot_matrices = art.math.axis_angle_to_rotation_matrix(
            new_axis_angles)  # shape: [batch_size, num_joints, 3, 3]

        # 7. Rotation Matrix -> 6D 表示
        new_pose_6d = art.math.rotation_matrix_to_r6d(new_rot_matrices)  # shape: [batch_size, num_joints, 6]

        # 8. 重塑回
        new_pose_6d = new_pose_6d.view(n, t, 144)
        return new_pose_6d