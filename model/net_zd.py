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
        self.msfke = NodeAwareMSFKE(
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