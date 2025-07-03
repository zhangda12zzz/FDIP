r"""
    空间数学工具，用于刚体的线性和角度计算。
    还包括用于关节体运动学的工具。
"""


__all__ = ['transformation_matrix_np', 'adjoint_transformation_matrix_np', 'transformation_matrix',
           'decode_transformation_matrix', 'inverse_transformation_matrix', 'bone_vector_to_joint_position',
           'joint_position_to_bone_vector', 'forward_kinematics_R', 'inverse_kinematics_R', 'forward_kinematics_T',
           'inverse_kinematics_T', 'forward_kinematics']


from .general import *
import numpy as np
import torch
from functools import partial


def transformation_matrix_np(R, p):
    r"""
    获取齐次变换矩阵。(numpy, 单例)

    变换矩阵 :math:`T_{sb} \in SE(3)` 的形状为 [4, 4]，可以将点或向量从 b 坐标系转换到 s 坐标系：:math:`x_s = T_{sb}x_b`。

    :param R: b 坐标系在 s 坐标系中的旋转矩阵 R_sb，形状为 [3, 3]。
    :param p: b 坐标系在 s 坐标系中的位置 p_s，形状为 [3]。
    :return: 变换矩阵 T_sb，形状为 [4, 4]。
    """
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = p
    T[3, 3] = 1
    return T


def adjoint_transformation_matrix_np(R, p):
    r"""
    获取变换矩阵的伴随表示。(numpy, 单例)     # TODO: 待验证：：：和数学中的伴随矩阵的定义不同

    伴随矩阵 :math:`[Ad_{T_{sb}}]` 的形状为 [6, 6]，可以在 b/s 坐标系之间转换空间扭转/力/雅可比矩阵。
        :math:`\mathcal{V}_s = [Ad_{T_{sb}}]\mathcal{V}_b`

        :math:`\mathcal{F}_b = [Ad_{T_{sb}}]^T\mathcal{F}_s`

        :math:`J_s = [Ad_{T_{sb}}]J_b`

    :param R: b 坐标系在 s 坐标系中的旋转矩阵 R_sb，形状为 [3, 3]。
    :param p: b 坐标系在 s 坐标系中的位置 p_s，形状为 [3]。
    :return: 变换矩阵 T_sb 的伴随表示，形状为 [6, 6]。
    """
    AdT = np.zeros((6, 6))
    AdT[:3, :3] = R
    AdT[3:, 3:] = R
    AdT[3:, :3] = np.dot(vector_cross_matrix_np(p), R)
    return AdT


def transformation_matrix(R: torch.Tensor, p: torch.Tensor):
    r"""
    获取齐次变换矩阵。(torch, 批量)

    变换矩阵 :math:`T_{sb} \in SE(3)` 的形状为 [4, 4]，可以将点或向量从 b 坐标系转换到 s 坐标系：:math:`x_s = T_{sb}x_b`。

    :param R: b 坐标系在 s 坐标系中的旋转矩阵 R_sb，形状为 [*, 3, 3]。
    :param p: b 坐标系在 s 坐标系中的位置 p_s，形状为 [*, 3]。
    :return: 变换矩阵 T_sb，形状为 [*, 4, 4]。
    """
    Rp = torch.cat((R, p.unsqueeze(-1)), dim=-1)
    OI = torch.cat((torch.zeros(list(Rp.shape[:-2]) + [1, 3], device=R.device),
                    torch.ones(list(Rp.shape[:-2]) + [1, 1], device=R.device)), dim=-1)
    T = torch.cat((Rp, OI), dim=-2)
    return T


def decode_transformation_matrix(T: torch.Tensor):
    r"""
    从输入的齐次变换矩阵中解码旋转和位置。(torch, 批量)

    :param T: 变换矩阵，形状为 [*, 4, 4]。
    :return: 旋转和位置，形状为 [*, 3, 3] 和 [*, 3]。
    """
    R = T[..., :3, :3].clone()
    p = T[..., :3, 3].clone()
    return R, p


def inverse_transformation_matrix(T: torch.Tensor):
    r"""
    获取输入齐次变换矩阵的逆矩阵。(torch, 批量)

    :param T: 变换矩阵，形状为 [*, 4, 4]。
    :return: 逆矩阵，形状为 [*, 4, 4]。
    """
    R, p = decode_transformation_matrix(T)
    invR = R.transpose(-1, -2)
    invp = -torch.matmul(invR, p.unsqueeze(-1)).squeeze(-1)
    invT = transformation_matrix(invR, invp)
    return invT


def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):     #沿着数结构做某种操作，每一批操作一次
    r"""
    沿着树分支乘/加矩阵。x_local [N, J, *]。parent [J]。
    """
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
    x_global = torch.stack(x_global, dim=1)
    return x_global


def _inverse_tree(x_global: torch.Tensor, parent, reduction_fn, inverse_fn):
    r"""
    沿着树分支逆乘/加矩阵。x_global [N, J, *]。parent [J]。
    """
    x_local = [x_global[:, 0]]
    for i in range(1, len(parent)):
        x_local.append(reduction_fn(inverse_fn(x_global[:, parent[i]]), x_global[:, i]))
    x_local = torch.stack(x_local, dim=1)
    return x_local


def bone_vector_to_joint_position(bone_vec: torch.Tensor, parent):
    r"""
    从骨骼向量（子关节和父关节的位置差）计算关节在基础坐标系中的位置。(torch, 批量)

    注意
    -----
    bone_vec[:, i] 是从 parent[i] 到 i 的向量。

    parent[i] 应该是关节 i 的父关节 ID。对于任何 i > 0，parent[i] 必须小于 i。

    参数
    -----
    :param bone_vec: 骨骼向量张量，形状为 [batch_size, *]，可以重塑为 [batch_size, num_joint, 3]。
    :param parent: 父关节 ID 列表，形状为 [num_joint]。对于基础 ID（parent[0]），使用 -1 或 None。
    :return: 关节位置，形状为 [batch_size, num_joint, 3]。
    """
    bone_vec = bone_vec.view(bone_vec.shape[0], -1, 3)
    joint_pos = _forward_tree(bone_vec, parent, torch.add)
    return joint_pos


def joint_position_to_bone_vector(joint_pos: torch.Tensor, parent):
    r"""
    从关节在基础坐标系中的位置计算骨骼向量（子关节和父关节的位置差）。(torch, 批量)

    注意
    -----
    bone_vec[:, i] 是从 parent[i] 到 i 的向量。

    parent[i] 应该是关节 i 的父关节 ID。对于任何 i > 0，parent[i] 必须小于 i。

    参数
    -----
    :param joint_pos: 关节位置张量，形状为 [batch_size, *]，可以重塑为 [batch_size, num_joint, 3]。
    :param parent: 父关节 ID 列表，形状为 [num_joint]。对于基础 ID（parent[0]），使用 -1 或 None。
    :return: 骨骼向量，形状为 [batch_size, num_joint, 3]。
    """
    joint_pos = joint_pos.view(joint_pos.shape[0], -1, 3)
    bone_vec = _inverse_tree(joint_pos, parent, torch.add, torch.neg)
    return bone_vec


def forward_kinematics_R(R_local: torch.Tensor, parent):
    r"""
    :math:`R_global = FK(R_local)`

    正向运动学，从局部旋转计算每个关节的全局旋转。(torch, 批量)

    注意
    -----
    关节的 *局部* 旋转在其父坐标系中表示。

    关节的 *全局* 旋转在基础（根父）坐标系中表示。

    R_local[:, i], parent[i] 应该是关节 i 的局部旋转和父关节 ID。对于任何 i > 0，parent[i] 必须小于 i。

    参数
    -----
    :param R_local: 关节局部旋转张量，形状为 [batch_size, *]，可以重塑为 [batch_size, num_joint, 3, 3]（旋转矩阵）。
    :param parent: 父关节 ID 列表，形状为 [num_joint]。对于基础 ID（parent[0]），使用 -1 或 None。
    :return: 关节全局旋转，形状为 [batch_size, num_joint, 3, 3]。
    """
    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    R_global = _forward_tree(R_local, parent, torch.bmm)
    return R_global


def inverse_kinematics_R(R_global: torch.Tensor, parent):
    r"""
    :math:`R_local = IK(R_global)`

    逆向运动学，从全局旋转计算每个关节的局部旋转。(torch, 批量)

    注意
    -----
    关节的 *局部* 旋转在其父坐标系中表示。

    关节的 *全局* 旋转在基础（根父）坐标系中表示。

    R_global[:, i], parent[i] 应该是关节 i 的全局旋转和父关节 ID。对于任何 i > 0，parent[i] 必须小于 i。

    参数
    -----
    :param R_global: 关节全局旋转张量，形状为 [batch_size, *]，可以重塑为 [batch_size, num_joint, 3, 3]（旋转矩阵）。
    :param parent: 父关节 ID 列表，形状为 [num_joint]。对于基础 ID（parent[0]），使用 -1 或 None。
    :return: 关节局部旋转，形状为 [batch_size, num_joint, 3, 3]。
    """
    R_global = R_global.view(R_global.shape[0], -1, 3, 3)
    R_local = _inverse_tree(R_global, parent, torch.bmm, partial(torch.transpose, dim0=1, dim1=2))
    return R_local


def forward_kinematics_T(T_local: torch.Tensor, parent):
    r"""
    :math:`T_global = FK(T_local)`

    正向运动学，从局部齐次变换计算每个关节的全局齐次变换。(torch, 批量)

    注意
    -----
    关节的 *局部* 变换在其父坐标系中表示。

    关节的 *全局* 变换在基础（根父）坐标系中表示。

    T_local[:, i], parent[i] 应该是关节 i 的局部变换矩阵和父关节 ID。对于任何 i > 0，parent[i] 必须小于 i。

    参数
    -----
    :param T_local: 关节局部变换张量，形状为 [batch_size, *]，可以重塑为 [batch_size, num_joint, 4, 4]（齐次变换矩阵）。
    :param parent: 父关节 ID 列表，形状为 [num_joint]。对于基础 ID（parent[0]），使用 -1 或 None。
    :return: 关节全局变换矩阵，形状为 [batch_size, num_joint, 4, 4]。
    """
    T_local = T_local.view(T_local.shape[0], -1, 4, 4)
    T_global = _forward_tree(T_local, parent, torch.bmm)
    return T_global


def inverse_kinematics_T(T_global: torch.Tensor, parent):
    r"""
    :math:`T_local = IK(T_global)`

    逆向运动学，从全局齐次变换计算每个关节的局部齐次变换。(torch, 批量)

    注意
    -----
    关节的 *局部* 变换在其父坐标系中表示。

    关节的 *全局* 变换在基础（根父）坐标系中表示。

    T_global[:, i], parent[i] 应该是关节 i 的全局变换矩阵和父关节 ID。对于任何 i > 0，parent[i] 必须小于 i。

    参数
    -----
    :param T_global: 关节全局变换张量，形状为 [batch_size, *]，可以重塑为 [batch_size, num_joint, 4, 4]（齐次变换矩阵）。
    :param parent: 父关节 ID 列表，形状为 [num_joint]。对于基础 ID（parent[0]），使用 -1 或 None。
    :return: 关节局部变换矩阵，形状为 [batch_size, num_joint, 4, 4]。
    """
    T_global = T_global.view(T_global.shape[0], -1, 4, 4)
    T_local = _inverse_tree(T_global, parent, torch.bmm, inverse_transformation_matrix)
    return T_local


def forward_kinematics(R_local: torch.Tensor, p_local: torch.Tensor, parent):   #正向运动学不用其次变换
    r"""
    :math:`R_global, p_global = FK(R_local, p_local)`

    正向运动学，从局部旋转和位置计算每个关节的全局旋转和位置。(torch, 批量)

    注意
    -----
    关节的 *局部* 旋转和位置在其父坐标系中表示。

    关节的 *全局* 旋转和位置在基础（根父）坐标系中表示。

    R_local[:, i], p_local[:, i], parent[i] 应该是关节 i 的局部旋转、局部位置和父关节 ID。对于任何 i > 0，parent[i] 必须小于 i。

    参数
    -----
    :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 3, 3] (rotation matrices).
    :param p_local: Joint local position tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 3] (zero-pose bone vectors).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint global rotation and position, in shape [batch_size, num_joint, 3, 3] and [batch_size, num_joint, 3].
    """
    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    p_local = p_local.view(p_local.shape[0], -1, 3)
    T_local = transformation_matrix(R_local, p_local)
    T_global = forward_kinematics_T(T_local, parent)
    return decode_transformation_matrix(T_global)
