r"""
    包含角度计算的数学工具。
"""


__all__ = ['RotationRepresentation', 'to_rotation_matrix', 'radian_to_degree', 'degree_to_radian', 'normalize_angle',
           'angle_difference', 'angle_between', 'svd_rotate', 'generate_random_rotation_matrix',
           'axis_angle_to_rotation_matrix', 'rotation_matrix_to_axis_angle', 'r6d_to_rotation_matrix',
           'rotation_matrix_to_r6d', 'quaternion_to_axis_angle', 'axis_angle_to_quaternion',
           'quaternion_to_rotation_matrix', 'quat_to_rotation_matrix', 'rotation_matrix_to_euler_angle', 'euler_angle_to_rotation_matrix',
           'rotation_matrix_to_euler_angle_np', 'euler_angle_to_rotation_matrix_np', 'euler_convert_np','rotation_matrix_to_quat']


from .general import *
import enum
import numpy as np
import torch


class RotationRepresentation(enum.Enum):
    r"""
    旋转表示。四元数为wxyz格式，欧拉角为局部XYZ格式。
    """
    AXIS_ANGLE = 0
    ROTATION_MATRIX = 1
    QUATERNION = 2
    R6D = 3
    EULER_ANGLE = 4


def to_rotation_matrix(r: torch.Tensor, rep: RotationRepresentation):
    r"""
    将任意旋转表示转换为旋转矩阵。(torch, 批量)

    :param r: 旋转张量。
    :param rep: 输入中使用的旋转表示。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵张量。
    """
    if rep == RotationRepresentation.AXIS_ANGLE:
        return axis_angle_to_rotation_matrix(r)
    elif rep == RotationRepresentation.QUATERNION:
        return quaternion_to_rotation_matrix(r)
    elif rep == RotationRepresentation.R6D:
        return r6d_to_rotation_matrix(r)
    elif rep == RotationRepresentation.EULER_ANGLE:
        return euler_angle_to_rotation_matrix(r)
    elif rep == RotationRepresentation.ROTATION_MATRIX:
        return r.view(-1, 3, 3)
    else:
        raise Exception('未知的旋转表示')
def radian_to_degree(q):
    r"""
    将弧度转换为角度。
    """
    return q * 180.0 / np.pi


def degree_to_radian(q):
    r"""
    将角度转换为弧度。
    """
    return q / 180.0 * np.pi


def normalize_angle(q):
    r"""
    将弧度归一化到[-pi, pi)区间。(np/torch, 批量)

    :param q: 弧度角张量(np/torch)。
    :return: 归一化后的张量，每个角度在[-pi, pi)区间内。
    """
    mod = q % (2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


def angle_difference(target, source):
    r"""
    计算归一化的target - source。(np/torch, 批量)
    """
    return normalize_angle(target - source)


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor, rep=RotationRepresentation.ROTATION_MATRIX):
    r"""
    计算两个旋转之间的角度（弧度）。(torch, 批量)

    :param rot1: 可以重塑为[batch_size, rep_dim]的旋转张量1。
    :param rot2: 可以重塑为[batch_size, rep_dim]的旋转张量2。
    :param rep: 输入中使用的旋转表示。
    :return: 形状为[batch_size]的弧度角张量。
    """
    rot1 = to_rotation_matrix(rot1, rep)
    rot2 = to_rotation_matrix(rot2, rep)
    offsets = rot1.transpose(1, 2).bmm(rot2)
    angles = rotation_matrix_to_axis_angle(offsets).norm(dim=1)
    return angles


def svd_rotate(source_points: torch.Tensor, target_points: torch.Tensor):
    r"""
    获取将源点旋转到对应目标点的旋转矩阵。(torch, 批量)

    :param source_points: 形状为[batch_size, m, n]的源点。m为点数，n为维度。
    :param target_points: 形状为[batch_size, m, n]的目标点。m为点数，n为维度。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵，将源点旋转到目标点。
    """
    usv = [m.svd() for m in source_points.transpose(1, 2).bmm(target_points)]
    u = torch.stack([_[0] for _ in usv])
    v = torch.stack([_[2] for _ in usv])
    vut = v.bmm(u.transpose(1, 2))
    for i in range(vut.shape[0]):
        if vut[i].det() < -0.9:
            v[i, 2].neg_()
            vut[i] = v[i].mm(u[i].t())
    return vut


def generate_random_rotation_matrix(n=1):
    r"""
    生成随机旋转矩阵。(torch, 批量)

    :param n: 要生成的旋转矩阵数量。
    :return: 形状为[n, 3, 3]的随机旋转矩阵。
    """
    q = torch.zeros(n, 4)
    while True:
        n = q.norm(dim=1)
        mask = (n == 0) | (n > 1)
        if q[mask].shape[0] == 0:
            break
        q[mask] = torch.rand_like(q[mask]) * 2 - 1
    q = q / q.norm(dim=1, keepdim=True)
    return quaternion_to_rotation_matrix(q)


def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    将轴角转换为旋转矩阵。(torch, 批量)

    :param a: 可以重塑为[batch_size, 3]的轴角张量。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵。
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r


def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    将旋转矩阵转换为轴角。(torch, 批量)

    :param r: 可以重塑为[batch_size, 3, 3]的旋转矩阵张量。
    :return: 形状为[batch_size, 3]的轴角张量。
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result


def r6d_to_rotation_matrix(r6d: torch.Tensor):
    r"""
    将6D向量转换为旋转矩阵。(torch, 批量)

    **警告:** 任何6D向量的两个3D向量必须线性无关。

    :param r6d: 可以重塑为[batch_size, 6]的6D向量张量。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵张量。
    """
    r6d = r6d.view(-1, 6)
    column0 = normalize_tensor(r6d[:, 0:3])
    column1 = normalize_tensor(r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0)
    column2 = column0.cross(column1, dim=1)
    r = torch.stack((column0, column1, column2), dim=-1)
    r[torch.isnan(r)] = 0
    return r


def rotation_matrix_to_r6d(r: torch.Tensor):
    r"""
    将旋转矩阵转换为6D向量。(torch, 批量)

    :param r: 可以重塑为[batch_size, 3, 3]的旋转矩阵张量。
    :return: 形状为[batch_size, 6]的6D向量张量。
    """
    return r.view(-1, 3, 3)[:, :, :2].transpose(1, 2).clone().contiguous().view(-1, 6)


def quaternion_to_axis_angle(q: torch.Tensor):
    r"""
    将（未归一化的）四元数wxyz转换为轴角。(torch, 批量)

    **警告**: 返回的轴角可能具有大于180度的旋转（在180 ~ 360度之间）。

    :param q: 可以重塑为[batch_size, 4]的四元数张量。
    :return: 形状为[batch_size, 3]的轴角张量。
    """
    q = normalize_tensor(q.view(-1, 4))
    theta_half = q[:, 0].clamp(min=-1, max=1).acos()
    a = (q[:, 1:] / theta_half.sin().view(-1, 1) * 2 * theta_half.view(-1, 1)).view(-1, 3)
    a[torch.isnan(a)] = 0
    return a


def axis_angle_to_quaternion(a: torch.Tensor):
    r"""
    将轴角转换为四元数。(torch, 批量)

    :param a: 可以重塑为[batch_size, 3]的轴角张量。
    :return: 形状为[batch_size, 4]的四元数wxyz张量。
    """
    axes, angles = normalize_tensor(a.view(-1, 3), return_norm=True)
    axes[torch.isnan(axes)] = 0
    q = torch.cat(((angles / 2).cos(), (angles / 2).sin() * axes), dim=1)
    return q


def quaternion_to_rotation_matrix(q: torch.Tensor):
    r"""
    将（未归一化的）四元数wxyz转换为旋转矩阵。(torch, 批量)

    :param q: 可以重塑为[batch_size, 4]的四元数张量。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵张量。
    """
    q = normalize_tensor(q.view(-1, 4))
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                   2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                   2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
    return r.view(-1, 3, 3)

def quat_to_rotation_matrix(q: torch.Tensor):
    r"""
    将四元数转换为旋转矩阵。(torch, 批量)

    :param q: 可以重塑为[batch_size, 4]的欧拉角张量。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵张量。
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(q.clone().detach().cpu().view(-1, 4).numpy())
    ret = torch.from_numpy(rot.as_matrix()).float().to(q.device)
    return ret


def rotation_matrix_to_euler_angle(r: torch.Tensor, seq='XYZ', deg=False):
    r"""
    将旋转矩阵转换为欧拉角。(torch, 批量)

    :param r: 可以重塑为[batch_size, 3, 3]的旋转矩阵张量。
    :param seq: 3个字符，属于{'X', 'Y', 'Z'}用于内在旋转，或{'x', 'y', 'z'}用于外在旋转（弧度）。
                详见scipy。
    :return: 形状为[batch_size, 3]的欧拉角张量。
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(r.clone().detach().cpu().view(-1, 3, 3).numpy())
    ret = torch.from_numpy(rot.as_euler(seq, degrees=deg)).float().to(r.device)
    return ret


def euler_angle_to_rotation_matrix(q: torch.Tensor, seq='XYZ'):
    r"""
    将欧拉角转换为旋转矩阵。(torch, 批量)

    :param q: 可以重塑为[batch_size, 3]的欧拉角张量。
    :param seq: 3个字符，属于{'X', 'Y', 'Z'}用于内在旋转，或{'x', 'y', 'z'}用于外在旋转（弧度）。
                详见scipy。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵张量。
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler(seq, q.clone().detach().cpu().view(-1, 3).numpy())
    ret = torch.from_numpy(rot.as_matrix()).float().to(q.device)
    return ret


def rotation_matrix_to_euler_angle_np(r, seq='XYZ'):
    r"""
    将旋转矩阵转换为欧拉角。(numpy, 批量)

    :param r: 可以重塑为[batch_size, 3, 3]的旋转矩阵(np/torch)。
    :param seq: 3个字符，属于{'X', 'Y', 'Z'}用于内在旋转，或{'x', 'y', 'z'}用于外在旋转（弧度）。
                详见scipy。
    :return: 形状为[batch_size, 3]的欧拉角ndarray。
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.array(r).reshape(-1, 3, 3)).as_euler(seq)


def euler_angle_to_rotation_matrix_np(q, seq='XYZ'):
    r"""
    将欧拉角转换为旋转矩阵。(numpy, 批量)

    :param q: 可以重塑为[batch_size, 3]的欧拉角(np/torch)。
    :param seq: 3个字符，属于{'X', 'Y', 'Z'}用于内在旋转，或{'x', 'y', 'z'}用于外在旋转（弧度）。
                详见scipy。
    :return: 形状为[batch_size, 3, 3]的旋转矩阵ndarray。
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(seq, np.array(q).reshape(-1, 3)).as_matrix()


def euler_convert_np(q, from_seq='XYZ', to_seq='XYZ'):          #旋转完一个轴，再旋转另一个轴
    r"""
    将欧拉角转换为不同的轴顺序。(numpy, 单/批量)

    :param q: 以from_seq顺序的欧拉角（弧度）ndarray。形状为[3]或[N, 3]。
    :param from_seq: 源（输入）轴顺序。详见scipy。
    :param to_seq: 目标（输出）轴顺序。详见scipy。
    :return: 相同大小的ndarray，但以to_seq顺序。
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(from_seq, q).as_euler(to_seq)


def rotation_matrix_to_quat(r: torch.Tensor):
    r"""
    将旋转矩阵转换为四元数。(numpy)

    :param r: 可以重塑为[batch_size, 3, 3]的旋转矩阵张量。
    :return: 形状为[batch_size, 4]的四元数张量。
    """
    from scipy.spatial.transform import Rotation
    tmp = Rotation.from_matrix(np.array(r).reshape(-1, 3, 3)).as_quat()
    return torch.from_numpy(tmp).float().to(r.device)
