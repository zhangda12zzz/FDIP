r"""
    通用数学工具。
"""


__all__ = ['lerp', 'normalize_tensor', 'append_value', 'append_zero', 'append_one', 'vector_cross_matrix',
           'vector_cross_matrix_np', 'block_diagonal_matrix_np']


import numpy as np
import torch
from functools import partial   #“记住”原始函数的某些固定参数


def lerp(a, b, t):
    r"""
    线性插值（未限制范围）。

    :param a: 起始值。
    :param b: 结束值。
    :param t: 插值权重。t = 0 返回 a；t = 1 返回 b。
    :return: 线性插值结果。
    """
    return a * (1 - t) + b * t


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    将张量在指定维度上归一化为单位范数。（PyTorch）

    :param x: 任意形状的张量。
    :param dim: 需要归一化的维度。
    :param return_norm: 如果为 True，同时返回范数（长度）张量。
    :return: 相同形状的张量。如果 return_norm 为 True，同时返回形状为 [*, 1, *]（1 在 dim 维度）的范数张量（keepdim=True）。
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def append_value(x: torch.Tensor, value: float, dim=-1):
    r"""
    在指定维度上向张量追加一个值。（PyTorch）

    例如，append_value(torch.zeros(3, 3, 3), 1, dim=1) 将返回一个形状为 [3, 4, 3] 的张量，其中追加的部分为 1。

    :param x: 任意形状的张量。
    :param value: 要追加的值。
    :param dim: 要扩展的维度。
    :return: 形状相同的张量，除了扩展的维度增加了 1。
    """
    app = torch.ones_like(x.index_select(dim, torch.tensor([0], device=x.device))) * value
    x = torch.cat((x, app), dim=dim)
    return x


append_zero = partial(append_value, value=0)
append_one = partial(append_value, value=1)


def vector_cross_matrix(x: torch.Tensor):
    r"""
    获取每个向量3 `v` 的斜对称矩阵 :math:`[v]_\times\in so(3)`。（PyTorch，批处理）

    :param x: 可以重塑为 [batch_size, 3] 的张量。
    :return: 形状为 [batch_size, 3, 3] 的斜对称矩阵。
    """
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)


def vector_cross_matrix_np(x):
    r"""
    获取向量3 `v` 的斜对称矩阵 :math:`[v]_\times\in so(3)`。（NumPy，单例）

    :param x: 形状为 [3] 的向量3。
    :return: 形状为 [3, 3] 的斜对称矩阵。
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]], dtype=float)


def block_diagonal_matrix_np(matrix2d_list):
    r"""
    使用一系列二维矩阵生成块对角矩阵。（NumPy，单例）

    :param matrix2d_list: 二维矩阵列表（2darray）。
    :return: 块对角矩阵。
    """
    ret = np.zeros(sum([np.array(m.shape) for m in matrix2d_list]))
    r, c = 0, 0
    for m in matrix2d_list:
        lr, lc = m.shape
        ret[r:r+lr, c:c+lc] = m
        r += lr
        c += lc
    return ret
