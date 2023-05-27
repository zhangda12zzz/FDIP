from torch.utils.data import Dataset
import os
import sys
import numpy as np
import torch

from option_parser import get_std_bvh
from utils.Quaternions import Quaternions


class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args):
        super(MotionData, self).__init__()
        name = args.dataset     # 模型名，比如'Smpl'
        file_path = './dataset/Mixamo/{}.npy'.format(name)     # 存储了所有bvh运动数据（归一化后）的npy文件

        if args.debug:
            file_path = file_path[:-4] + '_debug' + file_path[-4:]

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        self.std_bvh = get_std_bvh(args)    # 存储了标准身体模型数据（T姿势）的bvh文件路径
        self.args = args
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)     # 读取动态数据
        motions = list(motions)                             # 动态数据尺寸[n,t,C_dynamic], C_dynamic = 3*(v-1)+3
        new_windows = self.get_windows(motions)             # [n,t_w,C], t_w=64窗口大小，C = 4*(v-1)+3（使用了四元数表示的话）
        self.data.append(new_windows)
        self.data = torch.cat(self.data)            # 从list转化为tensor
        self.data = self.data.permute(0, 2, 1)      # [n,C,t_w]

        if args.normalization == 1:     # 需要归一化，
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.var = torch.var(self.data, (0, 2), keepdim=True)
            self.var = self.var ** (1/2)
            idx = self.var < 1e-5
            self.var[idx] = 1   # 对于方差很小的数据，不进行方差层面的归一化处理（不然数据会变得很大）
            self.data = (self.data - self.mean) / self.var
        else:   # 不需要的话【一般是已经提前归一化过了】，就记录均值为0，方差为1【统一后续的反归一化操作】
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.mean.zero_()
            self.var = torch.ones_like(self.mean)

        train_len = self.data.shape[0] * 95 // 100      # 训练集占95%
        self.test_set = self.data[train_len:, ...]      # 测试集是后5%的部分（这里是验证集吧？）
        self.data = self.data[:train_len, ...]          # 提取训练集
        self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())   # 时间倒序数据（数据增强用？）

        self.reset_length_flag = 0
        self.virtual_length = 0
        print('Window count: {}, total frame (without downsampling): {}'.format(len(self), self.total_frame))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
        if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
            return self.data[item]
        else:
            return self.data_reverse[item]

    def get_windows(self, motions): # 数据一律按照60fps处理
        new_windows = []

        for motion in motions:
            self.total_frame += motion.shape[0]         # 统计总帧数
            motion = self.subsample(motion)             # 降维采样，60fps->30fps
            self.motion_length.append(motion.shape[0])  # 统计降采样后的总帧数
            step_size = self.args.window_size // 2      # 计算步长【类比conv过程的step！】
            window_size = step_size * 2                 # 偶数化window_size
            n_window = motion.shape[0] // step_size - 1 # 窗口数【让窗口覆盖整个时间序列、窗口间有重叠】
            for i in range(n_window):
                begin = i * step_size       # 窗口起始帧
                end = begin + window_size   # 窗口结束帧

                new = motion[begin:end, :]  # 裁剪新的数据样本
                if self.args.rotation == 'quaternion':      # 如果用四元数表示旋转，
                    new = new.reshape(new.shape[0], -1, 3)
                    rotations = new[:, :-1, :]              # 提取旋转部分【最后一维表示的是根节点位移】
                    rotations = Quaternions.from_euler(np.radians(rotations)).qs
                    rotations = rotations.reshape(rotations.shape[0], -1)   # 转换成四元数，再flatten
                    positions = new[:, -1, :]
                    positions = np.concatenate((new, np.zeros((new.shape[0], new.shape[1], 1))), axis=2)    # 位移也扩充为4维（末尾补0）【这一步没用上！】
                    new = np.concatenate((rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)      # 拼接rot+pos，最终维度为 4*(v-1)+3

                new = new[np.newaxis, ...]  # [t,C] -> [1,t,C]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)   # [n,t,C], C=3*(v-1)+3或者4*(v-1)+3，应该是默认使用四元数表示

    def subsample(self, motion):
        return motion[::2, :]

    def denormalize(self, motion):
        if self.args.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans
