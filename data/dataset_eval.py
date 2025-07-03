"""
ImuMotionDataEval 类：用于评估阶段的 IMU 数据集处理

主要功能：
1. 数据加载与初始化：
   - 根据指定数据集（DIP-IMU、TotalCapture、Mixamo、SingleOne）加载预处理好的 IMU 传感器数据和 SMPL 全局姿态数据
   - 支持多数据集路径配置，自动适配不同数据源的文件路径

2. 数据预处理：
   - IMU 数据处理：将 numpy 格式的原始 IMU 数据转换为 PyTorch Tensor，并提取根节点运动信息
   - 姿态数据转换：将 SMPL 的 24 关节全局旋转矩阵（3x3）转换为 R6D 表示（6D 旋转表示格式）
   - 时间窗口分割：将长序列数据分割为重叠的时间窗口，便于模型处理（窗口化处理）

3. 数据提供：
   - 通过 __getitem__ 提供单条数据（包含 IMU 输入、目标姿态等）
   - 通过 getValData 提供完整的评估数据集
   - 支持 PyTorch 模型的设备（CPU/GPU）数据传输

核心转换流程：
原始数据 → R6D 格式转换 → 时间窗口分割 → 标准化数据格式 → 模型输入

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import paths, joint_set
import articulate as art
import option_parser
import random


class ImuMotionDataEval(Dataset):
    def __init__(self, args, dataset='dip', std_path=None):
        super(ImuMotionDataEval, self).__init__()
        self.args = args
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.body_model = art.ParametricModel(paths.smpl_file)
        self.dataset_name = dataset

        # dip
        if self.dataset_name == 'dip':
            imu_path = r'F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_IMU_nn\imu_own_test.npz'
            pose_path = 'data/data_all/DIP_IMU/Smpl_dipTrain_motion_SMPL24_test.npy'
        # tc
        elif self.dataset_name == 'tc':
            imu_path = 'data/test/Totalcapture/Smpl_tc_imus_test.npy'
            pose_path = 'data/test/Totalcapture/Smpl_tc_motion_SMPL24_test.npy'
        # mixamo
        elif self.dataset_name == 'mixamo':
            imu_path = r'F:\CodeForPaper\Dataset\mixamo\mixamoSMPLPos_imu_test.npy'
            pose_path = r'F:\CodeForPaper\Dataset\mixamo\mixamoSMPLPos_motion_SMPL24_test.npy'
        # singleone-imu
        else:
            imu_path = r'F:\CodeForPaper\Dataset\SingleOne\processed\Smpl_singleone_imus_test.npy'
            pose_path = r'F:\CodeForPaper\Dataset\SingleOne\processed\Smpl_singleone_motion_SMPL24_test.npy'

        imu_raw = np.load(imu_path, allow_pickle=True)
        pose_raw = np.load(pose_path, allow_pickle=True)

        imu_data, pose_data, root_data = [], [], []

        for a_imu in imu_raw:
            a_imu = torch.Tensor(a_imu)
            imu_data.append(a_imu.clone())

            root_windows = a_imu[:, -9:].view(a_imu.shape[0], 3, 3)
            root_data.append(root_windows)

        pose_data = self.smpl24ToR6dEval(pose_raw)    # 将全局姿态转换为 R6D 格式，并返回结果。

        self.imus = imu_data  # [t,72]*n
        self.poses = pose_data  # [t,90]*n
        self.roots = root_data  # [t,3,3]*n

    def __len__(self):
        return self.imus.shape[0]

    def __getitem__(self, item):  # 必定是train环境
        if isinstance(item, int): item %= self.imus.shape[0]
        return [self.imus[item].to(self.device), self.joints[item].to(self.device), self.poses[item].to(self.device),
                self.poses[item].to(self.device), self.shapes[item].to(self.device)]

    def getValData(self):
        # self.idxCount += 1
        return [self.imus, self.poses, self.roots]
        # return [self.imus, self.joints, self.poses, self.roots, self.poses_ref]

    # 将输入数据分割成多个时间窗口
    def get_windows(self, input):
        new_windows = []
        self.total_frame = 0
        for motion in input:  # [t,C,c,c,]
            motion = torch.Tensor(motion)
            self.total_frame += motion.shape[0]  # 统计总帧数

            step_size = self.args.window_size // 2  # 计算步长【类比conv过程的step！】
            window_size = step_size * 2  # 偶数化window_size
            n_window = motion.shape[0] // window_size - 1  # 窗口数【让窗口覆盖整个时间序列、窗口间有重叠】

            for i in range(n_window):
                begin = i * step_size  # 窗口起始帧
                end = begin + window_size  # 窗口结束帧

                new = motion[begin:end, :]
                new = new[np.newaxis, ...]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)
        return torch.cat(new_windows)

    def smpl24ToR6dEval(self, pose):
        '''
            pose: [n,t,24,3,3]
            >> return => r6d_windows:[n,t,15,6]
        '''
        res = []
        for a_pose in pose:
            a_pose = torch.Tensor(a_pose)
            t = a_pose.shape[0]
            a_pose = a_pose.view(-1, 24, 3, 3)  # [t,24,3,3]
            global_pose = self.body_model.forward_kinematics(a_pose, calc_mesh=False)[0]
            grot_reduce_raw = global_pose[:, joint_set.reduced]
            grot_reduce = global_pose[:, 0:1].transpose(2, 3).matmul(grot_reduce_raw)
            grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
            for j in range(grot_reduce.shape[0]):
                grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])  # tensor[15, 6]
            r6d_windows = grot_copy.contiguous()
            r6d_windows = r6d_windows.view(t, 15 * 6)
            res.append(r6d_windows)
        return res


if __name__ == '__main__':
    args = option_parser.get_args()
    device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
    args.dataset = 'Smpl'
    args.device = device
    dataset = ImuMotionDataEval(args)