import os
import numpy as np
import copy
import torch
from dataset.bvh_parser import BVH_file
from dataset.motion_dataset import MotionData
from option_parser import get_args, try_mkdir

def get_windows_imu(motions): # 数据一律按照60fps处理
        new_windows = []

        for motion in motions:  # [t,42]
            step_size = 64 // 2      # 计算步长【类比conv过程的step！】
            window_size = step_size * 2                 # 偶数化window_size
            n_window = motion.shape[0] // step_size - 1 # 窗口数【让窗口覆盖整个时间序列、窗口间有重叠】

            for i in range(n_window):
                begin = i * step_size       # 窗口起始帧
                end = begin + window_size   # 窗口结束帧

                new = motion[begin:end, :]  # 裁剪新的数据样本
                
                new = new.reshape(new.shape[0], -1) # [t,6,7]=>[t,42]
                new = new[np.newaxis, ...]  # [t,C] -> [1,t,C]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)

def get_windows_motion(motions): # 数据一律按照60fps处理
        new_windows = []

        for motion in motions:  # [t,C(87)] => [t,36]，方向(18)在前，加速度(18)在后
            step_size = 64 // 2      # 计算步长【类比conv过程的step！】
            window_size = step_size * 2                 # 偶数化window_size
            n_window = motion.shape[0] // step_size - 1 # 窗口数【让窗口覆盖整个时间序列、窗口间有重叠】
            
            for i in range(n_window):
                begin = i * step_size       # 窗口起始帧
                end = begin + window_size   # 窗口结束帧

                new = motion[begin:end, :]  # 裁剪新的数据样本
                
                rot_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]
                parent_index = [0,1,2,3,0,5,6,7,0,9,10,11,12,11,14,15,16,11,18,19,20]
                new = new[:,rot_index] #[n,22,4]
                
                # 按照原本的设置，需要把关节方向转化为骨骼方向，所以需要取父节点的数据！22->21
                new = new[:, parent_index]
                
                new = new.reshape(new.shape[0], -1)
                new = np.concatenate((new, np.zeros((new.shape[0], 3))), axis=1)
                new = new[np.newaxis, ...]  # [t,C] -> [1,t,87]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)


file = BVH_file('dataset/CIP/work/CIP_std_22.bvh')
new_motion = file.to_tensor().permute((1, 0)).numpy()
print(new_motion)

input_path = './dataset/CIP/work/Smpl_dip_imus.npy'
file_path = './dataset/CIP/work/Smpl_dip_motion.npy'
imu_data = np.load(input_path, allow_pickle=True)   # [n,t,6,7]
motions = np.load(file_path, allow_pickle=True)     # [n,t,22,4]


imu_windows = torch.Tensor(get_windows_imu(imu_data))
new_windows = torch.Tensor(get_windows_motion(motions))
data_motion = new_windows.permute(0, 2, 1)        #[n,C,t_w] 动作真值
data_imu = imu_windows.permute(0, 2, 1) 

mean_motion = torch.mean(data_motion, (0, 2), keepdim=True)
var_motion = torch.var(data_motion, (0, 2), keepdim=True)
var_motion = var_motion ** (1/2)
idx = var_motion < 1e-5
var_motion[idx] = 1   # 对于方差很小的数据，不进行方差层面的归一化处理（不然数据会变得很大）

mean_imu = torch.mean(data_imu, (0, 2), keepdim=True)
var_imu = torch.var(data_imu, (0, 2), keepdim=True)
var_imu = var_imu ** (1/2)
idx = var_imu < 1e-5
var_imu[idx] = 1   # 对于方差很小的数据，不进行方差层面的归一化处理（不然数据会变得很大

# np.save('./dataset/CIP/work/Smpl_mean_dip_imus.npy', mean_imu)
# np.save('./dataset/CIP/work/Smpl_var_dip_imus.npy', var_imu)
np.save('./dataset/CIP/work/Smpl_mean_dip_motion.npy', mean_motion)
np.save('./dataset/CIP/work/Smpl_var_dip_motion.npy', var_motion)