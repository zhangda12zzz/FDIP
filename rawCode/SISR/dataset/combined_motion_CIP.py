import os
import numpy as np
import torch
from torch.utils.data import Dataset
import copy

from dataset.bvh_parser import BVH_file
from dataset import get_test_set
from option_parser import get_std_bvh
from utils.Quaternions import Quaternions

class ImuMotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args, std_path=None):
        super(ImuMotionData, self).__init__()
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')

        name = args.dataset     # 模型名，比如'Smpl'
        input_path = './dataset/CIP/tpFirst/{}_amass_imus_TP.npy'.format(name)
        file_path = './dataset/CIP/work/{}_amass_motion.npy'.format(name)     # 存储了所有bvh运动数据（归一化后）的npy文件
        
        if args.debug:
            file_path = file_path[:-4] + '_debug' + file_path[-4:]

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        if std_path is not None:
            self.std_bvh = std_path
        else:
            self.std_bvh = get_std_bvh(args)    # 存储了标准身体模型数据（T姿势）的bvh文件路径
        self.args = args
        self.data = []
        self.motion_length = []
        
        # TODO:读取数据并处理
        imu_data = np.load(input_path, allow_pickle=True)   # [n,t,6,7]
        motions = np.load(file_path, allow_pickle=True)     # [n,t,22,4]
        motions = list(motions)                             # 动态数据尺寸[n,t,C_dynamic], C_dynamic = 3*(v-1)+3
        
        imu_windows = torch.Tensor(self.get_windows_imu(imu_data))
        new_windows = torch.Tensor(self.get_windows_motion(motions))           # [n,t_w,42], t_w=64窗口大小，C = 4*(v-1)+3（使用了四元数表示的话）
        
        self.data_motion = new_windows.permute(0, 2, 1)        #[n,C,t_w] 动作真值
        self.data_imu = imu_windows.permute(0, 2, 1)    #[n,7,t_w]
        self.data = [(self.data_imu[p], self.data_motion[p]) for p in range(self.data_imu.shape[0])]
        # self.data = self.data.permute(0, 2, 1)      # [n,C,t_w]

        if args.normalization == 1:     # 需要归一化，
            self.mean_motion = torch.mean(self.data_motion, (0, 2), keepdim=True)
            self.var_motion = torch.var(self.data_motion, (0, 2), keepdim=True)
            self.var_motion = self.var_motion ** (1/2)
            idx = self.var_motion < 1e-5
            self.var_motion[idx] = 1   # 对于方差很小的数据，不进行方差层面的归一化处理（不然数据会变得很大）
            self.data_motion = (self.data_motion - self.mean_motion) / self.var_motion
            
            self.mean_imu = torch.mean(self.data_imu, (0, 2), keepdim=True)
            self.var_imu = torch.var(self.data_imu, (0, 2), keepdim=True)
            self.var_imu = self.var_imu ** (1/2)
            idx = self.var_imu < 1e-5
            self.var_imu[idx] = 1   # 对于方差很小的数据，不进行方差层面的归一化处理（不然数据会变得很大）
            self.data_imu = (self.data_imu - self.mean_imu) / self.var_imu
            
        else:   # 不需要的话【一般是已经提前归一化过了】，就记录均值为0，方差为1【统一后续的反归一化操作】
            self.mean_motion = torch.mean(self.data_motion, (0, 2), keepdim=True)
            self.mean_motion.zero_()
            self.var_motion = torch.ones_like(self.mean_motion)
            self.mean_imu = torch.mean(self.data_imu, (0, 2), keepdim=True)
            self.mean_imu.zero_()
            self.var_imu = torch.ones_like(self.mean_imu)
            
        np.save('./dataset/CIP/work/{}_mean_imus.npy'.format(name), self.mean_imu)
        np.save('./dataset/CIP/work/{}_var_imus.npy'.format(name), self.var_imu)
        np.save('./dataset/CIP/work/{}_mean_motion.npy'.format(name), self.mean_motion)
        np.save('./dataset/CIP/work/{}_var_motion.npy'.format(name), self.var_motion)

        train_len = self.data_imu.shape[0] * 95 // 100      # 训练集占95%
        self.test_set_imu = self.data_imu[train_len:, ...]      # 测试集是后5%的部分（这里是验证集吧？）
        self.data_imu = self.data_imu[:train_len, ...]          # 提取训练集
        self.data_imu_reverse = torch.tensor(self.data_imu.numpy()[..., ::-1].copy())   # 时间倒序数据（数据增强用？）

        self.test_set_motion = self.data_motion[train_len:, ...]      # 测试集是后5%的部分（这里是验证集吧？）
        self.data_motion = self.data_motion[:train_len, ...]          # 提取训练集
        self.data_motion_reverse = torch.tensor(self.data_motion.numpy()[..., ::-1].copy())   # 时间倒序数据（数据增强用？）


        self.reset_length_flag = 0
        self.virtual_length = 0
        print('Window count: {}, total frame (without downsampling): {}'.format(len(self), self.total_frame))
        
        file = BVH_file(self.std_bvh)
        # file = BVH_file(get_std_bvh(dataset='Smpl'))       # std_bvh路径的file中存储了全部静态信息（预处理后的）
        # if i == 0:                                      # 对于这组模型（同类拓扑）的第一个模型，需要记录一下拓扑信息
        self.joint_topology = file.topology # 骨骼拓扑【就是每个节点的父节点编号】，[v]
        self.ee_ids = file.get_ee_id()        # 末端节点编号，[5]
        new_offset = file.offset                                    # 静态关节offset，[v,3]
        new_offset = torch.tensor(new_offset, dtype=torch.float)    # tensor[v,3]
        new_offset = new_offset.reshape((1,) + new_offset.shape)    # [1,v,3]
        self.offsets = new_offset.to(device)  # [1,v,3]

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data_motion.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data_motion.shape[0]
        # if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
        return [self.data_imu[item], self.data_motion[item]]
        # else:
        #     return self.data_reverse[item]
        
    def getValData(self):
        return [self.test_set_imu, self.test_set_motion]

    def get_windows_motion(self, motions): # 数据一律按照60fps处理
        new_windows = []

        for motion in motions:  # [t,C(87)] => [t,36]，方向(18)在前，加速度(18)在后
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
                # if self.args.rotation == 'quaternion':      # 如果用四元数表示旋转，
                #     new = new.reshape(new.shape[0], -1, 3)
                #     rotations = new[:, 0:6, :]              # 提取旋转部分【后一半表示的是加速度】
                #     rotations = Quaternions.from_euler(np.radians(rotations)).qs
                #     rotations = rotations.reshape(rotations.shape[0], 6, -1)   # 转换成四元数，再flatten
                #     # accelerations = new[:, -1, :]
                #     # accelerations = np.concatenate((new, np.zeros((new.shape[0], new.shape[1], 1))), axis=2)    # 位移也扩充为4维（末尾补0）【这一步没用上！】
                #     new = np.concatenate((rotations, new[:, 6:, :].reshape(new.shape[0], -1)), axis=1)      # 拼接rot+pos，最终维度为 4*(v-1)+3
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

        return torch.cat(new_windows)   # [n,t,C], C=4*6+3*6=42

    def get_windows_imu(self, motions): # 数据一律按照60fps处理
        new_windows = []

        for motion in motions:  # [t,42]
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
                
                new = new.reshape(new.shape[0], -1) # [t,6,7]=>[t,42]
                new = new[np.newaxis, ...]  # [t,C] -> [1,t,C]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)   # [n,t,C], C=4*6+3*6=42

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
    
    def denorm_imuData(self, imus):
        if self.args.normalization:
            if self.var_imu.device != imus.device:
                self.var_imu = self.var_imu.to(imus.device)
                self.mean_imu = self.mean_imu.to(imus.device)
            ans = imus * self.var_imu + self.mean_imu
        else: ans = imus
        return ans
    def denorm_motion(self, motion):
        if self.args.normalization:
            if self.var_motion.device != motion.device:
                self.var_motion = self.var_motion.to(motion.device)
                self.mean_motion = self.mean_motion.to(motion.device)
            ans = motion * self.var_motion + self.mean_motion
        else: ans = motion
        return ans
    def norm_motion(self, motion):
        if self.args.normalization:
            if self.var_motion.device != motion.device:
                self.var_motion = self.var_motion.to(motion.device)
                self.mean_motion = self.mean_motion.to(motion.device)
            ans = (motion - self.mean_motion) / self.var_motion
        else: ans = motion
        return ans

class testMotionData(Dataset):
    def __init__(self, args, std_path=None):
        super(testMotionData, self).__init__()
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')

        name = args.dataset     # 模型名，比如'Smpl'
        input_path = './dataset/CIP/work/{}_amass_test_imus_TP.npy'.format(name)
        file_path = './dataset/CIP/work/{}_amass_test_motion.npy'.format(name)     # 存储了所有bvh运动数据（归一化后）的npy文件

        if args.debug:
            file_path = file_path[:-4] + '_debug' + file_path[-4:]

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        if std_path is not None:
            self.std_bvh = std_path
        else:
            self.std_bvh = get_std_bvh(args)
        self.args = args
        self.data = []
        self.motion_length = []
        
        self.imu_data = np.load(input_path, allow_pickle=True)   # [n,t,6,7]
        self.motions = np.load(file_path, allow_pickle=True)     # [n,t,22,4]
        
        file = BVH_file(self.std_bvh)
        self.joint_topology = file.topology
        self.ee_ids = file.get_ee_id()
        
        new_offset = file.offset
        new_offset = torch.tensor(new_offset, dtype=torch.float)
        new_offset = new_offset.reshape((1,) + new_offset.shape)
        self.offsets = new_offset.to(self.device)
        
        mean_imu = np.load('./dataset/CIP/work/{}_mean_imus.npy'.format(name))
        var_imu = np.load('./dataset/CIP/work/{}_var_imus.npy'.format(name))
        mean_motion = np.load('./dataset/CIP/work/{}_mean_motion.npy'.format(name))
        var_motion = np.load('./dataset/CIP/work/{}_var_motion.npy'.format(name))
        mean_imu = torch.tensor(mean_imu).squeeze()
        mean_imu = mean_imu.reshape((1, ) + mean_imu.shape)
        var_imu = torch.tensor(var_imu).squeeze()
        var_imu = var_imu.reshape((1, ) + var_imu.shape)
        mean_motion = torch.tensor(mean_motion).squeeze()
        mean_motion = mean_motion.reshape((1, ) + mean_motion.shape)
        var_motion = torch.tensor(var_motion).squeeze()
        var_motion = var_motion.reshape((1, ) + var_motion.shape)
        self.mean_imu = mean_imu.to(self.device)
        self.var_imu = var_imu.to(self.device)
        self.mean_motion = mean_motion.to(self.device)
        self.var_motion = var_motion.to(self.device)
        
    def __getitem__(self, item):
        ref_shape = None
        
        new_data = self.get_item(item)  # [1,t,42], [1,t,87]
        new_imus = new_data[0]
        new_motion  = new_data[1]
        if new_motion is not None:
            # new_imus = (new_imus - self.mean_imu) / self.var_imu
            new_motion = (new_motion - self.mean_motion) / self.var_motion
            ref_shape = new_motion

        # if ref_shape is None:
        #     print('Bad at {}'.format(item))
        #     return None
        
        new_imus_ = new_imus.transpose(0,1)
        new_motion_ = new_motion.transpose(0,1)
        return [new_imus_, new_motion_]
    
    def __len__(self):
        return len(self.imu_data)

    def get_item(self, id):
        imus = self.imu_data[id]    # [t,6,7]
        
        # imus_item = torch.Tensor(imus).view(-1,42)    # 姿态估计
        imus_item = torch.Tensor(imus).view(-1,72)    # 姿态优化
        
        motion = self.motions[id]
        rot_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]
        parent_index = [0,1,2,3,0,5,6,7,0,9,10,11,12,11,14,15,16,11,18,19,20]
        
        motion = motion[:,rot_index] #[n,22,4]
        motion = motion[:, parent_index]
        motion = motion.reshape(motion.shape[0], -1)
        motion = np.concatenate((motion, np.zeros((motion.shape[0], 3))), axis=1)
        # motion = motion[np.newaxis, ...]  # [t,C] -> [1,t,87]
        motion_item = torch.tensor(motion, dtype=torch.float32)
        
        length = motion_item.shape[0]
        length = length // 4 * 4
        # res = [imus_item, motion_item]
        res = [imus_item[:length].to(self.device), motion_item[:length].to(self.device)]
        return res
        
    def denorm_imuData(self, imus):
        if self.args.normalization:
            if self.var_imu.device != imus.device:
                self.var_imu = self.var_imu.to(imus.device)
                self.mean_imu = self.mean_imu.to(imus.device)

            ans = imus * self.var_imu.unsqueeze(-1) + self.mean_imu.unsqueeze(-1)
        else: ans = imus
        return ans
    def denorm_motion(self, motion):
        if self.args.normalization:
            if self.var_motion.device != motion.device:
                self.var_motion = self.var_motion.unsqueeze(-1).to(motion.device)
                self.mean_motion = self.mean_motion.unsqueeze(-1).to(motion.device)
            ans = motion * self.var_motion.unsqueeze(-1) + self.mean_motion.unsqueeze(-1)
        else: ans = motion
        return ans
    def norm_motion(self, motion):
        if self.args.normalization:
            if self.var_motion.device != motion.device:
                self.var_motion = self.var_motion.unsqueeze(-1).to(motion.device)
                self.mean_motion = self.mean_motion.unsqueeze(-1).to(motion.device)
            ans = (motion - self.mean_motion.unsqueeze(-1)) / self.var_motion.unsqueeze(-1)
        else: ans = motion
        return ans
