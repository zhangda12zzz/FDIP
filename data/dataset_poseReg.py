import os
import numpy as np
import torch
from torch.utils.data import Dataset

from articulate.math.Quaternions import Quaternions
from config import paths, joint_set
import articulate as art


class ImuMotionData(Dataset):
    def __init__(self, args, std_path=None):
        super(ImuMotionData, self).__init__()
        self.args = args
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        body_model = art.ParametricModel(paths.smpl_file)

        if args.is_train:
            imu_path = ['dataset/CIP/tpFirst/Smpl_amass_imus_TP.npy', 'dataset/CIP/tpFirst/Smpl_dipTrain_imus_TP.npy', 'dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_imu_TP_train.npy']
            # joint_path = ['dataset/CIP/tpFirst/Smpl_amass_test_joints23_TP.npy']   # 换成joints_pos专门训练一下第三阶段网络
            pose_path = ['dataset/CIP/tpFirst/Smpl_amass_motion_SMPL24.npy', 'dataset/CIP/tpFirst/Smpl_dipTrain_motion_SMPL24.npy', 'dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_motion_SMPL24_train.npy']
            # imu_path = ['dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_imu_TP_train.npy']
            # pose_path = ['dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_motion_SMPL24_train.npy']
            # imu_path = ['dataset/CIP/tpFirst/Smpl_dipTrain_imus_TP.npy']
            # pose_path = ['dataset/CIP/tpFirst/Smpl_dipTrain_motion_SMPL24.npy']
        else:
            # dip
            # imu_path = ['dataset/CIP/tpFirst/Smpl_dip_imus_TP.npy']
            # pose_path = ['dataset/CIP/tpFirst/Smpl_dip_motion_SMPL24.npy']
            # tc
            # imu_path = ['dataset/CIP/tpFirst/Smpl_tc_imus_TP.npy']
            # pose_path = ['dataset/CIP/tpFirst/Smpl_tc_motion_SMPL24.npy']
            # mixamo
            # imu_path = ['dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_imu_TP.npy']
            imu_path = ['GGIP/data_all/Mixamo/mixamoSMPLPos_imu_test_withAccScale.npy']  # test
            # pose_path = ['dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_motion_SMPL24.npy']
            pose_path = ['GGIP/data_all/Mixamo/mixamoSMPLPos_motion_SMPL24_test.npy']
            joint_path = ['dataset/CIP/tpFirst/Smpl_amass_test_joints23_TP.npy']  # 没用上
    
        imu_data, pose_data = [], []
        for path in imu_path:
            imu = np.load(path, allow_pickle=True).tolist()   # [n,t,87]
            imu_wins = torch.Tensor(self.get_windows(imu))
            imu_data.append(imu_wins)
        for path in pose_path:  
            pose = np.load(path, allow_pickle=True).tolist()     # [n,t,24,3,3]
            pose_wins = torch.Tensor(self.get_windows(pose)) #[n,t,24,3,3]
            pose_data.append(pose_wins)
            
        # 关节位置
        # joint_data = []
        # for path in joint_path:  
        #     joint = np.load(path, allow_pickle=True).tolist()     # [n,t,23,3]
        #     joint_wins = torch.Tensor(self.get_windows(joint)) #[n,t,23,3]
        #     joint_data.append(joint_wins)
        # joint_windows = torch.concat(joint_data)
        # self.joints = joint_windows.to(device)
        
        # imu_windows = torch.Tensor(self.get_windows(imu_data)).to(device)
        # pose_windows = torch.Tensor(self.get_windows(pose_data)) #[n,t,24,3,3]
        
        # 调整比例
        # if args.is_train:
        #     smallest_length = imu_data[-1].shape[0]
            
        #     imu_data_most = imu_data[0].cpu().detach().numpy()
        #     np.random.shuffle(imu_data_most)
        #     imu_data[0] = torch.Tensor(imu_data_most[:smallest_length*9])
            
        #     pose_data_most = pose_data[0].cpu().detach().numpy()
        #     np.random.shuffle(pose_data_most)
        #     pose_data[0] = torch.Tensor(pose_data_most[:smallest_length*9])
            
        imu_windows = torch.concat(imu_data)
        pose_windows = torch.concat(pose_data)
        
        # 乱序
        pose_windows_ = pose_windows.view(pose_windows.shape[0], pose_windows.shape[1], -1)
        tmp = torch.concat((imu_windows, pose_windows_), dim=-1).cpu().detach().numpy()
        np.random.shuffle(tmp)
        imu_windows = torch.Tensor(tmp[:,:,:72]).to(device)
        pose_windows = torch.Tensor(tmp[:,:,72:]).view(pose_windows.shape[0], pose_windows.shape[1],24,3,3)
        
        self.poses_ref = pose_windows
        
        local_pose = pose_windows.view(-1,24,3,3)
        global_pose = body_model.forward_kinematics(local_pose, calc_mesh=False)[0]
        grot_reduce_raw = global_pose[:,joint_set.reduced]
        grot_reduce = global_pose[:,0:1].transpose(2, 3).matmul(grot_reduce_raw)
        grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
        for j in range(grot_reduce.shape[0]):
            grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])     # tensor[15, 6] 
        r6d_windows = grot_copy.contiguous().to(device)
        r6d_windows = r6d_windows.view(pose_windows.shape[0], -1, 15, 6)
        
        self.imus = imu_windows  #[n,t,87]
        root_windows = imu_windows[:,:,-9:].view(imu_windows.shape[0], imu_windows.shape[1],3,3)
        self.roots = root_windows
        self.poses = r6d_windows.view(r6d_windows.shape[0], r6d_windows.shape[1], 90) #[n,t,90]
        
        if args.is_train:
            train_len = self.imus.shape[0] * 5 // 100      # 训练集占95%
            self.test_imus = self.imus[:train_len, ...]      # 测试集是后5%的部分（这里是验证集吧？）
            self.imus = self.imus[train_len:, ...]          # 提取训练集
            self.test_poses = self.poses[:train_len, ...]      # 测试集是后5%的部分（这里是验证集吧？）
            self.poses = self.poses[train_len:, ...]          # 提取训练集
            self.test_roots = self.roots[:train_len, ...]
            self.roots = self.roots[train_len:, ...]
            self.test_poses_ref = self.poses_ref[:train_len, ...]
            self.poses_ref = self.poses_ref[train_len:, ...]
            # self.test_joints = self.joints[:train_len, ...]
            # self.joints = self.joints[train_len:, ...]
            

    def __len__(self):
        return self.imus.shape[0]
    
    def __getitem__(self, item):
        if isinstance(item, int): item %= self.imus.shape[0]
        # if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
        return [self.imus[item], self.poses[item]]
        # return [self.imus[item], self.joints[item], self.poses[item]]
    
    def getValData(self):
        if self.args.is_train:
            return [self.test_imus, self.test_poses, self.test_roots, self.test_poses_ref]
            # return [self.test_imus, self.test_joints, self.test_poses, self.test_roots, self.test_poses_ref]
        else:
            return [self.imus, self.poses, self.roots, self.poses_ref]
            # return [self.imus, self.joints, self.poses, self.roots, self.poses_ref]
    
    
    def get_windows(self, input):
        new_windows = []
        self.total_frame = 0
        for motion in input:  # [t,C,c,c,]
            motion = torch.Tensor(motion)
            self.total_frame += motion.shape[0]         # 统计总帧数
            
            step_size = self.args.window_size // 2      # 计算步长【类比conv过程的step！】
            window_size = step_size * 2                 # 偶数化window_size
            n_window = motion.shape[0] // window_size - 1 # 窗口数【让窗口覆盖整个时间序列、窗口间有重叠】
            
            for i in range(n_window):
                begin = i * step_size       # 窗口起始帧
                end = begin + window_size   # 窗口结束帧

                new = motion[begin:end, :]
                new = new[np.newaxis, ...]
                
                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)
        return torch.cat(new_windows)