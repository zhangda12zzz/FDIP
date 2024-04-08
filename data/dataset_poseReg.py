import os
import numpy as np
import torch
from torch.utils.data import Dataset
import copy

# from dataset.bvh_parser import BVH_file
# from dataset import get_test_set
# from option_parser import get_std_bvh
# from utils.Quaternions import Quaternions
from config import paths, joint_set
import articulate as art
import option_parser
import random

class ImuMotionData(Dataset):
    def __init__(self, args, std_path=None):
        super(ImuMotionData, self).__init__()
        self.args = args
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.body_model = art.ParametricModel(paths.smpl_file)

        if args.is_train:
            # pose_gan_ref_path = ['data/data_all/AMASS/Smpl_amass_motion_SMPL24.npy']
            # pose_gan_ref_data = []
            # for path in pose_gan_ref_path:
            #     pose_ref = np.load(path, allow_pickle=True).tolist()
            #     pose_ref_wins = torch.Tensor(self.get_windows(pose_ref))
            #     pose_gan_ref_data.append(pose_ref_wins)
            # pose_gan_ref_local = torch.concat(pose_gan_ref_data)     #[n,t,24,3,3]
            # self.pose_gan_ref = self.smpl24ToR6d(pose_gan_ref_local)
            # self.pose_gan_ref = self.pose_gan_ref.view(self.pose_gan_ref.shape[0], self.pose_gan_ref.shape[1], 90)
            
            # imu_path = ['data/data_all/AMASS/Smpl_amass_imus.npy']#, 'data/data_all/DIP_IMU/Smpl_dipTrain_imus.npy', 'data/data_all/SingleOne-IMU/Smpl_singleone_imus.npy']
            # joint_path = ['data/data_all/AMASS/Smpl_amass_joints23.npy']#, 'data/data_all/DIP_IMU/Smpl_dipTrain_joints23.npy', 'data/data_all/SingleOne-IMU/Smpl_singleone_joints23.npy']   # 换成joints_pos专门训练一下第三阶段网络
            # pose_path = ['data/data_all/AMASS/Smpl_amass_motion_SMPL24.npy']#, 'data/data_all/DIP_IMU/Smpl_dipTrain_motion_SMPL24.npy', 'data/data_all/SingleOne-IMU/Smpl_singleone_motion_SMPL24.npy']
            imu_path = ['data/data_all/DIP_IMU/Smpl_dipTrain_imus.npy', 'data/data_all/SingleOne-IMU/Smpl_singleone_imus.npy']
            joint_path = ['data/data_all/DIP_IMU/Smpl_dipTrain_joints23.npy', 'data/data_all/SingleOne-IMU/Smpl_singleone_joints23.npy']   # 换成joints_pos专门训练一下第三阶段网络
            pose_path = ['data/data_all/DIP_IMU/Smpl_dipTrain_motion_SMPL24.npy', 'data/data_all/SingleOne-IMU/Smpl_singleone_motion_SMPL24.npy']
        else:
            # dip
            imu_path = ['data/data_all/DIP_IMU/Smpl_dipTrain_imus_test.npy']
            pose_path = ['data/data_all/DIP_IMU/Smpl_dipTrain_motion_SMPL24_test.npy']
            # tc
            # imu_path = ['data/data_all/Totalcapture/Smpl_tc_imus_test.npy']
            # pose_path = ['data/data_all/Totalcapture/Smpl_tc_motion_SMPL24_test.npy']
            # mixamo
            # imu_path = ['data/data_all/Mixamo/mixamoSMPLPos_imu_test.npy']
            # pose_path = ['data/data_all/Mixamo/mixamoSMPLPos_motion_SMPL24_test.npy']
            # singleone-imu
            # imu_path = ['data/data_all/SingleOne-IMU/Smpl_singleone_imus_test.npy']
            # pose_path = ['data/data_all/SingleOne-IMU/Smpl_singleone_motion_SMPL24_test.npy']
            
            joint_path = ['dataset/CIP/tpFirst/Smpl_amass_test_joints23_TP.npy']  # 没用上
    
        imu_data, pose_data, shape_data = [], [], []
        for path in imu_path:
            imu = np.load(path, allow_pickle=True).tolist()   # [n,t,87]
            imu_wins = torch.Tensor(self.get_windows(imu))
            imu_data.append(imu_wins)
        
        for path in pose_path:
            if args.is_train:  
                pose, shape = np.load(path, allow_pickle=True).tolist()     # [n,t,24,3,3]+[n,t,10]
                shape_wins = torch.Tensor(self.get_windows(shape)) #[n,t,24,3,3]
                shape_data.append(shape_wins)
            else:
                pose = np.load(path, allow_pickle=True).tolist()
            pose_wins = torch.Tensor(self.get_windows(pose)) #[n,t,24,3,3]
            pose_data.append(pose_wins)
            
        # 关节位置
        joint_data = []
        for path in joint_path:  
            joint = np.load(path, allow_pickle=True).tolist()     # [n,t,23,3]
            joint_wins = torch.Tensor(self.get_windows(joint)) #[n,t,23,3]
            joint_data.append(joint_wins)
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
        joint_windows = torch.concat(joint_data)
        if args.is_train:
            shape_windows = torch.concat(shape_data)    # [n,t,10]?
        
        # 乱序
        pose_windows_ = pose_windows.view(pose_windows.shape[0], pose_windows.shape[1], -1)
        joint_windows_ = joint_windows.view(joint_windows.shape[0], joint_windows.shape[1], -1)
        if args.is_train:
            tmp = torch.concat((imu_windows, joint_windows_, pose_windows_, shape_windows), dim=-1).cpu().detach().numpy()
            np.random.shuffle(tmp)
            imu_windows = torch.Tensor(tmp[:,:,:72])
            joint_windows = torch.Tensor(tmp[:,:,72:72+69])
            pose_windows = torch.Tensor(tmp[:,:,72+69:-10]).view(pose_windows.shape[0], pose_windows.shape[1],24,3,3)
            shape_windows = torch.Tensor(tmp[:,:,-10:])
        else:
            tmp = torch.concat((imu_windows, pose_windows_), dim=-1).cpu().detach().numpy()
            np.random.shuffle(tmp)
            imu_windows = torch.Tensor(tmp[:,:,:72])
            pose_windows = torch.Tensor(tmp[:,:,72:]).view(pose_windows.shape[0], pose_windows.shape[1],24,3,3)
        
        self.poses_ref = pose_windows
        self.joints = joint_windows     #[n,t,69]
        if args.is_train:
            self.shapes = shape_windows
        
        r6d_windows = self.smpl24ToR6d(pose_windows)    # 针对6d版本
        # mat_windows = self.smpl24ToMat(pose_windows)    # 针对9d版本
        
        # local_pose = pose_windows.view(-1,24,3,3)
        # global_pose = body_model.forward_kinematics(local_pose, calc_mesh=False)[0]
        # grot_reduce_raw = global_pose[:,joint_set.reduced]
        # grot_reduce = global_pose[:,0:1].transpose(2, 3).matmul(grot_reduce_raw)
        # grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
        # for j in range(grot_reduce.shape[0]):
        #     grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])     # tensor[15, 6] 
        # r6d_windows = grot_copy.contiguous()
        # r6d_windows = r6d_windows.view(pose_windows.shape[0], -1, 15, 6)
        
        self.imus = imu_windows  #[n,t,72]
        root_windows = imu_windows[:,:,-9:].view(imu_windows.shape[0], imu_windows.shape[1],3,3)
        self.roots = root_windows
        self.poses = r6d_windows.view(r6d_windows.shape[0], r6d_windows.shape[1], 90) #[n,t,90]
        # self.poses = mat_windows.view(mat_windows.shape[0], mat_windows.shape[1], 15*9) #[n,t,90]
        
        if args.is_train:
            train_len = self.imus.shape[0] * 5 // 100      # 训练集占95%
            self.test_imus = self.imus[:train_len, ...]      # 测试集是后5%的部分（这里是验证集吧？）
            self.imus = self.imus          # 提取训练集
            self.test_poses = self.poses[:train_len, ...]      # 测试集是后5%的部分（这里是验证集吧？）
            self.poses = self.poses          # 提取训练集
            self.test_roots = self.roots[:train_len, ...]
            self.roots = self.roots
            self.test_poses_ref = self.poses_ref[:train_len, ...]
            self.poses_ref = self.poses_ref
            self.test_joints = self.joints[:train_len, ...]
            self.joints = self.joints
            self.test_shapes = self.shapes[:train_len, ...]
            self.shapes = self.shapes
            
            # self.idxCount = 0
            # self.gan_ref_order = [x for x in range(0, self.pose_gan_ref.shape[0]-1)]
            # random.shuffle(self.gan_ref_order)
            
        
            

    def __len__(self):
        return self.imus.shape[0]
    
    def __getitem__(self, item):  # 必定是train环境
        if isinstance(item, int): item %= self.imus.shape[0]
        # if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
        
        # return [self.imus[item].to(self.device), self.joints[item].to(self.device), self.poses[item].to(self.device), self.pose_gan_ref[self.gan_ref_order[item]].to(self.device)]
        return [self.imus[item].to(self.device), self.joints[item].to(self.device), self.poses[item].to(self.device), self.poses[item].to(self.device), self.shapes[item].to(self.device)]
    
    def getValData(self):
        # self.idxCount += 1
        if self.args.is_train:
            # random.shuffle(self.gan_ref_order)
            return [self.test_imus.to(self.device), self.test_joints.to(self.device), self.test_poses.to(self.device), self.test_roots.to(self.device), self.test_poses_ref.to(self.device)]
            # return [self.test_imus, self.test_joints, self.test_poses, self.test_roots, self.test_poses_ref]
        else:
            return [self.imus.to(self.device), self.poses.to(self.device), self.roots.to(self.device), self.poses_ref.to(self.device)]
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
    
    def smpl24ToR6d(self, pose):
        '''
            pose: [n,t,24,3,3]
            >> return => r6d_windows:[n,t,15,6]
        '''
        n,t,_,_,_ = pose.shape
        pose = pose.view(-1,24,3,3)
        global_pose = self.body_model.forward_kinematics(pose, calc_mesh=False)[0]
        grot_reduce_raw = global_pose[:,joint_set.reduced]
        grot_reduce = global_pose[:,0:1].transpose(2, 3).matmul(grot_reduce_raw)
        grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
        for j in range(grot_reduce.shape[0]):
            grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])     # tensor[15, 6] 
        r6d_windows = grot_copy.contiguous()
        r6d_windows = r6d_windows.view(n,t,15,6)
        return r6d_windows
    
    def smpl24ToMat(self, pose):
        '''
            pose: [n,t,24,3,3]
            >> return => r6d_windows:[n,t,15,9]
        '''
        n,t,_,_,_ = pose.shape
        pose = pose.view(-1,24,3,3)
        global_pose = self.body_model.forward_kinematics(pose, calc_mesh=False)[0]
        grot_reduce_raw = global_pose[:,joint_set.reduced]
        grot_reduce = global_pose[:,0:1].transpose(2, 3).matmul(grot_reduce_raw)
        mat_windows = grot_reduce.view(n,t,15,9)
        return mat_windows
        
    
if __name__ == '__main__':
    args = option_parser.get_args()
    device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
    args.dataset = 'Smpl'
    args.device = device
    dataset = ImuMotionData(args)