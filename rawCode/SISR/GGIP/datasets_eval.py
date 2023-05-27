import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import paths, joint_set
import articulate as art
import option_parser
import random

class ImuMotionDataEval(Dataset):
    def __init__(self, args, std_path=None):
        super(ImuMotionDataEval, self).__init__()
        self.args = args
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.body_model = art.ParametricModel(paths.smpl_file)

        
        # dip
        # imu_path = 'GGIP/data_all/DIP_IMU/Smpl_dipTrain_imus_test.npy'
        # pose_path = 'GGIP/data_all/DIP_IMU/Smpl_dipTrain_motion_SMPL24_test.npy'
        # tc
        imu_path = 'GGIP/data_all/Totalcapture/Smpl_tc_imus_test.npy'
        pose_path = 'GGIP/data_all/Totalcapture/Smpl_tc_motion_SMPL24_test.npy'
        # mixamo
        # imu_path = 'GGIP/data_all/Mixamo/mixamoSMPLPos_imu_test.npy'
        # pose_path = 'GGIP/data_all/Mixamo/mixamoSMPLPos_motion_SMPL24_test.npy'
        # singleone-imu
        # imu_path = 'GGIP/data_all/SingleOne-IMU/Smpl_singleone_imus_test.npy'
        # pose_path = 'GGIP/data_all/SingleOne-IMU/Smpl_singleone_motion_SMPL24_test.npy'
            
        imu_raw = np.load(imu_path, allow_pickle=True) 
        pose_raw = np.load(pose_path, allow_pickle=True)
            
        imu_data, pose_data, root_data = [],[],[]
        
        for a_imu in imu_raw:
            a_imu = torch.Tensor(a_imu)
            imu_data.append(a_imu.clone())
            
            root_windows = a_imu[:,-9:].view(a_imu.shape[0], 3,3)
            root_data.append(root_windows)
        
        pose_data = self.smpl24ToR6dEval(pose_raw) 


        self.imus = imu_data    # [t,72]*n
        self.poses = pose_data  # [t,90]*n
        self.roots = root_data  # [t,3,3]*n
        
        

    def __len__(self):
        return self.imus.shape[0]
    
    def __getitem__(self, item):  # 必定是train环境
        if isinstance(item, int): item %= self.imus.shape[0]
        return [self.imus[item].to(self.device), self.joints[item].to(self.device), self.poses[item].to(self.device), self.poses[item].to(self.device), self.shapes[item].to(self.device)]
    
    def getValData(self):
        # self.idxCount += 1
        return [self.imus, self.poses, self.roots]
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
    
    def smpl24ToR6dEval(self, pose):
        '''
            pose: [n,t,24,3,3]
            >> return => r6d_windows:[n,t,15,6]
        '''
        res = []
        for a_pose in pose:
            a_pose = torch.Tensor(a_pose)
            t = a_pose.shape[0]
            a_pose = a_pose.view(-1,24,3,3) #[t,24,3,3]
            global_pose = self.body_model.forward_kinematics(a_pose, calc_mesh=False)[0]
            grot_reduce_raw = global_pose[:,joint_set.reduced]
            grot_reduce = global_pose[:,0:1].transpose(2, 3).matmul(grot_reduce_raw)
            grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
            for j in range(grot_reduce.shape[0]):
                grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])     # tensor[15, 6] 
            r6d_windows = grot_copy.contiguous()
            r6d_windows = r6d_windows.view(t,15*6)
            res.append(r6d_windows)
        return res    
    
if __name__ == '__main__':
    args = option_parser.get_args()
    device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
    args.dataset = 'Smpl'
    args.device = device
    dataset = ImuMotionDataEval(args)