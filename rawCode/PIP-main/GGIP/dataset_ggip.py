import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import articulate as art
import config as conf


class ImuDataset(Dataset):
    def __init__(self, data_folders):
        r'''
            从 pt 文件中读取 acc、ori、pose、tran、joint、shape、statistic、dynamic、seq 信息。
        '''

        # 基本数据
        self.pose_raw = []
        self.acc_raw = []
        self.ori_raw = []
        self.grot_raw = []
        self.grot_axis_raw = []
        self.grot_euler_raw = []
        self.grot_quat_raw = []
        self.tran_raw = []
        self.joint_raw = []
        self.beta_raw = []
        self.vel_raw = []
        for data_folder in data_folders:
            self.pose_raw += torch.load(data_folder + '/pose.pt') # self.data_dict['pose']  # list[ tensor[[?, 24, 3]] ]          smpl姿态参数，24个节点的旋转
            self.acc_raw += torch.load(data_folder + '/vacc.pt') # self.data_dict['acc']    # list[ tensor[[?, 6, 3]] ]           6个节点的加速度             已经标准化
            self.ori_raw += torch.load(data_folder + '/vrot.pt') # self.data_dict['ori']    # list[ tensor[[?, 6, 3, 3]] ]           6个节点的方向               已经标准化
            self.grot_raw += torch.load(data_folder + '/grot.pt') # list[ tensor[[?, 24, 3, 3]] ] 
            self.grot_axis_raw += torch.load(data_folder + '/grot_axis.pt') # list[ tensor[[?, 24, 3, 3]] ] 
            self.grot_euler_raw += torch.load(data_folder + '/grot_euler.pt') # list[ tensor[[?, 24, 3, 3]] ] 
            self.grot_quat_raw += torch.load(data_folder + '/grot_quat.pt') # list[ tensor[[?, 24, 3, 3]] ] 
            self.tran_raw += torch.load(data_folder + '/tran.pt') # self.data_dict['tran']          # list[ tensor[[?, 3]] ]     根节点位置
            self.joint_raw += torch.load(data_folder + '/joint.pt') # self.data_dict['joint']        # list[ tensor[[?, 24, 3]] ]   24个关节的绝对位置（预处理已经加上了根节点）
            self.vel_raw += torch.load(data_folder + '/jvel.pt')
            self.beta_raw += torch.load(data_folder + '/shape.pt') # self.data_dict['shape']        # list[ tensor[10] ]    smpl形状参数
        
        # # 貌似不需要额外进行标准化的样子（batch normalization）
        # self.statistic = self.data_dict['statistic']
        
        # 对过长的序列进行一些裁剪操作
        self.max_frame_length = 200
        self.min_frame_length = 150
        self.pose = []
        self.acc = []
        self.ori = []
        self.tran = []
        self.joint = []
        self.beta = []
        self.grot = []
        self.grot_axis = []
        self.grot_euler = []
        self.grot_quat = []
        self.jvel = []
        for i in range(len(self.acc_raw)):
            a_pose = art.math.axis_angle_to_rotation_matrix(self.pose_raw[i]).view(-1, 24, 3, 3)
            a_acc = self.acc_raw[i].view(-1, 6, 3)
            a_ori = self.ori_raw[i].view(-1, 6, 3, 3)
            a_tran = self.tran_raw[i].view(-1, 3)
            a_joint = self.joint_raw[i].view(-1, 24, 3)
            a_shape = self.beta_raw[i].view(-1, 10)     # tensor[10]
            a_grot = self.grot_raw[i].view(-1,24,3,3)
            a_grot_axis = self.grot_axis_raw[i].view(-1,24,3)
            a_grot_euler = self.grot_euler_raw[i].view(-1,24,3)
            a_grot_quat = self.grot_quat_raw[i].view(-1,24,4)
            a_jvel = self.vel_raw[i].contiguous().view(-1, 72)
            # self.pose.append(a_pose)
            # self.acc.append(a_acc)
            # self.ori.append(a_ori)
            # self.tran.append(a_tran)
            # self.joint.append(a_joint)
            # # self.beta.append(a_shape)
            # self.grot.append(a_grot)
            # self.jvel.append(a_jvel)
            
            a_seq_len = a_acc.shape[0]
            if a_seq_len > self.max_frame_length:
                count = a_seq_len // self.max_frame_length
                for j in range(count):
                    self.pose.append(a_pose[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.acc.append(a_acc[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.ori.append(a_ori[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.tran.append(a_tran[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.joint.append(a_joint[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.beta.append(a_shape[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.grot.append(a_grot[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.grot_axis.append(a_grot_axis[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.grot_euler.append(a_grot_euler[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.grot_quat.append(a_grot_quat[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.jvel.append(a_jvel[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                rest = a_seq_len - self.max_frame_length * count
                # if rest > self.min_frame_length:
                #     self.pose.append(a_pose[self.max_frame_length * count:,:])
                #     self.acc.append(a_acc[self.max_frame_length * count:,:])
                #     self.ori.append(a_ori[self.max_frame_length * count:,:])
                #     self.tran.append(a_tran[self.max_frame_length * count:,:])
                #     self.joint.append(a_joint[self.max_frame_length * count:,:])
                #     # self.beta.append(a_shape)
                #     self.grot.append(a_grot[self.max_frame_length * count:,:])
                #     self.jvel.append(a_jvel[self.max_frame_length * count:,:])
            # elif a_seq_len > self.min_frame_length:
            #     self.pose.append(a_pose)
            #     self.acc.append(a_acc)
            #     self.ori.append(a_ori)
            #     self.tran.append(a_tran)
            #     self.joint.append(a_joint)
            #     # self.beta.append(a_shape)
            #     self.grot.append(a_grot)
            #     self.jvel.append(a_jvel)

    def __len__(self, sequence=False):
        res = len(self.acc)
        return res

    def __getitem__(self, index):

        # >> out_acc, out_ori，原本的顺序是右手左手、右脚左脚、头、根 => 根、左右脚、头、左右手
        acc_cal = self.acc[index]                       # tensor[?, 6, 3]
        ori_cal = self.ori[index]                       # tensor[?, 6, 3, 3]
        order = [2,3,4,0,1,5]   # 左右脚、头、左右手、根
        # order = [5,2,3,4,0,1]   # 根、左右脚、头、左右手
        acc_cal = acc_cal[:,order]
        ori_cal = ori_cal[:,order]
        
        # 通过这一步，变成了根、左右脚、头、左右手的顺序
        acc_tmp = torch.cat((acc_cal[:, 5:], acc_cal[:, :5] - acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) #/ conf.acc_scale
        ori_tmp = torch.cat((ori_cal[:, 5:], ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5])), dim=1)
        ori_tmp_ = art.math.rotation_matrix_to_quat(ori_tmp).view(-1,6,4)

        out_acc = acc_tmp.view(-1, 6, 3)
        out_ori_quat = ori_tmp_.view(-1, 6, 4)  # 用四元数表达旋转
        out_ori = ori_tmp.view(-1, 6, 9)
        
        # >> out_p_leaf, out_p_all
        joint = self.joint[index]                       # tensor[?, 24, 3]
        p_all_includeRoot = torch.cat((joint[:,0:1], joint[:,1:]-joint[:,0:1]), dim=1)
        out_leaf_pos = p_all_includeRoot[:, conf.joint_set.leaf]
        out_all_pos = p_all_includeRoot[:, conf.joint_set.full]
        
        
        # >> out_grot
        grot_reduce_raw = self.grot[index][:,conf.joint_set.reduced]
        grot_reduce = self.grot[index][:,0:1].transpose(2, 3).matmul(grot_reduce_raw)   # [1,15,3,3]，全局相对旋转
        
        grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
        # grot_copy_axis = torch.zeros(grot_reduce.shape[0], 15, 3)
        # grot_copy_eular = torch.zeros(grot_reduce.shape[0], 15, 3)
        for j in range(grot_reduce.shape[0]):
            grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])     # tensor[15, 6] 
            # grot_copy_axis[j] = art.math.rotation_matrix_to_axis_angle(grot_reduce[j])     # tensor[15, 3] 
            # grot_copy_eular[j] = art.math.rotation_matrix_to_euler_angle(grot_reduce[j])     # tensor[15, 3] 
        out_grot = grot_copy.contiguous()
        
        out_grot_axis = self.grot_axis[index][:,conf.joint_set.reduced]
        out_grot_euler = self.grot_euler[index][:,conf.joint_set.reduced]
        out_grot_quat = self.grot_quat[index][:,conf.joint_set.reduced]
        
        out_pose = self.pose[index][:,conf.joint_set.reduced] # [24,3,3]
        out_shape = self.beta[index]
        if min(out_shape.shape) == 0:
            out_shape = torch.zeros(1,10)
        if max(out_shape.shape) > 10:
            # print(out_shape)
            out_shape = torch.zeros(1,10)
            
        # return out_acc, out_ori, out_leaf_pos, out_all_pos, out_grot, grot_reduce, out_grot_axis, out_grot_euler, out_grot_quat
        return out_acc, out_ori, out_leaf_pos, out_all_pos, out_grot, grot_reduce, out_pose, out_shape

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # train_data_folder = ['data/dataset_work/AMASS/train']
    train_data_folder = ['data/dataset_work/DIP_IMU/train']
    val_data_folder = ["data/dataset_work/TotalCapture/train"]
    train_dataset = ImuDataset(train_data_folder)
    val_dataset = ImuDataset(val_data_folder)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    for batch_idx, data in enumerate(train_loader):
        acc = data[0].to(device).float()                # [batch_size, max_seq, 6, 3]
        # acc = acc.view(-1, acc.shape[1], 18)
        ori = data[7].to(device).float()                # [batch_size, max_seq, 6, 9]
        p_graphB = data[2].to(device).float()             # [batch_size, max_seq, 6, 3]
        p_graphP = data[3].to(device).float()       # [n,t,10,3]
        ori_graphP = data[8].to(device).float()       # [n,t,10,9]
        ori_graphJ = data[9].to(device).float()       # [n,t,16,9]
        p_graphJ = data[6].to(device).float()       # [n,t,16,3]
        
        print(ori)
