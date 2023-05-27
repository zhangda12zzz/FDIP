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
        self.tran_raw = []
        self.joint_raw = []
        self.shape_raw = []
        self.vel_raw = []
        for data_folder in data_folders:
            self.pose_raw += torch.load(data_folder + '/pose.pt') # self.data_dict['pose']  # list[ tensor[[?, 24, 3]] ]          smpl姿态参数，24个节点的旋转
            self.acc_raw += torch.load(data_folder + '/vacc.pt') # self.data_dict['acc']    # list[ tensor[[?, 6, 3]] ]           6个节点的加速度             已经标准化
            self.ori_raw += torch.load(data_folder + '/vrot.pt') # self.data_dict['ori']    # list[ tensor[[?, 6, 3, 3]] ]           6个节点的方向               已经标准化
            self.grot_raw += torch.load(data_folder + '/grot.pt') # list[ tensor[[?, 24, 3, 3]] ] 
            self.tran_raw += torch.load(data_folder + '/tran.pt') # self.data_dict['tran']          # list[ tensor[[?, 3]] ]     根节点位置
            self.joint_raw += torch.load(data_folder + '/joint.pt') # self.data_dict['joint']        # list[ tensor[[?, 24, 3]] ]   24个关节的绝对位置（预处理已经加上了根节点）
            self.vel_raw += torch.load(data_folder + '/jvel.pt')
            # self.shape_raw += torch.load(data_folder + '/shape.pt') # self.data_dict['shape']        # list[ tensor[10] ]    smpl形状参数
        
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
        # self.shape = []
        self.grot = []
        self.jvel = []
        for i in range(len(self.acc_raw)):
            a_pose = art.math.axis_angle_to_rotation_matrix(self.pose_raw[i]).view(-1, 24, 3, 3)
            a_acc = self.acc_raw[i].view(-1, 6, 3)
            a_ori = self.ori_raw[i].view(-1, 6, 3, 3)
            a_tran = self.tran_raw[i].view(-1, 3)
            a_joint = self.joint_raw[i].view(-1, 24, 3)
            # a_shape = self.shape_raw[i].view(-1, 10)     # tensor[10]
            a_grot = self.grot_raw[i].view(-1,24,3,3)
            a_jvel = self.vel_raw[i].contiguous().view(-1, 72)
            # self.pose.append(a_pose)
            # self.acc.append(a_acc)
            # self.ori.append(a_ori)
            # self.tran.append(a_tran)
            # self.joint.append(a_joint)
            # # self.shape.append(a_shape)
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
                    # self.shape.append(a_shape)
                    self.grot.append(a_grot[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    self.jvel.append(a_jvel[self.max_frame_length * j:self.max_frame_length * (j+1),:])
            # 为了统一数据、将训练集全部切为了200帧的视频
            #     rest = a_seq_len - self.max_frame_length * count
            #     if rest > self.min_frame_length:
            #         self.pose.append(a_pose[self.max_frame_length * count:,:])
            #         self.acc.append(a_acc[self.max_frame_length * count:,:])
            #         self.ori.append(a_ori[self.max_frame_length * count:,:])
            #         self.tran.append(a_tran[self.max_frame_length * count:,:])
            #         self.joint.append(a_joint[self.max_frame_length * count:,:])
            #         # self.shape.append(a_shape)
            #         self.grot.append(a_grot[self.max_frame_length * count:,:])
            #         self.jvel.append(a_jvel[self.max_frame_length * count:,:])
            # elif a_seq_len > self.min_frame_length:
            #     self.pose.append(a_pose)
            #     self.acc.append(a_acc)
            #     self.ori.append(a_ori)
            #     self.tran.append(a_tran)
            #     self.joint.append(a_joint)
            #     # self.shape.append(a_shape)
            #     self.grot.append(a_grot)
            #     self.jvel.append(a_jvel)

    def __len__(self, sequence=False):
        res = len(self.acc)
        return res

    def __getitem__(self, index):
        r'''
            每次迭代返回如下信息：
            > 归一化后的 acc （6*3）                          v
            > 归一化后的 ori （6*9）                          v
            > 叶关节和根的相对位置 p_leaf （5*3）               v
            > 所有关节和根的相对位置 p_all （23*3）              v
            > 所有非根关节相对于根关节的 6D 旋转 pose （15*6）    
            > 根关节旋转 p_root （9）（就是ori）                v
            > 根关节位置 tran

            > 所有关节的速度 jvel（72）
            > 
        '''
        # >> out_acc, out_ori
        acc_cal = self.acc[index]                       # tensor[?, 6, 3]
        ori_cal = self.ori[index]                       # tensor[?, 6, 3, 3]
        acc_tmp = torch.cat((acc_cal[:, :5] - acc_cal[:, 5:], acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) #/ conf.acc_scale
        ori_tmp = torch.cat((ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5]), ori_cal[:, 5:]), dim=1)
        out_acc = acc_tmp.view(-1, 18)
        out_ori = ori_tmp.view(-1, 54)
        
        # >> out_p_leaf, out_p_all
        joint = self.joint[index]                       # tensor[?, 24, 3]
        p_all = torch.cat((joint[:,0:1], joint[:,1:]-joint[:,0:1]), dim=1)
        out_p_all = p_all[:, conf.joint_set.full, :]    # tensor[?, 23, 3]
        out_p_leaf = p_all[:, conf.joint_set.leaf, :]   # tensor[?, 5, 3]
        
        # >> out_grot
        grot_reduce_raw = self.grot[index][:,conf.joint_set.reduced]      # tensor[?,15,3,3]
        grot_reduce = self.grot[index][:,0:1].transpose(2, 3).matmul(grot_reduce_raw)   # tensor[?,15,3,3]
        grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
        for j in range(grot_reduce.shape[0]):
            grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])     # tensor[15, 6] 
        out_grot = grot_copy.contiguous()

        # >> out_r_root, out_tran, out_jvel
        out_r_root = self.pose[index][:,0]              # tensor[?, 9]
        out_tran = self.tran[index]                     # tensor[?, 3]
        out_jvel = self.jvel[index]                      # tensor[?, 72]
        
        # >> out_contact
        out_contact = None
        if True:    # 如果只使用AMASS / tc的数据
            mu = 0.008
            pos_lfoot = self.joint[index][:,7]      # [:,10]    # tensor[?,3]
            pos_rfoot = self.joint[index][:,8]      # [:,11]    # tensor[?,3]
            contact_lfoot = torch.Tensor([(1.0 if torch.norm(pos_lfoot[i] - pos_lfoot[i-1]) < mu else 0.0) for i in range(1, pos_lfoot.shape[0])])
            contact_lfoot = torch.cat((contact_lfoot[:1].clone(), contact_lfoot)).view(-1,1)
            contact_rfoot = torch.Tensor([(1.0 if torch.norm(pos_rfoot[i] - pos_rfoot[i-1]) < mu else 0.0) for i in range(1, pos_rfoot.shape[0])])
            contact_rfoot = torch.cat((contact_rfoot[:1].clone(), contact_rfoot)).view(-1,1)
            out_contact = torch.cat((contact_lfoot, contact_rfoot), dim=1)
            # print(contact_lfoot, contact_rfoot)

        out_pose_gt = None
        if True:
            out_pose_gt = self.pose[index]
        
        return out_acc, out_ori, out_p_leaf, out_p_all, out_grot, out_r_root, out_tran, out_jvel, out_contact, out_pose_gt

