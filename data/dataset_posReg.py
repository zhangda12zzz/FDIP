import torch
from torch.utils.data import DataLoader, Dataset  #从原数据集中选取子集

import articulate as art
import config as conf

"""
数据加载：
从指定文件夹中加载预处理好的 IMU 数据（如加速度、方向、关节位置等）。
数据以 .pt 文件存储，包含以下内容：
pose.pt：SMPL 模型的姿态参数（24 个关节的轴角表示）。
vacc.pt：6 个 IMU 传感器的加速度数据。
vrot.pt：6 个 IMU 传感器的方向数据（旋转矩阵）。
其他数据如全局旋转、关节位置、形状参数等。

数据预处理：
序列截断：将过长的序列（超过 max_frame_length=200 帧）分割为多个子序列，确保数据长度符合模型输入要求。
数据重组：调整传感器数据的顺序（如将根节点放在首位），并转换坐标系（如通过矩阵乘法对加速度和方向进行坐标系对齐）。
旋转表示转换：将旋转矩阵转换为四元数（out_ori_quat）或 6D 旋转表示（out_grot），便于模型处理。

数据提供：
通过 __getitem__ 方法返回处理后的数据，包括：
输入数据：IMU 的加速度和方向（out_acc, out_ori）。
目标数据：叶关节的相对位置（out_leaf_pos）。
辅助信息：全局旋转、关节姿态、形状参数等。
"""

class ImuDataset(Dataset):
    def __init__(self, data_folders):
        r'''
            从 pt 文件中读取 acc、ori、pose、tran、joint、shape、statistic、dynamic、seq 信息。
        '''

        # 基本数据
        self.pose_raw = []
        self.acc_raw = []
        self.ori_raw = []
        # self.grot_raw = []
        # self.grot_axis_raw = []
        # self.grot_euler_raw = []
        # self.grot_quat_raw = []
        self.tran_raw = []
        self.joint_raw = []
        self.beta_raw = []
        # self.vel_raw = []
        for data_folder in data_folders:
            self.pose_raw += torch.load(data_folder + '/pose.pt') # self.data_dict['pose']  # list[ tensor[[?, 24, 3, 3]] ]          smpl姿态参数，24个节点的旋转
            self.acc_raw += torch.load(data_folder + '/vacc.pt') # self.data_dict['acc']    # list[ tensor[[?, 6, 3]] ]           6个节点的加速度             已经标准化
            self.ori_raw += torch.load(data_folder + '/vrot.pt') # self.data_dict['ori']    # list[ tensor[[?, 6, 3, 3]] ]           6个节点的方向               已经标准化

            # self.grot_raw += torch.load(data_folder + '/grot.pt') # list[ tensor[[?, 24, 3, 3]] ]
            # self.grot_axis_raw += torch.load(data_folder + '/grot_axis.pt') # list[ tensor[[?, 24, 3, 3]] ]
            # self.grot_euler_raw += torch.load(data_folder + '/grot_euler.pt') # list[ tensor[[?, 24, 3, 3]] ]
            # self.grot_quat_raw += torch.load(data_folder + '/grot_quat.pt') # list[ tensor[[?, 24, 3, 3]] ]

            self.tran_raw += torch.load(data_folder + '/tran.pt') # self.data_dict['tran']          # list[ tensor[[?, 3]] ]     根节点位置
            self.joint_raw += torch.load(data_folder + '/joint.pt') # self.data_dict['joint']        # list[ tensor[[?, 24, 3]] ]   24个关节的绝对位置（预处理已经加上了根节点）
            #self.vel_raw += torch.load(data_folder + '/jvel.pt')    #关节速度信息

            self.beta_raw += torch.load(data_folder + '/shape.pt') # self.data_dict['shape']        # list[ tensor[10] ]    smpl形状参数
        
        # # 貌似不需要额外进行标准化的样子（batch normalization）
        # self.statistic = self.data_dict['statistic']
        
        # 对过长的序列进行一些裁剪操作
        self.max_frame_length = 120
        self.stride = 60  # 每次移动的步长
        self.min_frame_length = 60
        self.pose = []
        self.acc = []
        self.ori = []
        self.tran = []
        self.joint = []
        self.beta = []
        # self.grot = []
        # self.grot_axis = []
        # self.grot_euler = []
        # self.grot_quat = []
        # self.jvel = []

        for i in range(len(self.acc_raw)):
            a_pose = self.pose_raw[i].view(-1, 24, 3, 3)
            a_acc = self.acc_raw[i].view(-1, 6, 3)
            a_ori = self.ori_raw[i].view(-1, 6, 3, 3)
            a_tran = self.tran_raw[i].view(-1, 3)
            a_joint = self.joint_raw[i].view(-1, 24, 3)
            a_shape = self.beta_raw[i].view(-1, 10)

            # 使用滑动窗口进行序列分割
            a_seq_len = a_acc.shape[0]
            if a_seq_len > self.max_frame_length:
                # 计算可以滑动的次数
                slide_count = (a_seq_len - self.max_frame_length) // self.stride + 1

                end = 0
                # 使用滑动窗口切分序列
                for j in range(slide_count + 1):  # +1 确保最后一个窗口也被处理
                    start_idx = min(j * self.stride, a_seq_len - self.max_frame_length)
                    end_idx = start_idx + self.max_frame_length
                    end = end_idx
                    # 如果已经到达序列末尾，就退出循环
                    if start_idx >= a_seq_len:
                        break

                    self.pose.append(a_pose[start_idx:end_idx, :])
                    self.acc.append(a_acc[start_idx:end_idx, :])
                    self.ori.append(a_ori[start_idx:end_idx, :])
                    self.tran.append(a_tran[start_idx:end_idx, :])
                    self.joint.append(a_joint[start_idx:end_idx, :])
                    self.beta.append(a_shape[start_idx:end_idx, :])

                # print("最后一个窗口的长度：" , self.pose[-1].shape,"最后：",end)
                    # self.grot.append(self.grot_raw[i][start_idx:end_idx, :])
            elif a_seq_len >= self.min_frame_length:
                # 300 到 500 帧：补零到 500 帧
                pad_length = self.max_frame_length - a_seq_len
                self.pose.append(torch.cat([a_pose, torch.zeros(pad_length, 24, 3, 3)], dim=0))
                self.acc.append(torch.cat([a_acc, torch.zeros(pad_length, 6, 3)], dim=0))
                self.ori.append(torch.cat([a_ori, torch.zeros(pad_length, 6, 3, 3)], dim=0))
                self.tran.append(torch.cat([a_tran, torch.zeros(pad_length, 3)], dim=0))
                self.joint.append(torch.cat([a_joint, torch.zeros(pad_length, 24, 3)], dim=0))
                self.beta.append(torch.cat([a_shape, torch.zeros(pad_length, 10)], dim=0))
                    # self.grot.append(a_grot[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    # self.grot_axis.append(a_grot_axis[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    # self.grot_euler.append(a_grot_euler[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    # self.grot_quat.append(a_grot_quat[self.max_frame_length * j:self.max_frame_length * (j+1),:])
                    # self.jvel.append(a_jvel[self.max_frame_length * j:self.max_frame_length * (j+1),:])

                # 剩余部分：序列长度为 300 帧时，分割为 1 个完整的 200 帧子序列。剩余150帧
                #rest = a_seq_len - self.max_frame_length * count
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

        # # >> 转换坐标系并生成6D旋转表示
        # # 将根节点的旋转作为全局坐标系基准
        # root_rot = ori_cal[:, -1]  # 根节点的旋转矩阵（3x3）
        # # 调整加速度坐标系：相对根节点的加速度 + 根节点方向
        # acc_tmp = torch.cat((acc_cal[:, 5:], acc_cal[:, :5] - acc_cal[:, 5:]), dim=1).bmm(root_rot.unsqueeze(1))
        # out_acc = acc_tmp.view(-1, 6, 3)
        #
        # # 转换方向为6D表示（取旋转矩阵的前两列并展平）
        # out_rot_6d = ori_cal[..., :2].view(ori_cal.shape[0], ori_cal.shape[1], 6)

        # 通过这一步，变成了根、左右脚、头、左右手的顺序    --- 进行坐标系转换（全局-根节点）
        acc_tmp = torch.cat((acc_cal[:, 5:], acc_cal[:, :5] - acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) #/ conf.acc_scale
        ori_tmp = torch.cat((ori_cal[:, 5:], ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5])), dim=1)
        #ori_tmp_ = art.math.rotation_matrix_to_quat(ori_tmp).view(-1,6,4) # 用四元数表达旋转

        out_acc = acc_tmp.view(-1, 6, 3)
        #out_ori_quat = ori_tmp_.view(-1, 6, 4)  # 用四元数表达旋转
        out_ori = ori_tmp.view(-1, 6, 9)

        # 转换为6D向量
        rot_6d = art.math.rotation_matrix_to_r6d(ori_tmp)  # 形状：(seq_len*6, 6)

        # 恢复为 (seq_len, 6, 6)
        out_rot_6d = rot_6d.view(ori_tmp.shape[0], ori_tmp.shape[1], 6)  # 最终形状：(seq_len, 6, 6)



        # >>叶子节点和全节点位置相对位置（根坐标系）
        joint = self.joint[index]                       # tensor[?, 24, 3]
        p_all_includeRoot = torch.cat((joint[:,0:1], joint[:,1:]-joint[:,0:1]), dim=1)
        out_leaf_pos = p_all_includeRoot[:, conf.joint_set.leaf]
        out_all_pos = p_all_includeRoot[:, conf.joint_set.full]
        
        #
        # # >> 旋转数据
        # grot_reduce_raw = self.grot[index][:,conf.joint_set.reduced] #全局
        # grot_reduce = self.grot[index][:,0:1].transpose(2, 3).matmul(grot_reduce_raw)   # [1,15,3,3]，全局相对旋转
        #
        # grot_copy = torch.zeros(grot_reduce.shape[0], 15, 6)
        # # grot_copy_axis = torch.zeros(grot_reduce.shape[0], 15, 3)
        # # grot_copy_eular = torch.zeros(grot_reduce.shape[0], 15, 3)

        # 全局相对旋转转为r6d
        # for j in range(grot_reduce.shape[0]):
        #     grot_copy[j] = art.math.rotation_matrix_to_r6d(grot_reduce[j])     # tensor[15, 6]
        #     # grot_copy_axis[j] = art.math.rotation_matrix_to_axis_angle(grot_reduce[j])     # tensor[15, 3]
        #     # grot_copy_eular[j] = art.math.rotation_matrix_to_euler_angle(grot_reduce[j])     # tensor[15, 3]
        # out_grot = grot_copy.contiguous()

        '''
        out_grot_axis：轴角表示法
        out_grot_euler：欧拉角表示法
        out_grot_quat：四元数表示法
        '''
        # out_grot_axis = self.grot_axis[index][:,conf.joint_set.reduced]
        # out_grot_euler = self.grot_euler[index][:,conf.joint_set.reduced]
        # out_grot_quat = self.grot_quat[index][:,conf.joint_set.reduced]

        out_pose = self.pose[index]# [15,3,3]
        #out_pose = self.pose[index][:,conf.joint_set.reduced] # [15,3,3]
        out_pose_6d = art.math.rotation_matrix_to_r6d(out_pose).view(out_pose.shape[0], out_pose.shape[1], 6)

        out_shape = self.beta[index]
        if min(out_shape.shape) == 0:
            out_shape = torch.zeros(1,10)
        if max(out_shape.shape) > 10:
            # print(out_shape)
            out_shape = torch.zeros(1,10)
            
        # return out_acc, out_ori, out_leaf_pos, out_all_pos, out_grot, grot_reduce, out_grot_axis, out_grot_euler, out_grot_quat
        return out_acc, out_ori, out_rot_6d, out_leaf_pos, out_all_pos,  out_pose, out_pose_6d, out_shape

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # train_data_folder = ['data/dataset_work/AMASS/train']
    train_data_folder = ["D:\Dataset\SingleOne\Pt"]
    #val_data_folder = ["data/dataset_work/TotalCapture/train"]
    train_dataset = ImuDataset(train_data_folder)
    #val_dataset = ImuDataset(val_data_folder)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    print(len(train_dataset))
    
    for batch_idx, data in enumerate(train_loader):
        # __getitem__返回格式确认
        #acc = data[0].to(device).float()

        #ori = data[1].to(device).float()

        #out_rot_6d = data[2].to(device).float()

        #out_leaf_pos = data[3].to(device).float()

        out_all_pos = data[4].to(device).float()

        #out_pos = data[5].to(device).float()     #(15,3,3)

        #leaf_pos = data[3].to(device).float()

        #out_shape = data[6].to(device).float()


        print("组数：", batch_idx, "数据：", out_all_pos.shape)
        #print(ori)
