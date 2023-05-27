import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用
import pickle
import torch
import numpy as np
from tqdm import tqdm
import glob

import articulate as art
from config import *
import config as conf

body_model = art.ParametricModel(paths.smpl_file)

# 对于单个pkl文件
# data = torch.load('dataset/PNTrial/SingleOne-IMU/work/s6.pkl')
# acc = data['acc']
# ori = data['ori']
# pose = data['pose']
# tran = data['tran']
# res = {'acc': acc, 'ori': ori, 'pose': pose, 'tran': tran}
# torch.save(res, 'dataset/PNTrial/SingleOne-IMU/work/s6.pt') # 问题很大！

out_joint, out_gt_pose, out_imus, out_shape = [], [], [], []

for subject_name in ['s2', 's6', 's4', 's5']:     # 构造训练集的部分
# for subject_name in ['s1']:                     # 构造测试集的部分
    path = 'dataset/PNTrial/SingleOne-IMU/work/' + subject_name + '.pkl'
    data = torch.load(path)
    acc = data['acc']
    ori = data['ori']
    pose = data['pose']
    tran = data['tran']
    
    length = len(acc)
    
    for i in tqdm(range(length)):
        a_pose = pose[i].view(-1,24,3,3)
        out_gt_pose.append(a_pose.detach().numpy())
        
        p, joint = body_model.forward_kinematics(a_pose)
        
        a_joint_all = joint[:, :24]
        p_all = torch.cat((a_joint_all[:,0:1], a_joint_all[:,1:]-a_joint_all[:,0:1]), dim=1)
        out_p_all = p_all[:, conf.joint_set.full, :]
        out_joint.append(out_p_all.detach().numpy())
        
        glb_acc = acc[i].clone().view(-1,6,3)   # 6,3
        glb_ori = ori[i].clone().view(-1,6,3,3)  # 6,3
        acc_tmp = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1])
        ori_tmp = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
        a_out_imu = torch.cat((acc_tmp.flatten(1), ori_tmp.flatten(1)), dim=1)
        out_imus.append(a_out_imu)
        
        a_shape = torch.zeros(a_pose.shape[0], 10)
        out_shape.append(a_shape)
        
print('Saving')
os.makedirs('GGIP/data_all/SingleOne-IMU', exist_ok=True)
# 构造训练集的部分
np.save(os.path.join('GGIP/data_all/SingleOne-IMU', 'Smpl_singleone_imus_trial.npy'), out_imus)
# np.save(os.path.join('GGIP/data_all/SingleOne-IMU', 'Smpl_singleone_motion_SMPL24.npy'), [out_gt_pose, out_shape])
# np.save(os.path.join('GGIP/data_all/SingleOne-IMU', 'Smpl_singleone_joints23.npy'), out_joint)
# 构造测试集的部分
# np.save(os.path.join('GGIP/data_all/SingleOne-IMU', 'Smpl_singleone_imus_test.npy'), out_imus)
# np.save(os.path.join('GGIP/data_all/SingleOne-IMU', 'Smpl_singleone_motion_SMPL24_test.npy'), out_gt_pose)
    
