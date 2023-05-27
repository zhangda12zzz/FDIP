import torch
import tqdm
from config import *
from utils import *
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import articulate as art
from articulate.utils.rbdl import *
from net import PIP, TransPoseNet
from gcn_utils.gnet import GGCNUnpool
import torch.nn as nn
import csv


aimIdx = 13

data_dir = paths.dipimu_dir  # paths.totalcapture_dir
# net = PIP()
net2 = TransPoseNet()
net3 = PIP()
net4 = GGCNUnpool()

data_name = os.path.basename(data_dir)  # 数据文件夹
# result_dir = os.path.join(paths.result_dir, data_name, net.name)
# result_dir2 = os.path.join(paths.result_dir, data_name, net2.name)
result_dir3 = os.path.join(paths.result_dir, data_name, net3.name)
# print_title('Evaluating "%s" on "%s"' % (net.name, data_name))

_, _, pose_t_all, tran_t_all = torch.load(os.path.join(data_dir, 'test.pt')).values()   # 加载数据（pose和tran的真值）

sequence_ids = list(range(len(pose_t_all))) # 构造遍历列表（和数据长度一致）

for i in tqdm.tqdm(sequence_ids):
    if i == aimIdx:
        # result = torch.load(os.path.join(result_dir, '%d.pt' % i))
        # pose_p, tran_p = result[0], result[1]               # pose:[n,24,3,3], tran:[n,3]
        
        pose_t, tran_t = pose_t_all[i], tran_t_all[i]
        # pose_t = art.math.axis_angle_to_rotation_matrix(pose_t).view_as(pose_p)
        
        # result2 = torch.load(os.path.join(result_dir2, '%d.pt' % i))
        # pose_p2, tran_p2 = result2[0], result2[1]               # pose:[n,24,3,3], tran:[n,3]
        
        result3 = torch.load(os.path.join(result_dir3, '%d.pt' % i))
        pose_p3, tran_p3 = result3[0], result3[1]               # pose:[n,24,3,3], tran:[n,3]
        
        # pose_p_quat = art.math.rotation_matrix_to_quat(pose_p).view(-1, 24, 4)  # [n, 24, 4]
        # data_p_csv = torch.cat((tran_p, pose_p_quat.view(-1, 96)), 1).detach().numpy()
        
        # pose_t_quat = art.math.rotation_matrix_to_quat(pose_t).view(-1, 24, 4)  # [n, 24, 4]
        # data_t_csv = torch.cat((tran_t, pose_t_quat.view(-1, 96)), 1).detach().numpy()
        
        # pose_p_quat2 = art.math.rotation_matrix_to_quat(pose_p2).view(-1, 24, 4)  # [n, 24, 4]
        # data_p_csv2 = torch.cat((tran_p2, pose_p_quat2.view(-1, 96)), 1).detach().numpy()
        
        pose_p_quat3 = art.math.rotation_matrix_to_quat(pose_p3).view(-1, 24, 4)  # [n, 24, 4]
        data_p_csv3 = torch.cat((tran_p3, pose_p_quat3.view(-1, 96)), 1).detach().numpy()
        
        
        

        # art.ParametricModel(paths.smpl_file).view_motion([pose_p], [tran_p])

        # 真值
        # with open('data/show/gt/gt_%d.csv' % i, 'w') as csvfile1:
        #     writer1 = csv.writer(csvfile1)

        #     writer1.writerow(range(24*4+3))
        #     writer1.writerows(data_t_csv)

        # PIP
        # with open('data/show/predict/PIP/pd_%d.csv' % i, 'w') as csvfile2:
        #     writer2 = csv.writer(csvfile2)
            
        #     writer2.writerow(range(24*4+3))
        #     writer2.writerows(data_p_csv)
            
        # TransPoseNet
        # with open('data/show/predict/TransPoseNet/pd_TP_%d.csv' % i, 'w') as csvfile3:
        #     writer3 = csv.writer(csvfile3)
            
        #     writer3.writerow(range(24*4+3))
        #     writer3.writerows(data_p_csv2)
        
        # break
    
        # PIP_my
        with open('data/show/predict/mine/pd_mine_%d.csv' % i, 'w') as csvfile4:
            writer4 = csv.writer(csvfile4)
            
            writer4.writerow(range(24*4+3))
            writer4.writerows(data_p_csv3)
        
        break