import os
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用
# from dataset.bvh_parser import BVH_file
# from dataset import get_test_set
# from option_parser import get_std_bvh
from utils.Quaternions import Quaternions
from config import paths, joint_set
import articulate as art
import option_parser
import csv

# pre_off, gt_off, err_off, pre_on, gt_on, err_on = np.load('GGIP/eval/gaip-sm/dip-imu_res.npy', allow_pickle=True)


# 挑选了最大抖动帧
# max_jerk_idx = 0
# max_jerk = 0

# for i in range(len(err_on)):
#     if err_on[i][4][0] > max_jerk:
#         max_jerk = err_on[i][4][0]
#         max_jerk_idx = i
        
# pre_on_max_jerk = pre_on[i].numpy() #[t,24,3,3]
# gt_on_max_jerk = gt_on[i].numpy()   #[t,24,3,3]

# pre_on_csv_lines = []
# gt_on_csv_lines = []
# for f in range(pre_on_max_jerk.shape[0]):   #[24,3,3]
#     quat = Quaternions.from_transforms(pre_on_max_jerk[f]).qs  # [24,4]
#     quat = torch.Tensor(quat)
#     quat = quat.flatten().numpy()
#     pre_on_csv_lines.append(quat)
    
#     quat_gt = Quaternions.from_transforms(gt_on_max_jerk[f]).qs
#     quat_gt = torch.Tensor(quat_gt)
#     quat_gt = quat_gt.flatten().numpy()
#     gt_on_csv_lines.append(quat_gt)
    

# with open("GGIP/eval/dip-imu_online_res_maxJerk.csv","w") as csvfile: 
#     writer = csv.writer(csvfile)    
#     writer.writerows(pre_on_csv_lines)
# with open("GGIP/eval/dip-imu_online_gt_maxJerk.csv","w") as csvfile: 
#     writer = csv.writer(csvfile)    
#     writer.writerows(gt_on_csv_lines)


# 那来挑选一下最佳估计========================
# tp_pre_off, tp_gt_off, tp_err_off, tp_pre_on, tp_gt_on, tp_err_on = np.load('GGIP/eval/transpose-sm/dip-imu_res.npy', allow_pickle=True)

# max_sip_dif_idx = 0
# max_sip_dif = 0

# 离线数据
# for i in range(len(err_off)):
#     if tp_err_off[i][0][0] - err_off[i][0][0] > max_sip_dif:
#         max_sip_dif = tp_err_off[i][0][0] - err_off[i][0][0]
#         max_sip_dif_idx = i
        
# a_pre_off = pre_off[i].squeeze(0).numpy() #[t,24,3,3]
# a_gt_off = gt_off[i].squeeze(0).numpy()   #[t,24,3,3]
# a_tp_pre_off = tp_pre_off[i].squeeze(0).numpy() #[t,24,3,3]
# a_tp_gt_off = tp_gt_off[i].squeeze(0).numpy()   #[t,24,3,3]

# pre_off_csv_lines = []
# gt_off_csv_lines = []
# tp_off_csv_lines = []
# for f in range(a_pre_off.shape[0]):   #[24,3,3]
#     quat = Quaternions.from_transforms(a_pre_off[f]).qs  # [24,4]
#     quat = torch.Tensor(quat)
#     quat = quat.flatten().numpy()
#     pre_off_csv_lines.append(quat)
    
#     quat_gt = Quaternions.from_transforms(a_gt_off[f]).qs
#     quat_gt = torch.Tensor(quat_gt)
#     quat_gt = quat_gt.flatten().numpy()
#     gt_off_csv_lines.append(quat_gt)
    
#     quat_tp = Quaternions.from_transforms(a_tp_pre_off[f]).qs
#     quat_tp = torch.Tensor(quat_tp)
#     quat_tp = quat_tp.flatten().numpy()
#     tp_off_csv_lines.append(quat_tp)
    
# with open("GGIP/eval/forUnity/dip-imu_offline_res_mybest.csv","w") as csvfile: 
#     writer = csv.writer(csvfile)    
#     writer.writerows(pre_off_csv_lines)
# with open("GGIP/eval/forUnity/dip-imu_offline_gt_mybest.csv","w") as csvfile: 
#     writer = csv.writer(csvfile)    
#     writer.writerows(gt_off_csv_lines)
# with open("GGIP/eval/forUnity/dip-imu_offline_tp_mybest.csv","w") as csvfile: 
#     writer = csv.writer(csvfile)    
#     writer.writerows(tp_off_csv_lines)
    
pre_off, gt_off, err_off, pre_on, gt_on, err_on = np.load('GGIP/eval/gaip-sm/singleone-imu_res.npy', allow_pickle=True)
tp_pre_off, tp_gt_off, tp_err_off, tp_pre_on, tp_gt_on, tp_err_on = np.load('GGIP/eval/transpose-sm/singleone-imu_res.npy', allow_pickle=True)

max_sip_dif_idx = 0
max_sip_dif = 0


# 在线数据    
for i in range(len(err_on)):
    if tp_err_on[i][0][0] - err_on[i][0][0] > max_sip_dif:
        max_sip_dif = tp_err_on[i][0][0] - err_on[i][0][0]
        max_sip_dif_idx = i

# i = 19
        
a_pre_on = pre_on[max_sip_dif_idx].squeeze().numpy() #[t,24,3,3]
a_gt_on = gt_on[max_sip_dif_idx].squeeze().numpy()   #[t,24,3,3]
a_tp_pre_on = tp_pre_on[max_sip_dif_idx].squeeze().numpy() #[t,24,3,3]
a_tp_gt_on = tp_gt_on[max_sip_dif_idx].squeeze().numpy()   #[t,24,3,3]

pre_on_csv_lines = []
gt_on_csv_lines = []
tp_on_csv_lines = []
for f in range(a_pre_on.shape[0]):   #[24,3,3]
    quat = Quaternions.from_transforms(a_pre_on[f]).qs  # [24,4]
    quat = torch.Tensor(quat)
    quat = quat.flatten().numpy()
    pre_on_csv_lines.append(quat)
    
    quat_gt = Quaternions.from_transforms(a_gt_on[f]).qs
    quat_gt = torch.Tensor(quat_gt)
    quat_gt = quat_gt.flatten().numpy()
    gt_on_csv_lines.append(quat_gt)
    
    quat_tp = Quaternions.from_transforms(a_tp_pre_on[f]).qs
    quat_tp = torch.Tensor(quat_tp)
    quat_tp = quat_tp.flatten().numpy()
    tp_on_csv_lines.append(quat_tp)
    
with open("GGIP/eval/forUnity/so-v_res_mybest.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(pre_on_csv_lines)
with open("GGIP/eval/forUnity/so-v_gt_mybest.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(gt_on_csv_lines)
with open("GGIP/eval/forUnity/so-v_tp_mybest.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(tp_on_csv_lines)
