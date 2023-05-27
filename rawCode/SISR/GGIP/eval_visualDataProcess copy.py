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

data = np.load('dataset/PNTrial/SingleOne-IMU/raw/s2/kongfu_chr00_MAYA_pose_tran.npy', allow_pickle=True)

# pre_off, gt_off, err_off, pre_on, gt_on, err_on = np.load('GGIP/eval/gaip-sm/singleone-imu_res.npy', allow_pickle=True)
# tp_pre_off, tp_gt_off, tp_err_off, tp_pre_on, tp_gt_on, tp_err_on = np.load('GGIP/eval/transpose-sm/singleone-imu_res.npy', allow_pickle=True)

# max_sip_dif_idx = 0
# max_sip_dif = 0
        
pose = data[0]  # [t,24,3,3]

pre_on_csv_lines = []

for f in range(pose.shape[0]):   #[24,3,3]
    quat = Quaternions.from_transforms(pose[f]).qs  # [24,4]
    quat = torch.Tensor(quat)
    quat = quat.flatten().numpy()
    pre_on_csv_lines.append(quat)
    
with open("GGIP/eval/forUnity/record.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(pre_on_csv_lines)
