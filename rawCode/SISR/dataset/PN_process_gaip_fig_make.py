import os
import numpy as np
import copy
import torch
from os import listdir
from dataset.bvh_parser import BVH_file
from dataset.motion_dataset import MotionData
from option_parser import get_args, try_mkdir
from utils.Quaternions import Quaternions
import csv


file = BVH_file('dataset/gaip_data_bvh/take001_chr00_MAYA.bvh')
rotations = file.anim.rotations[1:, file.corps, :]   # 取所有关节的旋转[frame, num_joint, 3]
rotations = Quaternions.from_euler(np.radians(rotations)).qs    # 欧拉角->四元数,[frame, num_joint, 4]
order = [0, 1,5,9,2,6,10,3,7,11,4,8, 12, 14,19,13,15,20, 16,21,17,22,18,23]

rotations = torch.Tensor(rotations)
rotations = rotations[:,order]    # [n,24,4]
rotations = rotations.view(-1, 96).numpy()


# quat_gt = Quaternions.from_transforms(gt_on_max_jerk[f]).qs
# quat_gt = torch.Tensor(quat_gt)
# quat_gt = quat_gt.flatten().numpy()
# gt_on_csv_lines.append(quat_gt)

with open("dataset/gaip_data_bvh/fig0_show.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(rotations)