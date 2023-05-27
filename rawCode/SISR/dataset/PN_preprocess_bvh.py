import os
import numpy as np
import copy
import torch
from os import listdir
from dataset.bvh_parser import BVH_file
from dataset.motion_dataset import MotionData
from option_parser import get_args, try_mkdir
from utils.Quaternions import Quaternions

'''读取data_path里所有文件夹中的bvh文件，将pose+tran结果输出到aim_path的同名文件夹中'''

def processBvh(data_path, aim_path):

    # data_path = 'dataset/PNTrial/axis/'
    # aim_path = 'dataset/PNTrial/axis_work/'
    # directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
    directories = ['s2']

    for d in directories:
        files = sorted([f for f in listdir(data_path + d) if f.endswith("_MAYA.bvh")])

        for i, motion in enumerate(files):
            if not os.path.exists(data_path + d + '/' + motion):
                continue
            
            pure_motion_name = motion[:-4]
            
            file = BVH_file(data_path + d + '/' + motion)
        
            # 对于bvh文件(mixamo版本)提取SMPL姿态参数：
            # file = BVH_file('dataset/PNTrial/axis/yankChair_chr00_MAYA.bvh')
            rotations = file.anim.rotations[1:, file.corps, :]   # 取所有关节的旋转[frame, num_joint, 3]
            rotations = Quaternions.from_euler(np.radians(rotations))    # 欧拉角->四元数,[frame, num_joint, 4]
            order = [0, 1,5,9,2,6,10,3,7,11,4,8, 12, 14,19,13,15,20, 16,21,17,22,18,23]

            rotations_mat = rotations.transforms()
            rotations_mat = torch.Tensor(rotations_mat)[:,order]        # [frame, 24, 3, 3]
            
            positions = file.anim.positions[1:, 0, :]    # 只取根节点位置    [frame, 3]
            positions = torch.Tensor(positions)
            
            res = [rotations_mat, positions]
            np.save(aim_path + d + '/' + pure_motion_name + '_pose_tran.npy', res)


    