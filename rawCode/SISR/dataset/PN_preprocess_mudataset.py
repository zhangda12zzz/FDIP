import os
import numpy as np
import copy
import torch
from os import listdir
from dataset.PN_preprocess_bvh import processBvh
from dataset.PN_preprocess_csv import processCsv
import articulate as art
from config import *


'''提取acc+ori & pose+tran, 然后构造成对的singleOne-imu的惯性数据'''

data_path = 'dataset/PNTrial/axis/SingleOne-IMU/'
aim_path = 'dataset/PNTrial/SingleOne-IMU/raw/'
work_path = 'dataset/PNTrial/SingleOne-IMU/work/'


# processBvh(data_path, aim_path)
# processCsv(data_path, aim_path)



directories = sorted([f for f in listdir(aim_path) if not f.startswith(".")])
directories = ['s2']
frames = []

for d in directories:
    all_acc = []
    all_ori = []
    all_pose = []
    all_tran = []
    
    
    files = sorted([f for f in listdir(aim_path + d) if f.endswith("acc_ori.npy")])
    frame = 0
    
    for i, motion in enumerate(files):
        pure_motion_name = motion[:-12]
        
        inertialDataPath = aim_path + d + '/' + pure_motion_name + '_acc_ori.npy'
        groundtDataPath = aim_path + d + '/' + pure_motion_name + '_MAYA_pose_tran.npy'
        
        inertialData = np.load(inertialDataPath, allow_pickle=True)
        groundtData = np.load(groundtDataPath, allow_pickle=True)
        
        order = [5,4,2,1,3,0]   #  右左脚，右左手，头，根
        
        acc = inertialData[0]
        ori = inertialData[1].squeeze()
        acc = acc[:,order]
        ori = ori[:,order]
        
        pose = groundtData[0]
        
        # debug
        # body_model = art.ParametricModel(paths.smpl_file)
        # grot, _ = body_model.forward_kinematics(pose)
        
        
        tran = groundtData[1]
        frame += acc.shape[0]
        
        all_acc.append(acc)
        all_ori.append(ori)
        all_pose.append(pose)
        all_tran.append(tran)
        
    data = {'acc': all_acc, 'ori': all_ori, 'pose': all_pose, 'tran': all_tran}
    # np.save(work_path + d + '.npy', data)
    torch.save(data, work_path + d + '.pkl')
    
    # frames.append(frame)
    
# print(frames)
 
