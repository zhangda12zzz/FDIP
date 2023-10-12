import torch
import os
from articulate.math.Quaternions import Quaternions
from os import listdir
import numpy as np

'''读取data_path里所有文件夹中的csv文件，将acc+ori结果输出到aim_path的同名文件夹中'''

def read_csv_frame(data_path):
    directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
    all_frames = []
    names = []
    for d in directories:
        files = sorted([f for f in listdir(data_path + d) if f.endswith(".csv")])
        d_frames = []
        for i, motion in enumerate(files):
            if not os.path.exists(data_path + d + '/' + motion):
                continue
            
            pure_motion_name = motion[:-4]
            
            path_tmp = data_path + d + '/' + motion
            frame, ori, acc = read_csv_data(path_tmp)
            d_frames.append([motion, frame])
        all_frames.append(d_frames)  
        names.append(d)
    
    return all_frames, names
            
            
            

def read_csv_data(path):
    file = open(path, 'r', encoding="gbk")
    context = file.read()
    list_result = context.split("\n")

    length = len(list_result)-1
    frame_ori = []
    frame_acc = []
    buffer_len = 0
    for i in range(length):
        if i == 0:
            list_result[i] = list_result[i].split(",")
            buffer_len += int(list_result[i][0])
        if i > 1:
            list_result[i] = list_result[i].split(",")
            float_map = list(map(float, list_result[i]))
            line = torch.Tensor(float_map)

            acc_line = []
            ori_line = []
            joints = [0, 3, 6, 12, 16, 39] # 顺序都是：根、左右脚、头、左右手
            for joint in joints:
                # a_quat = line[2+joint*21:6+joint*21]    # TODO: ori的四元数也要记住xyzw的顺序需要调整！
                a_acc = line[1+8+joint*21:1+11+joint*21]
                a_acc = a_acc.numpy()
                
                a_ori = line[1+17+joint*21:1+21+joint*21]
                a_ori = a_ori.numpy()
                
                quat_order = [3,0,1,2]
                a_ori = a_ori[quat_order]
                a_ori = Quaternions(a_ori)
                ori_mat = a_ori.transforms()    #[3,3]
                
                acc_line.append(torch.Tensor(a_acc))  #[6,3]
                
                ori_line.append(torch.Tensor(ori_mat))  #[6,3,3]
                
            frame_ori.append(torch.stack(ori_line))
            frame_acc.append(torch.stack(acc_line))
    
    oris = torch.stack(frame_ori)
    accs = torch.stack(frame_acc)
    return buffer_len, oris, accs

def processCsv(data_path, aim_path):
    # data_path = 'dataset/PNTrial/axis/'
    # aim_path = 'dataset/PNTrial/axis_work/'
    # directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
    directories = ['s2']

    for d in directories:
        files = sorted([f for f in listdir(data_path + d) if f.endswith(".csv")])

        for i, motion in enumerate(files):
            if not os.path.exists(data_path + d + '/' + motion):
                continue
            
            pure_motion_name = motion[:-4]
            
            path_tmp = data_path + d + '/' + motion
            frame, ori, acc = read_csv_data(path_tmp)
            
            res = [acc, ori]
            
            np.save(aim_path + d + '/' + pure_motion_name + '_acc_ori.npy', res)


if __name__ == '__main__':
    frames, names = read_csv_frame('dataset/PNTrial/axis/SingleOne-IMU/')
    print(frames)
    print(names)
    
    
    # s1: 32413 / 8547
    # s2: 43840 / 8511
    # s4: 35908 / 10305
    # s5: 33813 / 5559
    # s6: 48139 / 15286
    
    