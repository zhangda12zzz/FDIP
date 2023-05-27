import torch
from articulate.math.angular import quaternion_to_rotation_matrix
from config import paths
import os
import articulate as art
from utils.Quaternions import Quaternions

def read_csv_data(path):
    file = open(path, 'r', encoding="gbk")
    context = file.read()
    list_result = context.split("\n")

    length = len(list_result)-2  # 行数-2,不知道为什么最后一行会没有数据
    frame_ori = []
    frame_acc = []
    frame_pose = []
    frame_tran = []
    buffer_len = 0
    for i in range(length):
        if i == 0:
            list_result[i] = list_result[i].split(",")
            buffer_len += int(list_result[i][0])
        if i > 1:
            list_result[i] = list_result[i].split(",")
            float_map = list(map(float, list_result[i]))
            line = torch.Tensor(float_map)

            quat = []
            acc_line = []
            pose_line = []
            for joint in range(59):
                a_quat = line[2+joint*21:6+joint*21]    # TODO: ori的四元数也要记住xyzw的顺序需要调整！
                a_acc = line[9+joint*21:12+joint*21]
                a_pose = line[18+joint*21:22+joint*21]
                
                a_pose[0], a_pose[3] = a_pose[3], a_pose[0]
                
                quat.append(a_quat)
                acc_line.append(a_acc)
                pose_line.append(a_pose)
                
            a_pos = line[15:18]
            frame_tran.append(torch.Tensor(a_pos))

            tensor_quat = torch.stack(quat)
            tensor_ori = quaternion_to_rotation_matrix(tensor_quat).view(-1,3,3)    #顺序是xyzw
            frame_ori.append(tensor_ori)
            frame_acc.append(torch.stack(acc_line))
            
            pose_quat = torch.stack(pose_line)
            tensor_pose = quaternion_to_rotation_matrix(pose_quat).view(-1,3,3)
            frame_pose.append(tensor_pose)
    
    oris = torch.stack(frame_ori)
    accs = torch.stack(frame_acc)
    poses = torch.stack(frame_pose)
    trans = torch.stack(frame_tran)
    return buffer_len-1, oris, accs, poses, trans


def getExampleCsv(path = 'dataset/DIPres/take002_chr00.csv'):
    source_par   = [0,0,1,2,0, 4,5,0,7,8, 9,    10,8, 13, 14,15, 8,36,37, 38]
    source_index = [0,1,2,3,4, 5,6,7,8,9, 10,   12,13,14, 15,16,36,37,38, 39]   # 20
    change_index = [0,1,2,3, 5,6,7, 9,10,11,12,13,14,15,16,17,18,19,20,21]      # 20
    buffer_len, oris, accs, poses, trans = read_csv_data(path)
    
    # pose_raw = poses[:,source_index]
    # pose_neck_up = poses[:, 11] # [t,4]
    # pose_head = poses[:, 12]    # [t,4]
    pose = []
    for frame in range(buffer_len):
        a_pose = poses[frame]   # [59,3,3]
        pose_result = torch.zeros(22,3,3)
        pose_result[0] = a_pose[0]  # 根节点
        pose_result[4] = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]])
        pose_result[8] = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]])
        for j in range(len(source_index)-1):
            joint = j + 1
            parent = a_pose[source_par[joint]]
            globalRot = a_pose[source_index[joint]]
            rot_local = parent.transpose(0,1).matmul(globalRot)
            pose_result[change_index[joint]] = rot_local
        pose_result = pose_result.contiguous()  # [22,3,3]
        
        # TODO: 转四元数
        # quat = pose_result.new_zeros(22,4)
        quat = Quaternions.from_transforms(pose_result).qs
        # quat[:,2] = -quat[:,2]
        quat[:,3] = -quat[:,3]
        
        pose.append(torch.Tensor(quat))
    pose = torch.stack(pose)
    tran = trans

    return pose, tran   # [t,22,4] + [t,3]

if __name__ == '__main__':
    getExampleCsv()

