'''
    在构建数据集时，别忘了要去bvh_parser.py里把delete_root_4dataset函数加上，要用的。不然关节数量对不上。
    在我自己的方法里（GGIP），acc不需要缩放。
'''

import os
import torch
import numpy as np

import articulate as art
from data.bvhProcess.bvh_parser import BVH_file
from articulate.math.Quaternions import Quaternions
import config as conf
from scipy.spatial.transform import Rotation
import pickle


device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
# vi_mask = torch.tensor([3021, 1176, 4662, 411, 1961, 5424])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])   # 右手左手、右脚左脚、头、根
# ji_mask = torch.tensor([0, 4, 5, 15, 18, 19])
body_model = art.ParametricModel(conf.paths.smpl_file)

def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def _syn_vel(joints, root_rot):
    r"""
    Synthesize velocity from joints positions.
    
    >> joints: joints global position, can remake shape to [batch_size, 24, 3]
    >> root_rot: ground truth root rotation, can remake shape to [batch_size, 3, 3] 
    """
    joints = joints.view(-1, 24, 3)
    root_rot = root_rot.view(-1, 3, 3)
    vel = torch.stack([(joints[i] - joints[i-1]) * 60 for i in range(1, joints.shape[0])])
    vel = torch.cat((torch.zeros_like(vel[:1]), vel))
    vel = vel.view(-1, 24, 3) / 3 #vel_scale=3
    # vel = root_rot.transpose(1, 2).matmul(vel.transpose(1, 2))
    # print(vel)
    return vel

def process_mixamo():
    motions = []
    out_pose, out_imus = [], []
    out_tran = []
    out_imus_scale = []
    count = 0
    
    # args = option_parser.get_args()
    # args.dataset = 'Smpl'
    # args.is_train = False
    # dataset = testMotionData(args, std_path='dataset/CIP/work/CIP_std_22.bvh')
    # model = create_CIPmodel(args, dataset, std_paths='dataset/CIP/work/CIP_std_22.bvh')
    
    # for bvh_fname in tqdm(glob.glob(os.path.join('dataset/Mixamo/Smpl', '*.bvh'))):
    files = sorted([f for f in os.listdir('dataset/Mixamo/Smpl/') if f.endswith(".bvh")])
    # files = ['2hand Idle.bvh', '180 Turn W_ Briefcase.bvh']
    length = len(files)
    train_split = length // 10 * 8
    for _, motion in enumerate(files):
    #     if not os.path.exists('dataset/Mixamo/Smpl/' + motion):
    #         continue
        file = BVH_file('dataset/Mixamo/Smpl/' + motion)
        # file = BVH_file('dataset/Mixamo/Smpl/2hand Idle.bvh')
        new_position = file.anim.positions[:,0]  # [t,3]
        new_motion = file.anim.rotations#[:, file.corps, :] # [t,22,3]此时是欧拉角
        # new_motion = file.to_tensor().permute((1, 0)).numpy()   # [n,66]
        new_motion = Quaternions.from_euler(np.radians(new_motion))
        motion_m = new_motion.transforms()
        motion_m = torch.Tensor(motion_m)
        
        full_motion = torch.eye(3).repeat(new_motion.shape[0], 24, 1, 1)
        # smpl2mixamo = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,14,17,19,21]
        order_modify = [0,1,5,9,2, 6,10,3,7,11, 4,8,12,14,19, 13,15,20,16,21, 17,22,18,23]
        full_motion = motion_m[:, order_modify]
        full_position = torch.Tensor(new_position)
        
        
        # length += 1
        # motions.append([full_motion, full_position])
        
        # smpl2CIP = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,22,14,17,19,21,23]
    
        p = full_motion
        trans = full_position / 100.0
        out_tran.append(trans)
        
        # art.ParametricModel(conf.paths.smpl_file).view_motion([p], [trans])
        # break
        
        grot, joint, vert = body_model.forward_kinematics(p, tran=trans, calc_mesh=True)
        
        a_out_pose = p.clone()  # 24,3
        # a_out_pose = Quaternions.from_transforms(a_out_pose.detach().numpy()).qs#.view(-1,24,4) # 24,4
        # a_out_pose_ = a_out_pose[:,smpl2CIP]
        out_pose.append(a_out_pose.detach().numpy())
        
        a_out_vacc = _syn_acc(vert[:, vi_mask])   # [t,6,3]
        a_out_vrot = grot[:, ji_mask]  # [t,6,3,3]
        # acc_tmp = torch.cat((a_out_vacc[:, :1], a_out_vacc[:, 1:] - a_out_vacc[:, :1]), dim=1).bmm(a_out_vrot[:, 0]) / conf.acc_scale
        # ori_tmp = torch.cat((a_out_vrot[:, :1], a_out_vrot[:, :1].transpose(2, 3).matmul(a_out_vrot[:, 1:])), dim=1)
        acc_tmp_scale = torch.cat((a_out_vacc[:, :5] - a_out_vacc[:, 5:], a_out_vacc[:, 5:]), dim=1).bmm(a_out_vrot[:, -1]) / conf.acc_scale
        acc_tmp = torch.cat((a_out_vacc[:, :5] - a_out_vacc[:, 5:], a_out_vacc[:, 5:]), dim=1).bmm(a_out_vrot[:, -1])# / conf.acc_scale
        ori_tmp = torch.cat((a_out_vrot[:, 5:].transpose(2, 3).matmul(a_out_vrot[:, :5]), a_out_vrot[:, 5:]), dim=1)
        # ori_tmp = Quaternions.from_transforms(ori_tmp.detach().numpy()).qs # 6,4
        # acc_tmp = acc_tmp.detach().numpy() 
        # a_out_imu = np.concatenate((acc_tmp, ori_tmp), axis=-1)   # N,6,7
        a_out_imu_scale = torch.cat((acc_tmp_scale.flatten(1), ori_tmp.flatten(1)), dim=1)
        out_imus_scale.append(a_out_imu_scale.detach().numpy())
        a_out_imu = torch.cat((acc_tmp.flatten(1), ori_tmp.flatten(1)), dim=1)
        out_imus.append(a_out_imu.detach().numpy())
        
        count += 1
        # if count >= 100:
        #     break
        
    print('play complete')
    # np.save('./dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_motion_SMPL24_train.npy', out_pose[:train_split])  
    # np.save('./dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_imu_TP_train.npy', out_imus[:train_split]) 
    # np.save('./dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_tran_train.npy', out_tran[:train_split])
    # np.save('./dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_motion_SMPL24_test.npy', out_pose[train_split:])  
    # np.save('./dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_imu_TP_test.npy', out_imus[train_split:]) 
    # np.save('./dataset/CIP/mixamoSMPLPos/mixamoSMPLPos_tran_test.npy', out_tran[train_split:])
    np.save('GGIP/data_all/Mixamo/mixamoSMPLPos_motion_SMPL24_test.npy', out_pose)  
    # np.save('GGIP/data_all/Mixamo/mixamoSMPLPos_imu_test_withAccScale.npy', out_imus_scale)
    np.save('GGIP/data_all/Mixamo/mixamoSMPLPos_imu_test.npy', out_imus)      
    
    
def process_mixamo_tip():
    motions = []
    out_pose, out_imus = [], []
    out_tran = []
    out_imus_scale = []
    count = 0
    
    files = sorted([f for f in os.listdir('data/dataset_raw/Mixamo/Smpl/') if f.endswith(".bvh")])
    # files = ['2hand Idle.bvh', '180 Turn W_ Briefcase.bvh']
    length = len(files)
    train_split = length // 10 * 8
    for _, motion in enumerate(files):
        file = BVH_file('data/dataset_raw/Mixamo/Smpl/' + motion)
        # file = BVH_file('dataset/Mixamo/Smpl/2hand Idle.bvh')
        new_position = file.anim.positions[:,0]  # [t,3]
        new_motion = file.anim.rotations#[:, file.corps, :] # [t,22,3]此时是欧拉角
        # new_motion = file.to_tensor().permute((1, 0)).numpy()   # [n,66]
        new_motion = Quaternions.from_euler(np.radians(new_motion))
        motion_m = new_motion.transforms()
        motion_m = torch.Tensor(motion_m)
        
        full_motion = torch.eye(3).repeat(new_motion.shape[0], 24, 1, 1)
        # smpl2mixamo = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,14,17,19,21]
        order_modify = [0,1,5,9,2, 6,10,3,7,11, 4,8,12,14,19, 13,15,20,16,21, 17,22,18,23]
        full_motion = motion_m[:, order_modify]
        full_position = torch.Tensor(new_position)
        
        p = full_motion
        trans = full_position / 100.0
        out_tran.append(trans)
        
        grot, joint, vert = body_model.forward_kinematics(p, tran=trans, calc_mesh=True)
        
        a_out_pose = p.clone()  # 24,3
        # a_out_pose = Quaternions.from_transforms(a_out_pose.detach().numpy()).qs#.view(-1,24,4) # 24,4
        # a_out_pose_ = a_out_pose[:,smpl2CIP]
        out_pose.append(a_out_pose.detach().numpy())
        
        a_out_vacc = _syn_acc(vert[:, vi_mask])   # [t,6,3]
        a_out_vrot = grot[:, ji_mask]  # [t,6,3,3]
        
        a_acc = np.array(a_out_vacc.tolist())
        a_ori = np.array(a_out_vrot.tolist())
        a_pose_raw = p.view(-1,24,3,3)  # [t,24,3,3] -> [72]
        a_pose = np.zeros((len(a_pose_raw), 72))
        for j in range(len(a_pose_raw)):
            matrix = a_pose_raw[j]
            rot = Rotation.from_matrix(matrix).as_rotvec()
            a_pose[j] = rot.flatten()
            
        a_tran = np.array(out_tran[0].tolist())
                
        save_dict = {'accs': a_acc, 'oris': a_ori, 'poses': a_pose, 'trans': a_tran}
        save_name = 'data/dataset_work/Mixamo/s_' + motion + '.pkl'
        save_path = save_name.replace(' ', '_')
        with open(save_path, 'wb') as fo:     # 将数据写入pkl文件
            pickle.dump(save_dict, fo)
    
    
if __name__ == '__main__':
    process_mixamo()
    # process_mixamo_tip()
