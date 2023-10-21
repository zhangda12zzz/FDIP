'''
    进行数据预处理，产生三种数据：
    imu数据：关节点顺序为右手左手、右脚左脚、头、根，acc（18）+ori（54），已经经过标准化。(加速度没有除以缩放因子)
    joint数据：按SMPL顺序的23个节点坐标，已经经过根节点标准化。
    pose：SMPL姿态参数真值，24个关节数据。
'''


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用
import pickle
import torch
import numpy as np
from tqdm import tqdm
import glob

import articulate as art
from config import *
import data.utils.bvh as bvh
import data.utils.quat as quat
import config as conf

vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
# vi_mask = torch.tensor([3021, 1176, 4662, 411, 1961, 5424])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])   # 右手左手、右脚左脚、头、根
# ji_mask = torch.tensor([0, 4, 5, 15, 18, 19])
body_model = art.ParametricModel(paths.smpl_file)


def _syn_acc(v, smooth_n=2): # 4
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
    vel = vel.view(-1, 24, 3) / vel_scale
    # vel = root_rot.transpose(1, 2).matmul(vel.transpose(1, 2))
    # print(vel)
    return vel

def process_amass():
    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
    # for ds_name in amass_data_test_tmp:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose = []
    out_imus = []
    out_gt_pose = []
    out_joint = []
    out_shape = []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        
        # out_pose.append(pose[b:b + l].clone())  # N, 24, 3 
        a_out_pose = pose[b:b + l].clone()  # 24,3
        a_out_pose_ = art.math.axis_angle_to_quaternion(a_out_pose).view(-1,24,4) # 24,4
        a_out_pose_ = a_out_pose_[:,joint_set.graphA]
        out_pose.append(a_out_pose_.detach().numpy())
        
        a_pose_gt = art.math.axis_angle_to_rotation_matrix(a_out_pose).view(-1, 24, 3, 3)
        out_gt_pose.append(a_pose_gt.detach().numpy())
        
        a_joint_all = joint[:, :24]
        p_all = torch.cat((a_joint_all[:,0:1], a_joint_all[:,1:]-a_joint_all[:,0:1]), dim=1)
        out_p_all = p_all[:, conf.joint_set.full, :]
        out_joint.append(out_p_all.detach().numpy())
        
        glb_acc = _syn_acc(vert[:, vi_mask])   # t,6,3
        glb_ori = grot[:, ji_mask]  # t,6,3,3
        
        
        # acc_tmp = torch.cat((a_out_vacc[:, :1], a_out_vacc[:, 1:] - a_out_vacc[:, :1]), dim=1).bmm(a_out_vrot[:, 0]) / conf.acc_scale
        # ori_tmp = torch.cat((a_out_vrot[:, :1], a_out_vrot[:, :1].transpose(2, 3).matmul(a_out_vrot[:, 1:])), dim=1)
        # ori_tmp = quat.from_xform(ori_tmp.detach().numpy()) # 6,4
        # acc_tmp = acc_tmp.detach().numpy() 
        # a_out_imu = np.concatenate((acc_tmp, ori_tmp), axis=-1)   # N,6,7
        
        acc_tmp = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1])# / acc_scale
        ori_tmp = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
        a_out_imu = torch.cat((acc_tmp.flatten(1), ori_tmp.flatten(1)), dim=1)
        
        out_imus.append(a_out_imu)
        
        a_shape = shape[i].repeat(a_pose_gt.shape[0],1)
        out_shape.append(a_shape)  

        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    # np.save(os.path.join(paths.amass_dir, 'Smpl_amass_test_motion.npy'), out_pose)
    # np.save(os.path.join('data/dataset_work/GGIP_used/amass', 'Smpl_amass_joints23.npy'), out_joint)
    # np.save(os.path.join('data/dataset_work/GGIP_used/amass', 'Smpl_amass_imus.npy'), out_imus)
    np.save(os.path.join('data/dataset_work/GGIP_used/amass', 'Smpl_amass_motion_SMPL24.npy'), [out_gt_pose, out_shape])
    # np.save(os.path.join('data/dataset_work/GGIP_used/amass', 'Smpl_amass_shape.npy'), out_shape)
    

def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]
    # imu_mask = [2, 11, 12, 0, 7, 8] # 根、左脚右脚、头、左手右手
    train_split = ['s_01','s_02','s_03','s_04','s_05','s_06','s_07','s_08']
    # train_split = ['s_09','s_10']
    accs, oris, poses, trans = [], [], [], []

    # for subject_name in test_split:
    for subject_name in train_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    length = len(accs)
    
    out_pose = []
    out_imus = []
    out_gt_pose = []
    out_joint = []
    out_shape = []
    for i in tqdm(range(length)):
        p = art.math.axis_angle_to_rotation_matrix(poses[i]).view(-1, 24, 3, 3)
        grot, joint = body_model.forward_kinematics(p)
        a_joint_all = joint[:, :24]
        p_all = torch.cat((a_joint_all[:,0:1], a_joint_all[:,1:]-a_joint_all[:,0:1]), dim=1)
        out_p_all = p_all[:, conf.joint_set.full, :]
        out_joint.append(out_p_all.detach().numpy())
        
        a_out_pose = poses[i].clone().view(-1,24,3)  # 24,3
        a_out_pose_ = art.math.axis_angle_to_quaternion(a_out_pose).view(-1,24,4) # 24,4
        a_out_pose_ = a_out_pose_[:,joint_set.graphA]
        out_pose.append(a_out_pose_.detach().numpy())
        
        a_pose_gt = art.math.axis_angle_to_rotation_matrix(a_out_pose).view(-1, 24, 3, 3)
        out_gt_pose.append(a_pose_gt.detach().numpy())
        
        glb_acc = accs[i].clone().view(-1,6,3)   # 6,3
        glb_ori = oris[i].clone().view(-1,6,3,3)  # 6,3
        # acc_tmp = torch.cat((a_out_vacc[:, :1], a_out_vacc[:, 1:] - a_out_vacc[:, :1]), dim=1).bmm(a_out_vrot[:, 0]) / conf.acc_scale
        # ori_tmp = torch.cat((a_out_vrot[:, :1], a_out_vrot[:, :1].transpose(2, 3).matmul(a_out_vrot[:, 1:])), dim=1)
        # ori_tmp = quat.from_xform(ori_tmp.detach().numpy()) # 6,4
        # acc_tmp = acc_tmp.detach().numpy() 
        # a_out_imu = np.concatenate((acc_tmp, ori_tmp), axis=-1)   # N,6,7
        acc_tmp = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1])# / acc_scale
        ori_tmp = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
        a_out_imu = torch.cat((acc_tmp.flatten(1), ori_tmp.flatten(1)), dim=1)
        
        out_imus.append(a_out_imu)
        
        a_shape = torch.zeros(a_pose_gt.shape[0], 10)
        out_shape.append(a_shape)
        
    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    # np.save(os.path.join(paths.amass_dir, 'Smpl_dipTrain_motion.npy'), out_pose)
    # np.save(os.path.join('data/dataset_work/GGIP_used/dip', 'Smpl_dipTrain_imus.npy'), out_imus)
    np.save(os.path.join('data/dataset_work/GGIP_used/dip', 'Smpl_dipTrain_motion_SMPL24.npy'), [out_gt_pose,out_shape])
    # np.save(os.path.join('data/dataset_work/GGIP_used/dip', 'Smpl_dipTrain_joints23.npy'), out_joint)
    

def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'
    imuOrder = [2, 3, 0, 1, 4, 5]
    # imuOrder = [5,0,1,4,2,3]

    accs, oris, poses, trans = [], [], [], []
    # for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
    for file_folder in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        for file in sorted(os.listdir(os.path.join(paths.raw_totalcapture_dip_dir, file_folder))):
            data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file_folder, file), 'rb'), encoding='latin1')
            ori = torch.from_numpy(data['ori']).float()[:, torch.tensor(imuOrder)]
            acc = torch.from_numpy(data['acc']).float()[:, torch.tensor(imuOrder)]
            pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

            # acc/ori and gt pose do not match in the dataset
            if acc.shape[0] < pose.shape[0]:
                pose = pose[:acc.shape[0]]
            elif acc.shape[0] > pose.shape[0]:
                acc = acc[:pose.shape[0]]
                ori = ori[:pose.shape[0]]

            assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
            accs.append(acc)    # N, 6, 3
            oris.append(ori)    # N, 6, 3, 3
            poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    # remove acceleration bias
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        _, _, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    # 提取用于val训练结果的tc数据
    length = len(accs)
    out_pose, out_imus = [], []
    out_gt_pose = []
    for i in tqdm(range(length)):
        # out_pose.append(poses[i].clone())  # N, 24, 3
        a_out_pose = poses[i].clone().view(-1,24,3)  # 24,3
        a_out_pose_ = art.math.axis_angle_to_quaternion(a_out_pose).view(-1,24,4) # 24,4
        a_out_pose_ = a_out_pose_[:,joint_set.graphA]
        out_pose.append(a_out_pose_.detach().numpy())
        
        a_pose_gt = art.math.axis_angle_to_rotation_matrix(a_out_pose).view(-1, 24, 3, 3)
        out_gt_pose.append(a_pose_gt.detach().numpy())
           
        # out_vacc.append(accs[i].clone())  # N, 6, 3
        # out_vrot.append(oris[i].clone())  # N, 6, 3, 3
        glb_acc = accs[i].clone().view(-1,6,3)   # 6,3
        glb_ori = oris[i].clone().view(-1,6,3,3)  # 6,3
        # acc_tmp = torch.cat((a_out_vacc[:, :1], a_out_vacc[:, 1:] - a_out_vacc[:, :1]), dim=1).bmm(a_out_vrot[:, 0]) / conf.acc_scale
        # ori_tmp = torch.cat((a_out_vrot[:, :1], a_out_vrot[:, :1].transpose(2, 3).matmul(a_out_vrot[:, 1:])), dim=1)
        # ori_tmp = quat.from_xform(ori_tmp.detach().numpy()) # 6,4
        # acc_tmp = acc_tmp.detach().numpy() 
        # a_out_imu = np.concatenate((acc_tmp, ori_tmp), axis=-1)   # N,6,7
        acc_tmp = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1])# / acc_scale
        ori_tmp = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
        a_out_imu = torch.cat((acc_tmp.flatten(1), ori_tmp.flatten(1)), dim=1)
        
        out_imus.append(a_out_imu)
    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    # np.save(os.path.join('data/dataset_work/GGIP_used/tc', 'Smpl_tc_motion_test.npy'), out_pose)
    np.save(os.path.join('data/dataset_work/GGIP_used/tc', 'Smpl_tc_imus_test.npy'), out_imus)
    np.save(os.path.join('data/dataset_work/GGIP_used/tc', 'Smpl_tc_motion_SMPL24_test.npy'), out_gt_pose)


if __name__ == '__main__':
    process_amass()
    # process_dipimu()
    # process_totalcapture()
