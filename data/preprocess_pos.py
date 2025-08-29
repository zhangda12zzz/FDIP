"""
人体姿态估计数据预处理工具集：
1. 支持三大数据集处理：
   - AMASS：处理运动捕捉数据，合成加速度/方向数据
   - DIP-IMU：处理惯性测量单元数据，填补缺失值
   - TotalCapture：对齐多模态数据并校正加速度偏差

2. 核心处理流程：
   (1) AMASS预处理：
       - 根据帧率调整采样步长（120fps→60fps）
       - 使用SMPL模型计算骨骼运动学（关节位置/顶点坐标）
       - 合成6个IMU传感器的加速度数据
       - 旋转对齐使数据与DIP格式兼容

   (2) DIP-IMU预处理：
       - 处理缺失值（NaN）：使用最近邻插值填充
       - 提取6个关键传感器数据（左手/右手/双脚/头/根节点）
       - 通过正向运动学计算骨骼姿态
       - 保存标准化后的姿态/形状/平移数据

   (3) TotalCapture预处理：
       - 对齐加速度/方向/姿态数据长度
       - 从文本文件提取全局位置数据（英尺转米制）
       - 校正加速度偏差：通过虚拟顶点计算真实加速度基准
       - 保存多模态对齐的测试数据集

3. 输出规范：
   - 统一保存为PyTorch张量文件（.pt）
   - 包含以下关键数据：
     * pose.pt：SMPL轴角姿态参数
     * vrot.pt：传感器方向旋转矩阵
     * vacc.pt：合成加速度数据
     * joint.pt：全局关节位置
     * shape.pt：体型参数

4. 关键技术：
   - 使用Articulate库进行骨骼运动学计算
   - 通过顶点位置差分合成加速度（中心差商法）
   - 多传感器数据对齐与插值处理
   - 物理约束校正（加速度偏差消除）
"""
import sys

import articulate as art
import torch
import os
import pickle
from config import paths, amass_data, amass_data_test_tmp
import numpy as np
from tqdm import tqdm
import glob  #查找符合特定规则的文件路径


vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])       # 右手左手、右脚左脚、头、根
body_model = art.ParametricModel(paths.smpl_file)

#合成加速度
def _syn_acc(v, smooth_n=4):
    r"""
    # 从顶点位置合成加速度
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

#处理AMASS数据集-.npz文件处理
'''
joint.pt     --- 全局关节位置
shape.pt     --- 体型参数
vrot.pt      --- 传感器方向旋转矩阵
vacc.pt      --- 合成加速度数据
pose.pt      --- SMPL旋转矩阵
'''
def process_amass():
    data_pose, data_trans, data_beta, length = [], [], [], []
    # for ds_name in amass_data:
    for ds_name in amass_data_test_tmp:
        print('\rReading', ds_name)
        #注意路径格式
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1    #60帧
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
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(p.clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
    torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)


#预处理DIP-IMU数据集
'''
读取并处理原始数据。.pkl文件
填补 NaN 值。   --- nearest-neighbor imputation
将数据转换为所需格式。
进行运动学推理，计算姿势和旋转矩阵。
保存处理后的数据。  ---.pt文件
'''
def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_01','s_02','s_03','s_04','s_05','s_06','s_07','s_08','s_09','s_10']
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):    #路径
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

    # os.makedirs(paths.dipimu_dir, exist_ok=True)
    # torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    # print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)
    length = len(accs)
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i in tqdm(range(length)):
        p = art.math.axis_angle_to_rotation_matrix(poses[i]).view(-1, 24, 3, 3)
        grot, joint = body_model.forward_kinematics(p, calc_mesh=False)
        out_pose.append(p.clone())  # N, 24,3,3
        print(out_pose[i].shape)
        out_tran.append(trans[i].clone())  # N, 3
        out_shape.append(torch.zeros(p.shape[0], 10))  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(accs[i].clone())  # N, 6, 3
        out_vrot.append(oris[i].clone())  # N, 6, 3, 3
        
    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.dipimu_dir, 'pose.pt'))
    data = torch.load(os.path.join(paths.dipimu_dir, 'pose.pt'))
    print(data[:2])
    torch.save(out_shape, os.path.join(paths.dipimu_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.dipimu_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(paths.dipimu_dir, 'joint.pt'))
    torch.save(out_vrot, os.path.join(paths.dipimu_dir, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(paths.dipimu_dir, 'vacc.pt'))


# TotalCapture 数据集处理
'''
读取并处理原始的加速度、方向和姿势数据，确保它们的帧长度匹配。  --- .pkl文件
读取 gt_skel_gbl_pos.txt 文件中的位置信息，并将其转换为平移数据。(未找到)
对平移数据与加速度数据进行对齐，确保它们的长度一致。
使用运动学模型计算虚拟加速度，并去除加速度的偏差。
保存最终的加速度、方向、姿势和平移数据。   --- .pt文件
'''
def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    joints, shapes = [], []  #新增：存储joint 和 shape数据
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):    #索引排序
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
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
    for idx, (iacc, pose, tran) in enumerate(zip(accs, poses, trans)):
        poses[idx] = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)  # 就地修改

        print("poses.shape:", poses[0].shape)  # 期望 (batch, 24, 3, 3)
        print("pose.shape:", pose.shape)

        grot, joint, vert = body_model.forward_kinematics(poses[idx], tran=tran, calc_mesh=True)
        joints.append(joint[:, :24].contiguous().clone())  # 存储前24个关节


        # 计算 shape（假设 TotalCapture 没有提供 shape，用零张量填充）
        shape = torch.zeros(pose.shape[0], 10)  # 10 是 SMPL 的 shape 维度
        shapes.append(shape)

        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'vacc': accs, 'vort': oris, 'pose': poses, 'tran': trans, 'joint': joints, 'shape': shapes},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)



"""
TotalCapture数据集拆分为多个文件"""
def split_totalcapture_data():
    # 加载原始数据
    data_path = os.path.join(paths.totalcapture_dir, 'test.pt')
    data = torch.load(data_path)

    # 创建输出目录
    output_dir = os.path.join(paths.totalcapture_dir, 'split_actions')
    os.makedirs(output_dir, exist_ok=True)

    # 获取动作数量
    num_actions = len(data['vacc'])

    # 方案1：按数据类型保存（与DIP-IMU一致）
    for key in ['vacc', 'vort', 'pose', 'tran', 'joint', 'shape']:
        torch.save(data[key], os.path.join(output_dir, f'{key}.pt'))

    # # 方案2：每个动作保存为单独文件（推荐）
    # for i in range(num_actions):
    #     # 提取当前动作的所有数据
    #     action_data = {
    #         'acc': data['acc'][i],
    #         'ori': data['ori'][i],
    #         'pose': data['pose'][i],
    #         'tran': data['tran'][i]
    #     }
    #
    #     # 保存为action_{i}.pt
    #     file_name = f'action_{i + 1:04d}.pt'
    #     torch.save(action_data, os.path.join(output_dir, file_name))

    print(f"TotalCapture数据共有{num_actions}个动作，保存至：{output_dir}")


def process_from_npy():
    accs, oris, poses, trans = [], [], [], []

    # 存储不同类型的数据
    imu_test_files = []
    smpl24_testji_files = []

    # 遍历目录
    for folder in paths.raw_npy_dir:
        npy_files = glob.glob(os.path.join(folder, '*.npy'))

        for file in npy_files:
            filename = os.path.basename(file)
            if filename.endswith('imu_test.npy'):
                imu_test_files.append(file)
            elif filename.endswith('SMPL24_test.npy'):
                smpl24_testji_files.append(file)

    # 处理 imu 数据
    imu_data = [np.load(f, allow_pickle=True) for f in imu_test_files]
    for data in imu_data:
        data = data.tolist()
        for arr in data:
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr)
            print(type(arr), arr.shape)

            acc = arr[:, :18].float().reshape(-1, 6, 3)
            accs.append(acc)

            ori = arr[:, 18:].float().reshape(-1, 6, 3, 3)
            oris.append(ori)

    # 处理 SMPL 数据
    smpl24_data = [np.load(f, allow_pickle=True) for f in smpl24_testji_files]
    for data in smpl24_data:
        data = data.tolist()
        for arr in data:
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr)
            print(type(arr), arr.shape)
            poses.append(arr)
            # 创建与姿态数据形状匹配的平移数据
            trans.append(torch.zeros(arr.shape[0], 3))

    assert len(accs) == len(poses), "IMU和姿势数据数量不匹配"
    assert len(accs) == len(trans)
    length = len(accs)

    # 进行运动学推理和保存
    out_shape, out_tran, out_joint = [], [], []
    for i in tqdm(range(length)):
        grot, joint = body_model.forward_kinematics(poses[i], calc_mesh=False)
        out_shape.append(torch.zeros(poses[i].shape[0], 10))
        out_joint.append(joint[:, :24].contiguous().clone())
        # 关键修复：将trans[i]添加到out_tran中
        out_tran.append(trans[i])  # 添加这一行！

    # 保存数据
    torch.save(poses, os.path.join(paths.npy_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(paths.npy_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.npy_dir, 'tran.pt'))  # 现在有数据了
    torch.save(out_joint, os.path.join(paths.npy_dir, 'joint.pt'))
    torch.save(oris, os.path.join(paths.npy_dir, 'vrot.pt'))
    torch.save(accs, os.path.join(paths.npy_dir, 'vacc.pt'))


if __name__ == '__main__':
    process_amass()
    # #process_dipimu()
    # #process_totalcapture()
    #
    # #split_totalcapture_data()
    # process_from_npy()




    # pose.pt
    # 的第一个元素是绝对的，其余元素都是相对的
    #
    # tran.pt 始终代表绝对的全局位移
    #
    # vacc.pt(加速度) - 绝对坐标
    #
    # vacc.pt(加速度) - 绝对坐标
    #
    # joint.pt  - 绝对坐标
    #




