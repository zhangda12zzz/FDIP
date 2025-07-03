r"""
    预处理DIP-IMU和TotalCapture测试数据集。
    合成AMASS数据集。   60hz

    请参考`config.py`中的`paths`并正确设置每个数据集的路径。
"""


import articulate as art
import torch
import os
import pickle     #、序列化与反序列化，加载pkl文件
from config import paths, amass_data
import numpy as np
from tqdm import tqdm
import glob       #用于查找符合特定模式的文件路径

"""
    数据读取：遍历AMASS数据集中的.npz文件，提取姿态、平移、形状参数。
    帧率处理：根据原始帧率（120/60/59）对数据降采样。
    坐标对齐：将AMASS坐标系转换为DIP坐标系（通过旋转矩阵调整）。
    运动学计算：使用SMPL模型计算关节和顶点位置。
    加速度合成：通过顶点位置差分计算加速度（_syn_acc函数）。
    数据保存：将处理后的张量保存为.pt文件
    
    pose.pt：姿态数据（轴角表示）
    shape.pt：形状参数
    tran.pt：平移向量
    joint.pt：关节3D坐标
    vrot.pt：旋转速度（关节旋转矩阵）
    vacc.pt：加速度（合成IMU加速度）
"""
def process_amass(smooth_n=4): 

    def _syn_acc(v):
        r"""
        从顶点位置合成加速度数据。
        """
        mid = smooth_n // 2           #平滑参数
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])        #顶点索引
    ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])                      #关节索引
    body_model = art.ParametricModel(paths.smpl_file)                #加载SMPL模型

    data_pose, data_trans, data_beta, length = [], [], [], []        #数据列表  
    for ds_name in amass_data:
        print('\r读取', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, ds_name, '*/*_poses.npz'))):    #匹配所有文件的列表
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])     #帧率
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))       #将数据添加到列表中
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, '未找到AMASS数据集。请检查config.py或注释掉process_amass()函数'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))    #转换为数组再转换为张量
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # 右手
    pose = pose[:, :24].clone()   # 仅使用身体部分

    # 将AMASS全局坐标系与DIP对齐
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)    #位移数据加一个维度在最后，再相乘，最后变成tran形状
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('合成IMU加速度和方向数据')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\t丢弃一个长度为', l, '的序列'); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        b += l

    print('保存数据')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
    #旋转速度
    torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
    #旋转加速度
    torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
    print('合成的AMASS数据集已保存至', paths.amass_dir)


def process_dipimu():
    """
 功能：预处理DIP-IMU测试数据，处理缺失值并裁剪有效片段。

输入数据：原始DIP-IMU数据（.pkl文件）。

输出数据：

test.pt：包含以下键值：
acc：6个IMU的加速度数据（N,6,3）
ori：6个IMU的方向数据（N,6,3,3）
pose：SMPL姿态参数（轴角表示，N,24,3）
tran：占位零向量（DIP-IMU无平移数据）

    """
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_01','s_02','s_03','s_04','s_05','s_06','s_07','s_08','s_09', 's_10']
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # 用最近邻填充nan值
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
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu不包含平移数据
            else:
                print('DIP-IMU: %s/%s 包含过多nan值！已丢弃！' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    print('预处理后的DIP-IMU数据集已保存至', paths.dipimu_dir)


def process_totalcapture():
    """
    输入数据：

DIP格式的预处理数据（.pkl）
官方原始数据（gt_skel_gbl_pos.txt）
输出数据：

test.pt：包含以下键值：
acc：6个IMU的加速度（N,6,3）
ori：6个IMU的方向（N,6,3,3）
pose：SMPL姿态参数（N,24,3）
tran：全局平移向量（N,3）

    """
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # 数据集中acc/ori和gt pose不匹配
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
                continue   # 没有SMPL姿势
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

    # 将平移数据与姿势匹配
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('预处理后的TotalCapture数据集已保存至', paths.totalcapture_dir)


if __name__ == '__main__':
    # process_amass()
    process_dipimu()
    #process_totalcapture()
