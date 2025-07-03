import glob
import os
import cv2
import sys
import numpy as np
import chumpy as ch
import pickle as pkl

from SMPL.smpl.smpl_webuser.serialization import load_model
from SMPL.smpl.smpl_webuser.lbs import global_rigid_transformation

MODEL_PATH = '../SMPL/smpl/models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl'
model_male = load_model(MODEL_PATH % 'm')
model_female = load_model(MODEL_PATH % 'f')

Jdirs_male = np.dstack([model_male.J_regressor.dot(model_male.shapedirs[:,:,i]) for i in range(10)])
Jdirs_female = np.dstack([model_female.J_regressor.dot(model_male.shapedirs[:,:,i]) for i in range(10)])

VERTEX_IDS = [1962, 5431, 1096, 4583, 412, 3021]
TARGET_FPS = 60
SMPL_IDS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]

def get_ori_accel(A_global_list, vertex, frame_rate):
    """
    计算全局坐标系下的关节旋转矩阵和加速度
    :param A_global_list: 全局旋转矩阵列表
    :param vertex: 顶点位置列表
    :param frame_rate: 帧率
    :return: 旋转矩阵列表和加速度列表
    """
    orientation = []
    acceleration = []

    for a_global in A_global_list:
        ori_left_arm = a_global[18][:3, :3].r
        ori_right_arm = a_global[19][:3, :3].r
        ori_left_leg = a_global[4][:3, :3].r
        ori_right_leg = a_global[5][:3, :3].r
        ori_head = a_global[15][:3, :3].r
        ori_root = a_global[0][:3, :3].r

        ori_tmp = []
        ori_tmp.append(ori_left_arm)
        ori_tmp.append(ori_right_arm)
        ori_tmp.append(ori_left_leg)
        ori_tmp.append(ori_right_leg)
        ori_tmp.append(ori_head)
        ori_tmp.append(ori_root)

        orientation.append(np.array(ori_tmp))

    time_interval = 1.0 / frame_rate
    total_number = len(A_global_list)
    for idx in range(1, total_number-1):
        vertex_0 = vertex[idx-1].astype(float)
        vertex_1 = vertex[idx].astype(float)
        vertex_2 = vertex[idx+1].astype(float)
        accel_tmp = (vertex_2 + vertex_0 - 2*vertex_1) / (time_interval*time_interval)

        acceleration.append(accel_tmp)

    return orientation[1:-1], acceleration

def compute_imu_data(gender, betas, poses, frame_rate):
    """
    计算IMU数据（旋转矩阵和加速度）
    :param gender: 性别
    :param betas: SMPL形状参数
    :param poses: 姿态参数
    :param frame_rate: 帧率
    :return: 旋转矩阵列表和加速度列表
    """
    if gender == 'male':
        Jdirs = Jdirs_male
        model = model_male
    else:
        Jdirs = Jdirs_female
        model = model_female

    betas[:] = 0
    J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(model.v_template.r)

    A_global_list = []
    for idx, p in enumerate(poses):
        (_, A_global) = global_rigid_transformation(p, J_onbetas, model.kintree_table, xp=ch)
        A_global_list.append(A_global)

    vertex = []
    for idx, p in enumerate(poses):
        model.pose[:] = p
        model.betas[:] = 0
        model.betas[:10] = betas
        tmp =  model.r[VERTEX_IDS]


    orientation, acceleration = get_ori_accel(A_global_list, vertex, frame_rate)

    return orientation, acceleration


def findNearest(t, t_list):
    """
    找到时间t在时间列表t_list中最接近的两个时间点
    :param t: 目标时间
    :param t_list: 时间列表
    :return: 最接近的两个时间点的索引
    """
    list_tmp = np.array(t_list) - t
    list_tmp = np.abs(list_tmp)
    index = np.argsort(list_tmp)[:2]
    return index


def interpolation_integer(poses_ori, fps):
    """
    整数倍插值，将原始帧率转换为目标帧率
    :param poses_ori: 原始姿态数据
    :param fps: 原始帧率
    :return: 插值后的姿态数据
    """
    poses = []
    n_tmp = int(fps / TARGET_FPS)
    poses_ori = poses_ori[::n_tmp]

    for t in poses_ori:
        poses.append(t)

    return poses

def interpolation(poses_ori, fps):
    """
    非整数倍插值，将原始帧率转换为目标帧率
    :param poses_ori: 原始姿态数据
    :param fps: 原始帧率
    :return: 插值后的姿态数据
    """
    poses = []
    total_time = len(poses_ori) / fps
    times_ori = np.arange(0, total_time, 1.0 / fps)
    times = np.arange(0, total_time, 1.0 / TARGET_FPS)

    for t in times:
        index = findNearest(t, times_ori)
        a = poses_ori[index[0]]
        t_a = times_ori[index[0]]
        b = poses_ori[index[1]]
        t_b = times_ori[index[1]]

        if t_a == t:
            tmp_pose = a
        elif t_b == t:
            tmp_pose = b
        else:
            tmp_pose = a + (b-a)*((t_b-t)/(t_b-t_a))
        poses.append(tmp_pose)

    return poses


def generate_data(pkl_path, res_path):
    """
    从pkl文件中提取姿态参数并生成IMU数据
    :param pkl_path: 输入pkl文件路径
    :param res_path: 输出pkl文件路径
    """
    if os.path.exists(res_path):
        return

    with open(pkl_path,'rb') as fin:
        data_in = pkl.load(fin, encoding='latin1')

    data_out = {}
    data_out['gender'] = data_in['gender']
    data_out['betas'] = np.array(data_in['betas'][:10])

    fps_ori = data_in['frame_rate']
    if (fps_ori % TARGET_FPS) == 0:
        data_out['poses'] = interpolation_integer(data_in['poses'], fps_ori)
    else:
        data_out['poses'] = interpolation(data_in['poses'], fps_ori)

    data_out['ori'], data_out['acc'] = compute_imu_data(data_out['gender'], data_out['betas'], data_out['poses'], TARGET_FPS)

    data_out['poses'] = data_out['poses'][1:-1]

    for fdx in range(0, len(data_out['poses'])):
        pose_tmp = []
        for jdx in SMPL_IDS:
            tmp = data_out['poses'][fdx][jdx*3:(jdx+1)*3]
            tmp = cv2.Rodrigues(tmp)[0].flatten().tolist()
            pose_tmp = pose_tmp + tmp

        data_out['poses'][fdx] = []
        data_out['poses'][fdx] = pose_tmp

    print("First entry of data_out:")
    first_entry = {key: value[0] if isinstance(value, list) or isinstance(value, np.ndarray) else value for key, value in data_out.items()}
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(first_entry)

    with open(res_path, 'wb') as fout:
            pkl.dump(data_out, fout)
    print( pkl_path )
    print( res_path )
    print( len(data_out['acc']) )
    print( '' )


def main(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有 .pkl 文件
    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

    for pkl_path in pkl_files:
        # 生成输出文件路径
        base_name = os.path.basename(pkl_path)
        res_data_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_synthesis.pkl")

        # 处理每个文件
        generate_data(pkl_path, res_data_path)

if __name__ == '__main__':
    input_dir = r"F:\CodeForPaper\Dataset\HEva"  # 输入目录
    output_dir = r"F:\CodeForPaper\Dataset\HEva_synthesis"  # 输出目录

    main(input_dir, output_dir)
