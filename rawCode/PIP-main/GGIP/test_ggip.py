import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用

import torch
import tqdm
from config import *
from utils import *
import numpy as np
import shutil
import matplotlib.pyplot as plt
import articulate as art
from articulate.utils.rbdl import *
# from process_final import makeSingleBvh
import torch.nn as nn
from GGIP.ggip_net import GGIP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        r'''
        返回【蒙面关节全局角度误差】【关节全局角度误差】【关节位置错误】【顶点位置错误】*100 【预测运动抖动】/100
        '''
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100, errs[5] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


def run_pipeline(net, data_dir, pose_t_all, sequence_ids=None):
    r"""
    Run `net` using the imu data loaded from `data_dir`.
    Save the estimated [Pose[num_frames, 24, 3, 3], Tran[num_frames, 3]] for each of `sequence_ids`.
    """
    evaluator = PoseEvaluator()

    print('Loading imu data from "%s"' % data_dir)
    accs, rots, poses, _ = torch.load(os.path.join(data_dir, 'test.pt')).values()   # 提取数据（加速度、方向、姿势）
    init_poses = [art.math.axis_angle_to_rotation_matrix(_[0]) for _ in poses]  # 提取初始帧的pose

    if sequence_ids is None:
        sequence_ids = list(range(len(accs)))

    offline_errs = []
    # print('Saving the results at "%s"' % output_dir)
    for i in tqdm.tqdm(sequence_ids):
        a_pose = net.predictPose_single(accs[i], rots[i], preprocess=False, normalized=False)
        
        pose_t = pose_t_all[i]
        # poses[i] == pose_t  YES!!!
        
        # 如果是singleOne-IMU的数据，就不需要轴角转换
        pose_t = art.math.axis_angle_to_rotation_matrix(pose_t).view_as(a_pose)
        offline_errs.append(evaluator.eval(a_pose, pose_t))


    print('============== offline ================')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))


def evaluate(net, data_dir, sequence_ids=None, flush_cache=False):
    r"""
    Evaluate poses and translations of `net` on all sequences in `sequence_ids` from `data_dir`.
    `net` should implement `net.name` and `net.predict(glb_acc, glb_rot)`.
    """
    data_name = os.path.basename(data_dir)  # 数据文件夹
    result_dir = os.path.join(paths.result_dir, data_name, net.name)
    print_title('Evaluating "%s" on "%s"' % (net.name, data_name))

    _, _, pose_t_all, tran_t_all = torch.load(os.path.join(data_dir, 'test.pt')).values()   # 加载数据（pose和tran的真值）

    if sequence_ids is None:
        sequence_ids = list(range(len(pose_t_all))) # 构造遍历列表（和数据长度一致）
    if flush_cache and os.path.exists(result_dir):
        shutil.rmtree(result_dir)   # 清缓存（如果需要的话、把之前的结果删了）

    if True:    # 如果有没跑完的结果
        run_pipeline(net, data_dir, pose_t_all, sequence_ids=sequence_ids)    # 开始训练！输入（网络net，数据文件夹data_dir，还没跑完的序列missing_ids）




if __name__ == '__main__':
    with torch.no_grad():
        net = GGIP()
        net.eval()

        print('\n')
        # evaluate(net, paths.dipimu_dir, flush_cache=False)
        # evaluate(net, paths.totalcapture_dir, flush_cache=False)
        # evaluate(net, 'data/dataset_work/SingleOne-IMU', flush_cache=False)
        
        torch.save(net.state_dict(), 'GGIP/checkpoints/ggip_all_6d_optloss_spatial.pt')
