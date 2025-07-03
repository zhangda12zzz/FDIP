"""
这篇代码实现了一个基于GAN（生成对抗网络）的姿势评估系统，主要用于评估从IMU（惯性测量单元）数据中预测的人体姿势与真实姿势之
间的误差。代码的主要功能包括数据加载、模型初始化、姿势预测、误差计算和结果输出。

"""



import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 设置可见的GPU设备
import sys
import time
import tqdm
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from model.architecture import GAN_model_GIP  # 导入GAN模型
from data.dataset_eval import ImuMotionDataEval  # 导入评估数据集
import option_parser  # 导入参数解析模块
from config import paths, joint_set  # 导入配置路径和关节设置
import articulate as art  # 导入articulate库用于运动评估

class PoseEvaluator:
    """
    PoseEvaluator类用于评估姿势预测结果的误差。
    """
    def __init__(self):
        # 初始化FullMotionEvaluator，用于评估运动数据
        self._eval_fn = art.FullMotionEvaluator('data/SMPLmodel/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        # 评估预测姿势和真实姿势的误差
        pose_p = pose_p.clone().view(-1, 24, 3, 3)  # 将预测姿势转换为矩阵形式
        pose_t = pose_t.clone().view(-1, 24, 3, 3)  # 将真实姿势转换为矩阵形式
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)  # 忽略特定关节
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)  # 忽略特定关节
        errs = self._eval_fn(pose_p, pose_t)  # 计算误差
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100, errs[5] / 100, errs[11] / 100, errs[12] / 100])

    @staticmethod   # 静态方法
    def print(errors):
        # 打印误差结果
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)', 'Mesh Error (cm)',
                                  'Jitter Error (100m/s^3)', 'Jitter Error GT(100m/s^3)', 'Jitter Error2 (100m/s^3)', 'Jitter Error2 GT(100m/s^3)']):
            print('%.2f (+/- %.2f)' % (errors[i, 0], errors[i, 1]))


def main():
    args = option_parser.get_args()  # 获取命令行参数
    args.dataset = 'Smpl'  # 设置数据集类型
    args.is_train = False  # 设置为评估模式
    device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')  # 设置设备
    args.device = device
    evaluator = PoseEvaluator()  # 初始化姿势评估器

    dataset = ImuMotionDataEval(args, dataset='tc')  # 加载评估数据集

    model = GAN_model_GIP(args, dataset, log_path='logs')  # 初始化GAN模型
    model.models.pose_encoder.eval()  # 设置模型为评估模式
    # 补充实验
    for epo in range(301,306):  # 遍历指定范围的epoch
        model.load(epoch=epo, suffix='train/checkpoints/expe_pretrainCompare/GAIP/doublefinetune')  # 加载模型权重

        offline_errs = []  # 离线误差列表
        online_errs = []  # 在线误差列表
        past_frame = 20  # 过去帧数
        future_frame = 5  # 未来帧数

        pre_poses, gt_poses, pre_pose_onlines, gt_pose_onlines = [],[],[],[]  # 存储预测和真实姿势

        imu, pose, root = dataset.getValData()  # 获取验证数据
        for i in tqdm.tqdm(range(len(imu))):  # 遍历每个样本
            imu_test = imu[i].unsqueeze(0).to(device)  # 将IMU数据移动到设备
            pose_test = pose[i].unsqueeze(0).to(device)  # 将姿势数据移动到设备
            root_test = root[i].unsqueeze(0).to(device)  # 将根节点数据移动到设备

            # 离线评估
            test_data = [imu_test, pose_test, root_test]  # 准备测试数据
            loss, gt_pose, pre_pose = model.SMPLtest(test_data)  # 进行测试
            offline_errs.append(evaluator.eval(pre_pose, gt_pose))  # 计算并存储离线误差

        evaluator.print(torch.stack(offline_errs).mean(dim=0))  # 打印离线误差结果
if __name__ == '__main__':
    main()
