import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用
import sys
import time
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from model.architecture_GIP import GAN_model_GIP
from dataset.dataset_CIP import ImuMotionData
import option_parser
from option_parser import try_mkdir
from model.tp_rnn import TransPoseNet
from config import paths, joint_set
import articulate as art

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator('data/SMPLmodel/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', joint_mask=torch.tensor([1, 2, 16, 17]))

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
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)', 'Jitter Error GT(100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


def main():
    args = option_parser.get_args()
    args.dataset = 'Smpl'
    args.is_train = False
    device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
    args.device = device
    # characters = get_character_names(args)  # 两组模型，对应两种拓扑 [['Smpl'], ['Aj']]
    evaluator = PoseEvaluator()


    dataset = ImuMotionData(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GAN_model_GIP(args, dataset, log_path='pretrained/logs_GIP')
    # model.load(epoch=600, suffix='pretrained/models_AGGRU')     # 320最佳
    # model.load(epoch=1850, suffix='pretrained/models_compare_pureTP_allDataTrain')
    # model.load(epoch=1000, suffix='pretrained/models_AGGRU_spatialGCN')
    
    to15Joints = [1,2,7,12,3, 8,13,15,16,19, 24,20,25,21,26]    # 按照smpl原本的标准关节顺序定义的15个躯干节点
    reduced = [0,1,2,3,4, 5,6,9,12,13, 14,16,17,18,19]          # 没有头，但是包含了根节点
    offline_errs = []

    output = []
    imu, pose, root, gt24 = dataset.getValData()
    # imu, joint, pose, root, gt24 = dataset.getValData()
    for i in range(imu.shape[0]):
        test_data = [imu[i:i+1], pose[i:i+1], root[i:i+1]]
        # test_data = [imu[i:i+1], joint[i:i+1], pose[i:i+1], root[i:i+1]]
        gt = gt24[i:i+1].to(device)
        loss, gt_pose, pre_pose = model.SMPLtest(test_data)
        
        # if i == 2 or i == 3:
        #     output.append(pre_pose.cpu())

        offline_errs.append(evaluator.eval(pre_pose, gt_pose))   # 比较经过四元数变换的数据是否和原本一样，结论：一样
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    # np.save('./dataset/CIP/result.npy', output)

if __name__ == '__main__':
    main()
