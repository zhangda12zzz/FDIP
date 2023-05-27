import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用
import sys
import time
import tqdm
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from GGIP.gan_archi import GAN_model_GIP
from GGIP.datasets_eval import ImuMotionDataEval
import option_parser
from option_parser import try_mkdir
from model.tp_rnn import TransPoseNet
from config import paths, joint_set
import articulate as art

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator('articulate/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', joint_mask=torch.tensor([1, 2, 16, 17]))

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
    evaluator = PoseEvaluator()


    dataset = ImuMotionDataEval(args)

    model = GAN_model_GIP(args, dataset, log_path='GGIP/logs/tmp')
    model.models.pose_encoder.eval()    # GAIP
    model.models.auto_encoder.eval()    # Transpose
    model.load(epoch=224, suffix='GGIP/checkpoints/gaip')  # GAIP的好结果：40 => 224[jerk减少，tc不行了] => 270
    # model.load(epoch=300, suffix='GGIP/checkpoints/gaip_spaticalM')  # GAIP的消融：v1[50], v2[160]
    # model.load(epoch=70, suffix='GGIP/checkpoints/transpose')
    # model.load(epoch=160, suffix='GGIP/checkpoints/gaip_uniformM')
    
    offline_errs = []
    online_errs = []
    past_frame = 20
    future_frame = 5

    pre_poses, gt_poses, pre_pose_onlines, gt_pose_onlines = [],[],[],[]
    
    
    imu, pose, root = dataset.getValData()
    for i in tqdm.tqdm(range(len(imu))):
        imu_test = imu[i].unsqueeze(0).to(device)
        pose_test = pose[i].unsqueeze(0).to(device)
        root_test = root[i].unsqueeze(0).to(device)
        
        # offline
        test_data = [imu_test, pose_test, root_test]
        loss, gt_pose, pre_pose = model.SMPLtest(test_data)
        offline_errs.append(evaluator.eval(pre_pose, gt_pose))   # 比较经过四元数变换的数据是否和原本一样，结论：一样
        
        # online
        # frame = imu_test.shape[1]
        # gt_pose_online = []
        # pre_pose_online = []
        # for t in range(frame-past_frame-future_frame):
        #     imu_test_tmp = imu_test[:,t:t+past_frame+future_frame]
        #     pose_test_tmp = pose_test[:,t:t+past_frame+future_frame]
        #     root_test_tmp = root_test[:,t:t+past_frame+future_frame]
        #     test_data_tmp = [imu_test_tmp, pose_test_tmp, root_test_tmp]
        #     _, gt_pose_tmp, pre_pose_tmp = model.SMPLtest(test_data_tmp)
        #     gt_pose_online.append(gt_pose_tmp[:,past_frame])
        #     pre_pose_online.append(pre_pose_tmp[:,past_frame])
        # pre_pose_online = torch.stack(pre_pose_online)
        # gt_pose_online = torch.stack(gt_pose_online)
        # online_errs.append(evaluator.eval(pre_pose_online, gt_pose_online))
        
        pre_poses.append(pre_pose.cpu())
        gt_poses.append(gt_pose.cpu())
        # pre_pose_onlines.append(pre_pose_online.cpu())
        # gt_pose_onlines.append(gt_pose_online.cpu())
        
    # visual_res = [pre_poses, gt_poses, offline_errs, pre_pose_onlines, gt_pose_onlines, online_errs]
    # np.save('GGIP/eval/transpose-sm/singleOne-imu_res.npy', visual_res)
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    # evaluator.print(torch.stack(online_errs).mean(dim=0))

if __name__ == '__main__':
    main()
