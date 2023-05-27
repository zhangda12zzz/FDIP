import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用
import sys
import time
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from model import create_model, create_CIPmodel
from dataset import create_dataset_CIP
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
    # characters = get_character_names(args)  # 两组模型，对应两种拓扑 [['Smpl'], ['Aj']]
    evaluator = PoseEvaluator()


    dataset = create_dataset_CIP(args, std_paths='dataset/CIP/work/CIP_std_22.bvh')
    data_loader = DataLoader(dataset, batch_size=1)

    model = create_CIPmodel(args, dataset, std_paths='dataset/CIP/work/CIP_std_22.bvh')
    model.load(epoch=0, suffix='pretrained/models_CIP_refine')

    net = TransPoseNet(num_past_frame=20, num_future_frame=5).to(device)
    net.load_state_dict(torch.load(paths.weights_file))
    net.eval()
    
    to15Joints = [1,2,7,12,3, 8,13,15,16,19, 24,20,25,21,26]    # 按照smpl原本的标准关节顺序定义的15个躯干节点
    reduced = [0,1,2,3,4, 5,6,9,12,13, 14,16,17,18,19]          # 没有头，但是包含了根节点
    offline_errs = []
    
    smplGTmotion = np.load('dataset/CIP/work/Smpl_amass_test_motion_SMPL24.npy', allow_pickle=True)
    
    for step, motions in enumerate(data_loader):    # motion: [n,42,64]+[n,87,64]
        imu = motions[0]
        imu = imu.permute(0,2,1).to(device)
        
        motion = motions[1]
        motion_denorm = model.dataset.denorm_motion(motion)
        
        motion_preprocess_local, motion_preprocess = net.calculatePose(imu)  #[n,t,72]
        motion_preprocess_local = torch.Tensor(motion_preprocess_local).squeeze(0)
        motion_gt_local = torch.Tensor(smplGTmotion[step])[:motion_preprocess_local.shape[0]]
        # 已经验证 motion_preprocess_local 和transpose的输出基本一致，接下来的任务就是在此基础上进行优化
        
        motion_preprocess = torch.Tensor(motion_preprocess).permute(0,2,1).to(device)
        test_input = [motion_preprocess, motion]
        res_fullrot, gt_fullrot, loss, loss2 = model.SMPLtest(test_input)
        
        gt_fullRot = model.comput_fullRotate(motion_preprocess)   # 直接处理SMPLtest的输入数据，不经过refine网络
        
        fullrot = torch.eye(3).view(1,1,3,3).repeat(gt_fullRot.shape[0],24,1,1)
        fullrot[:,reduced] = gt_fullRot[:,to15Joints]
        fullrot[:,15] = motion_gt_local[:,15]
        # offline_errs.append(evaluator.eval(motion_preprocess_local, motion_gt_local))
        offline_errs.append(evaluator.eval(fullrot, motion_gt_local))   # 比较经过四元数变换的数据是否和原本一样，结论：一样
    evaluator.print(torch.stack(offline_errs).mean(dim=0))

if __name__ == '__main__':
    main()
