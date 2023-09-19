import os
import sys
import time
import tqdm
import torch
import numpy as np
import articulate as art
from config import paths, joint_set


class PoseJitterEvaluator:
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
        return torch.stack([errs[4] / 100, errs[5] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['Jitter Error (100m/s^3)', 'Jitter Error GT(100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
            

def eval(path):
    evaluator = PoseJitterEvaluator()
    
    poses = torch.load(path)
    offline_errs = []
    for i in tqdm.tqdm(range(len(poses))):
        pose_gt = poses[i]
        offline_errs.append(evaluator.eval(pose_gt, pose_gt))
        
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    

if __name__ == '__main__':
    eval('data/work/AMASS/pose.pt')