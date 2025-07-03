"""
PoseJitterEvaluator 类：姿态预测误差评估工具

主要功能：
1. 全局姿态误差评估：
   - 计算预测姿态与真实姿态之间的多维度误差指标，包括：
     - SIP误差（运动一致性误差）
     - 关节角度误差（度数）
     - 关节位置误差（厘米）
     - 三维网格顶点误差（厘米）
     - 运动抖动误差（加速度变化率，单位 100m/s³）

2. 数据处理流程：
   - 输入姿态数据标准化：将姿态旋转矩阵调整为统一格式
   - 忽略预定义的无关关节（如手掌、手指等）
   - 调用 articulate 库的运动评估模块计算误差

3. 输出格式：
   - 返回误差的平均值与标准差
   - 支持直接打印格式化评估结果

核心指标说明：
- **SIP Error**：运动一致性误差，衡量预测姿态与真实运动模式的匹配度
- **Angular Error**：关节角度绝对误差（度数）
- **Positional/Mesh Error**：关节/网格顶点位置误差（厘米）
- **Jitter Error**：预测姿态的加速度抖动误差，反映运动平滑性
"""
import os
import sys
import time
import tqdm
import torch
import numpy as np
import articulate as art
from config import paths, joint_set, paths


#评估全局姿态误差
class PoseJitterEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        r'''
        返回【蒙面关节全局角度误差】【关节全局角度误差】【关节位置错误】【顶点位置错误】*100 【预测运动抖动】/100
        '''
        pose_p = pose_p.clone().view(-1, 24, 3, 3)  #预测
        pose_t = pose_t.clone().view(-1, 24, 3, 3)  #真值
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        # return torch.stack([errs[4] / 100, errs[5] / 100])
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100, errs[5] / 100])
    '''
    errs[9]：SIP误差（推测是与运动相关的误差）。
    errs[3]：角度误差（度数）。
    errs[0]：位置误差（cm），乘以100后转为厘米。
    errs[1]：网格误差（cm），乘以100后转为厘米。
    errs[4] / 100：抖动误差（100m/s³），除以100后转为合适单位。
    errs[5] / 100：GT抖动误差（100m/s³），除以100后转为合适单位。
    '''


    @staticmethod
    def print(errors):
        # for i, name in enumerate(['Jitter Error (100m/s^3)', 'Jitter Error GT(100m/s^3)']):
        #     print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)', 'Jitter Error GT(100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
        #errors[i, 0] 是误差的平均值，errors[i, 1] 是误差的标准差。

def eval(path):
    evaluator = PoseJitterEvaluator()
    
    poses = torch.load(path)
    offline_errs = []
    for i in tqdm.tqdm(range(len(poses))):
        pose_gt = poses[i]
        offline_errs.append(evaluator.eval(pose_gt, pose_gt))
        
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    

if __name__ == '__main__':
    eval('data/dataset_work/DIP_IMU/pose.pt')