import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import config as conf
from dataset import ImuDataset
from articulate.utils.torch import *
from net import PIP, TransPoseNet
import articulate as art


learning_rate = 0.0002
epochs = 3
checkpoint_interval = 5
sigma = 0.025   # tp推荐的是0.025，但是这岂不是直接比数据还大了
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(conf.paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        r'''
        返回【蒙面关节全局角度误差】【关节全局角度误差】【关节位置错误】【顶点位置错误】*100 【预测运动抖动】/100
        '''
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, conf.joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, conf.joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
        # for i, name in enumerate(['SIP Error (deg)']):
        #     print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))



if __name__ == '__main__':
    batch_size = 1
    pretrain_epoch = 0
    evaluator = PoseEvaluator()
    net = PIP().float()#.to(device)

    # 采用分开赋值的方法构造验证集和训练集
    train_data_folder = ["data/dataset_work/TotalCapture/train"]
    train_dataset = ImuDataset(train_data_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss(reduction="sum").to(device)

    val_best_loss = 10000.0
    test_loss = 0
    pose_errors = []
    # 结论：直接用数据集的真值依然有很多无解的结果

    for batch_idx, data in enumerate(train_loader):
        '''
        data include:
            > sequence length  (int)
            > 归一化后的 acc （6*3）                          
            > 归一化后的 ori （6*9）                          
            > 叶关节和根的相对位置 p_leaf （5*3）               
            > 所有关节和根的相对位置 p_all （23*3）             
            > 所有非根关节相对于根关节的 6D 旋转 pose （15*6）    
            > 根关节旋转 p_root （9）（就是ori） 
            > 根关节位置 tran (3)              
        '''     
        acc = data[0].squeeze(0).view(-1,6,3).float()                # [batch_size, max_seq, 18]  batch_size=1
        ori = data[1].squeeze(0).view(-1,6,3,3).float()                # [batch_size, max_seq, 54]
        # p_leaf = data[2].to(device).float()             # [batch_size, max_seq, 5, 3]
        # p_all = data[3].to(device).float()              # [batch_size, max_seq, 23, 3]
        global_6d_pose = data[4].float()               # [batch_size, max_seq, 15, 6]
        # r_root = data[5].to(device).float()             # [batch_size, max_seq, 9]
        tran_t = data[6].squeeze(0).float()               # [batch_size, max_seq, 3]
        joint_velocity = data[7].float()
        contact = data[8].squeeze(0).float()
        pose_t = data[9].squeeze(0).float()

        pose = net._reduced_glb_6d_to_full_local_mat(ori[:, -1], global_6d_pose)
    
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(ori[:, -1].transpose(1, 2)) * conf.vel_scale
    
        pose_opt, tran_opt = [], []
        print("=============idx:", batch_idx)
        for p, v, c, a in zip(pose, joint_velocity, contact, acc):  # 局部旋转矩阵的姿势、局部关节速度、脚地接触概率、全局加速度
            p, t = net.dynamics_optimizer.optimize_frame(p, v, c, a)   # 传入动态优化器逐帧优化，返回优化后的pose和tran
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        
        loss = evaluator.eval(pose_opt, pose_t)
        pose_errors.append(loss)

        if (batch_idx * batch_size) % 200 == 0:
            print('Epoch: {}\t'.format(batch_idx))
            evaluator.print(loss)

    print('\nVal:')
    evaluator.print(torch.stack(pose_errors).mean(dim=0))

            
    # # 测试集
    # data_name = os.path.basename(conf.paths.dipimu_dir)  # 数据文件夹
    # result_dir = os.path.join(conf.paths.result_dir, data_name, net.name)
    # print('Evaluating "%s" on "%s"' % (net.name, data_name))
    # acc_t_all, ori_t_all, pose_t_all, tran_t_all = torch.load(os.path.join(conf.paths.dipimu_dir, 'test.pt')).values()   # 加载数据（pose和tran的真值）
    
    # # x = list(torch.cat((acc_t_all, ori_t_all), -1))
    # # lj_init = list(pose_t_all[:,0].view(-1, 15))
    # # jvel_init = list(torch.rand(acc.shape[1], 72).to(device))        # 只是为了debug
    # # input = list(zip(x, lj_init, jvel_init))   #[1,72+15]
    
    # offline_errs = []
    # for i in range(len(acc_t_all)):
        
    #     pose_t = pose_t_all[i]
    #     pose_t = art.math.axis_angle_to_rotation_matrix(pose_t).view(-1, 24, 3, 3)
        
    #     acc = acc_t_all[i]
    #     ori = ori_t_all[i]
    #     initial_pose = pose_t[0]
        
    #     logits = net.predictPose(acc, ori, initial_pose)
        
    #     loss = evaluator.eval(logits, pose_t)
        
    #     print('Epoch: {}\t'.format(i))
    #     evaluator.print(loss)
        
    #     offline_errs.append(loss)
        
    # print('\nVal:')
    # evaluator.print(torch.stack(offline_errs).mean(dim=0))
