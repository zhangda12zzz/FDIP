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
    # train_data_folder = ["data/dataset_work/AMASS/train","data/dataset_work/DIP_IMU/train"]
    # val_data_folder = ["data/dataset_work/TotalCapture/train"]
    # train_dataset = ImuDataset(train_data_folder)
    # val_dataset = ImuDataset(val_data_folder)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 采用划分方法构造训练集+验证集
    # train_percent = 0.8
    # train_data_folder = ["data/dataset_work/DIP_IMU/train", "data/dataset_work/TotalCapture/train"] # "data/dataset_work/AMASS/train", 
    # custom_dataset = ImuDataset(train_data_folder)
    # train_size = int(len(custom_dataset) * train_percent)
    # val_size = int(len(custom_dataset)) - train_size
    # train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    
    criterion = nn.MSELoss(reduction="sum").to(device)

    val_best_loss = 10000.0
    test_loss = 0
    
    # 预训练的设置
    # pretrain_path = 'checkpoints/trial1016/net/net__checkpoint_30_epoch.pkl'  # amasstrial中的final     11.0607 评价：虽然训练loss没上面低，但是验证loss相似
    # pretrain_data = torch.load(pretrain_path)
    # net.load_state_dict(pretrain_data['model_state_dict'])
    # pretrain_epoch = pretrain_data['epoch']+1
    # optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])

    # for epoch in range(epochs):
    #     for batch_idx, data in enumerate(train_loader):
    #         '''
    #         data include:
    #             > sequence length  (int)
    #             > 归一化后的 acc （6*3）                          
    #             > 归一化后的 ori （6*9）                          
    #             > 叶关节和根的相对位置 p_leaf （5*3）               
    #             > 所有关节和根的相对位置 p_all （23*3）             
    #             > 所有非根关节相对于根关节的 6D 旋转 pose （15*6）    
    #             > 根关节旋转 p_root （9）（就是ori） 
    #             > 根关节位置 tran (3)              
    #         '''     
    #         acc = data[0].to(device).float()                # [batch_size, max_seq, 18]  batch_size=1
    #         ori = data[1].to(device).float()                # [batch_size, max_seq, 54]
    #         p_leaf = data[2].to(device).float()             # [batch_size, max_seq, 5, 3]
    #         p_all = data[3].to(device).float()              # [batch_size, max_seq, 23, 3]
    #         pose = data[4].to(device).float()               # [batch_size, max_seq, 15, 6]
    #         # r_root = data[5].to(device).float()             # [batch_size, max_seq, 9]
    #         # tran = data[6].to(device).float()               # [batch_size, max_seq, 3]
            
    #         # PIP
    #         x = list(torch.cat((acc, ori), -1))
    #         lj_init = list(p_leaf[:,0].view(-1, 15))
    #         jvel_init = list(torch.rand(acc.shape[1], 72).to(device))        # 只是为了debug
    #         input = list(zip(x, lj_init, jvel_init))   #[1,72+15]
            
    #         target = pose.view(-1, pose.shape[1], 90)              # [batch_size, max_seq, 90]
    #         # Transpose
    #         # input = torch.cat((acc, ori), -1).squeeze(0)                   # [batch_size, max_seq, 72]
    #         # target = pose.view(-1, pose.shape[1], 90).squeeze(0)               # [batch_size, max_seq, 90]

    #         logits = net(input)[2][0]
            
    #         logits = net._reduced_glb_6d_to_full_local_mat(ori.view(-1, 6, 3, 3)[:, -1], logits)
    #         target = net._reduced_glb_6d_to_full_local_mat(ori.view(-1, 6, 3, 3)[:, -1], target)
            
    #         # 损失计算
    #         # loss_mat = criterion(logits, target).to(device)
    #         # seq_len = acc.shape[1]
    #         # loss = loss_mat / seq_len
    #         # if (batch_idx * batch_size) % 200 == 0:
    #         #         print('Train Epoch: {} [{}/{}]\tLoss: {:.6f})'.format(
    #         #             epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), loss.item()))
            
    #         loss = evaluator.eval(logits, target)
    #         if (batch_idx * batch_size) % 200 == 0:
    #             print('Train Epoch: {} [{}/{}]\t'.format(epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset)))
    #             evaluator.print(loss)


    #     offline_errs = []
        
    #     net.eval()
    #     with torch.no_grad():
    #         for batch_idx_val, data_val in enumerate(val_loader):
    #             acc_val = data_val[0].to(device).float()                # [batch_size, max_seq, 18]
    #             ori_val = data_val[1].to(device).float()                # [batch_size, max_seq, 54]
    #             p_leaf_val = data_val[2].to(device).float()             # [batch_size, max_seq, 5, 3]
    #             p_all_val = data_val[3].to(device).float()              # [batch_size, max_seq, 23, 3]
    #             pose_val = data_val[4].to(device).float()               # [batch_size, max_seq, 15, 6]
    #             # r_root_val = data_val[5].to(device).float()             # [batch_size, max_seq, 9]
    #             # tran_val = data_val[6].to(device).float()               # [batch_size, max_seq, 3]
                
    #             # PIP
    #             x_val = list(torch.cat((acc_val, ori_val), -1))
    #             lj_init_val = list(p_leaf_val[:,0].view(-1, 15))
    #             jvel_init_val = list(torch.rand(acc_val.shape[1], 72).to(device))    # debug
    #             input_val = list(zip(x_val, lj_init_val, jvel_init_val))   #[1,72+15]
                
    #             target_val = pose_val.view(-1, pose_val.shape[1], 90)                 # [batch_size, max_seq, 90]
    #             # Transpose
    #             # input_val = torch.cat((acc_val, ori_val), -1).squeeze(0)                   # [batch_size, max_seq, 72]
    #             # target_val = pose_val.view(-1, pose_val.shape[1], 90).squeeze(0)               # [batch_size, max_seq, 90]
                
    #             logits_val = net(input_val)[2][0]
            
    #             logits_val = net._reduced_glb_6d_to_full_local_mat(ori_val.view(-1, 6, 3, 3)[:, -1], logits_val)
    #             target_val = net._reduced_glb_6d_to_full_local_mat(ori_val.view(-1, 6, 3, 3)[:, -1], target_val)

    #             # 损失计算
    #             # loss_mat_val = criterion(logits_val[2][0], target_val[0]).to(device)
    #             # val_seq_length += acc_val.shape[1]
    #             # test_loss += loss_mat_val
                
    #             offline_errs.append(evaluator.eval(logits_val, target_val))

    #         # test_loss /= val_seq_length
    #         # print('\nVAL set: Average loss: {:.4f} \n'.format(test_loss))
    #         print('\nVAL set: Average loss:\n')
    #         evaluator.print(torch.stack(offline_errs).mean(dim=0))
            
            
    # 测试集
    data_name = os.path.basename(conf.paths.dipimu_dir)  # 数据文件夹
    result_dir = os.path.join(conf.paths.result_dir, data_name, net.name)
    print('Evaluating "%s" on "%s"' % (net.name, data_name))
    acc_t_all, ori_t_all, pose_t_all, tran_t_all = torch.load(os.path.join(conf.paths.dipimu_dir, 'test.pt')).values()   # 加载数据（pose和tran的真值）
    
    # x = list(torch.cat((acc_t_all, ori_t_all), -1))
    # lj_init = list(pose_t_all[:,0].view(-1, 15))
    # jvel_init = list(torch.rand(acc.shape[1], 72).to(device))        # 只是为了debug
    # input = list(zip(x, lj_init, jvel_init))   #[1,72+15]
    
    offline_errs = []
    for i in range(len(acc_t_all)):
        
        pose_t = pose_t_all[i]
        pose_t = art.math.axis_angle_to_rotation_matrix(pose_t).view(-1, 24, 3, 3)
        
        acc = acc_t_all[i]
        ori = ori_t_all[i]
        initial_pose = pose_t[0]
        
        logits = net.predictPose(acc, ori, initial_pose)
        
        loss = evaluator.eval(logits, pose_t)
        
        print('Epoch: {}\t'.format(i))
        evaluator.print(loss)
        
        offline_errs.append(loss)
        
    print('\nVal:')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
