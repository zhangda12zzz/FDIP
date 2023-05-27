import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset

from torch.utils.tensorboard import SummaryWriter

import config as conf
from GGIP.dataset_ggip import ImuDataset
# from articulate.utils.torch import *
from GGIP.ggip_net import AGGRU_1, AGGRU_2, AGGRU_3

learning_rate = 2e-4
epochs = 300
checkpoint_interval = 20
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
log_on = True
sigma = 0.025


class cos_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        r'''
            x,y: [n,t,15*9]
        '''
        n,t,_ = x.shape
        x = x.view(n,t,15,3,3)
        y = y.view(n,t,15,3,3)
        all = n*t*15
        x = x.view(all,3,3)
        y = y.view(all,3,3)
        loss = x.new_zeros(all,1)
        for i in range(all):
            loss[i] = abs(torch.acos((torch.trace(x[i].transpose(0,1).matmul(y[i]))-1)/2))
        return torch.mean(loss)


if __name__ == '__main__':
    batch_size = 128
    pretrain_epoch = 0

    # DataSet & DataLoader Setting
    # train_data_folder = ['data/dataset_work/DIP_IMU/train']
    
    # 采用分开赋值的方法构造验证集和训练集
    train_percent = 0.9
    train_data_folder = ["data/dataset_work/AMASS/train", "data/dataset_work/DIP_IMU/train"]
    # val_data_folder = ["data/dataset_work/DIP_IMU/train"]
    custom_dataset = ImuDataset(train_data_folder)
    train_size = int(len(custom_dataset) * train_percent)
    val_size = int(len(custom_dataset)) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    # train_dataset = ImuDataset(train_data_folder)
    # val_dataset = ImuDataset(val_data_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    rnn3 = AGGRU_3(24*3+16*12, 256, 15*9).to(device)

    criterion = nn.MSELoss()
    criterion_cos = cos_loss()
    optimizer = optim.Adam(rnn3.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # init TensorBoard Summary Writer
    writer = SummaryWriter('GGIP/log/ggip3_cosloss')
    
    # 预训练的设置
    # pretrain_path = 'GGIP/checkpoints/ggip3/epoch_190.pkl'
    # pretrain_data = torch.load(pretrain_path)
    # rnn3.load_state_dict(pretrain_data['model_state_dict'])
    # pretrain_epoch = pretrain_data['epoch'] + 1
    # optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])
    
    train_loss_list = []
    train_loss_cos_list = []
    val_loss_list = []
    val_loss_cos_list = []

    for epoch in range(epochs):
        epoch_step = 0
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
            acc = data[0].to(device).float()                # [batch_size, max_seq, 18]
            ori = data[1].to(device).float()                # [batch_size, max_seq, 54]
            # p_leaf = data[2].to(device).float()             # [batch_size, max_seq, 5, 3]
            p_all = data[3].to(device).float()              # [batch_size, max_seq, 23, 3]
            # pose = data[4].to(device).float()               # [batch_size, max_seq, 15, 6]
            pose = data[5].to(device).float()               # [batch_size, max_seq, 15, 3,3]
            n,t,_,_ = acc.shape
            
            # PIP 训练(need to be List)
            p_all_modify = p_all.view(-1, p_all.shape[1], 69)
            noise = sigma * torch.randn(p_all_modify.shape).to(device).float()   # 为了鲁棒添加的高斯噪声，标准差为0.4
            p_all_noise =  p_all_modify + noise
            p_all = p_all_noise.view(p_all_noise.shape[0], p_all_noise.shape[1], 23, 3)

            p_all = torch.cat((p_all.new_zeros(p_all.shape[0], p_all.shape[1], 1, 3), p_all), -2).view(p_all.shape[0], p_all.shape[1], 72)
            
            acc = acc.view(n,t,6,3)
            ori = ori.view(n,t,6,9)
            full_acc = acc.new_zeros(n, t, 16, 3)
            full_ori = ori.new_zeros(n, t, 16, 9)
            imu_pos = [0,4,5,11,14,15]
            full_acc[:,:,imu_pos] = acc
            full_ori[:,:,imu_pos] = ori
            full_acc = full_acc.view(n,t,-1)
            full_ori = full_ori.view(n,t,-1)
            input = torch.concat((full_acc, full_ori, p_all), dim=-1)
            
            target = pose.view(-1, pose.shape[1], 15*9)
            
            rnn3.train()
            logits = rnn3(input)
            optimizer.zero_grad()
            
            # 改了mseLoss为mean之后的计算（纯粹为了比较）
            loss = criterion(logits, target)
            # loss_cos = criterion_cos(logits, target)
            
            if log_on:
                writer.add_scalar('mse_step/train', loss, epoch_step)
                # writer.add_scalar('cos_step/train', loss_cos, epoch_step)
            train_loss_list.append(loss)
            # train_loss_cos_list.append(loss_cos)
                
            # loss_all = 1 * loss + 1 * loss_cos
            
            loss.backward()
            optimizer.step()
            epoch_step += 1
            
            if (batch_idx * batch_size) % 200 == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f})'.format(
                        epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), loss))

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        # avg_train_cos_loss = sum(train_loss_cos_list) / len(train_loss_cos_list)
        if log_on:
            writer.add_scalar('mse/train', avg_train_loss, epoch)
            # writer.add_scalar('cos/train', avg_train_cos_loss, epoch)

        test_loss = 0
        val_seq_length = 0
        
        rnn3.eval()
        with torch.no_grad():
            epoch_step = 0
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()                # [batch_size, max_seq, 18]
                ori_val = data_val[1].to(device).float()                # [batch_size, max_seq, 54]
                # p_leaf_val = data_val[2].to(device).float()             # [batch_size, max_seq, 5, 3]
                p_all_val = data_val[3].to(device).float()              # [batch_size, max_seq, 23, 3]
                # pose_val = data_val[4].to(device).float()               # [batch_size, max_seq, 15, 6]
                pose_val = data_val[5].to(device).float()               # [batch_size, max_seq, 15, 3,3]
                # tran_val = data_val[6].to(device).float()               # [batch_size, max_seq, 3]
                n,t,_,_ = acc_val.shape
                
                # PIP 训练(need to be List)
                p_all_val = torch.cat((p_all_val.new_zeros(p_all_val.shape[0], p_all_val.shape[1], 1, 3), p_all_val), -2).view(p_all_val.shape[0], p_all_val.shape[1], 72)
                
                acc_val = acc_val.view(n,t,6,3)
                ori_val = ori_val.view(n,t,6,9)
                full_acc_val = acc_val.new_zeros(n, t, 16, 3)
                full_ori_val = ori_val.new_zeros(n, t, 16, 9)
                imu_pos = [0,4,5,11,14,15]       
                full_acc_val[:,:,imu_pos] = acc_val
                full_ori_val[:,:,imu_pos] = ori_val
                full_acc_val = full_acc_val.view(n,t,-1)
                full_ori_val = full_ori_val.view(n,t,-1)
                input_val = torch.concat((full_acc_val, full_ori_val, p_all_val), dim=-1)
                
                target_val = pose_val.view(-1, pose_val.shape[1], 15*9)
                
                logits_val = rnn3(input_val)

                # 损失计算
                loss_val = criterion(logits_val, target_val)
                # loss_cos_val = criterion_cos(logits_val, target_val)
                if log_on:
                    writer.add_scalar('mse_step/val', loss_val, epoch_step)
                    # writer.add_scalar('cos_step/val', loss_cos_val, epoch_step)
                    
                epoch_step += 1
                val_loss_list.append(loss_val)
                # val_loss_cos_list.append(loss_cos_val)
                
            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            # avg_val_cos_loss = sum(val_loss_cos_list) / len(val_loss_cos_list)
            if log_on:
                writer.add_scalar('mse/val', avg_val_loss, epoch)
                # writer.add_scalar('cos/val', avg_val_cos_loss, epoch)
                
            # if avg_val_loss < val_best_loss:
            #     val_best_loss = avg_val_loss
            #     print('\nval loss reset')
            #     checkpoint = {"model_state_dict": rnn1.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "epoch": epoch+pretrain_epoch}
            #     if epoch > 0:
            #         path_checkpoint = "./checkpoints/trial0129/rnn1/best__{}_epoch_{}.pkl".format(epoch+pretrain_epoch, avg_val_loss)
            #         torch.save(checkpoint, path_checkpoint)


        if (epoch+pretrain_epoch) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": rnn3.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch+pretrain_epoch}
            path_checkpoint = "GGIP/checkpoints/ggip3_cosloss/epoch_{}.pkl".format(epoch+pretrain_epoch)
            torch.save(checkpoint, path_checkpoint)
            
    # checkpoint = {"model_state_dict": rnn1.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "epoch": epochs+pretrain_epoch-1}
    # path_checkpoint = "./checkpoints/trial0129/rnn1/checkpoint_final_{}_epoch_{}.pkl".format(epochs+pretrain_epoch-1, avg_val_loss)
    # torch.save(checkpoint, path_checkpoint)