"""
使用了一种名为AGGRU（Attention-based GRU）的神经网络模型。代码的主要功能包括数据加载、模型训练、验证、损失计算、
模型保存以及使用TensorBoard进行日志记录。

环境设置与导入模块：

代码首先设置了可见的GPU设备，并导入了必要的Python库，如torch、torch.nn、torch.optim等。
导入了自定义的配置模块（config）、数据集类（ImuDataset）和模型类（AGGRU_1, AGGRU_2, AGGRU_3）。
参数设置：

定义了训练的超参数，如学习率（learning_rate）、训练轮数（epochs）、检查点保存间隔（checkpoint_interval）等。
设置了设备（device）为GPU（如果可用）或CPU。
数据集加载与划分：

使用ImuDataset类加载训练数据，数据集路径包括AMASS和DIP_IMU数据集。
将数据集按比例划分为训练集和验证集，使用random_split方法进行划分。
使用DataLoader加载训练集和验证集，训练集的批次大小为128，验证集的批次大小为1。
模型初始化与优化器设置：

初始化了AGGRU_2模型，并将其移动到指定的设备上。
使用均方误差损失函数（MSELoss）和Adam优化器进行模型训练。
TensorBoard日志记录：

使用SummaryWriter初始化TensorBoard日志记录器，用于记录训练和验证过程中的损失值。
训练循环：

在每个epoch中，遍历训练集数据，计算损失并进行反向传播和优化。
在训练过程中，添加高斯噪声以提高模型的鲁棒性。
每200个batch打印一次训练损失。
使用TensorBoard记录每个epoch的平均训练损失。
验证循环：

在每个epoch结束后，使用验证集数据进行模型验证。
计算验证损失，并使用TensorBoard记录每个epoch的平均验证损失。
模型保存：

每隔checkpoint_interval个epoch保存一次模型检查点，包括模型状态、优化器状态和当前epoch数。

"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset

from torch.utils.tensorboard import SummaryWriter

import config as conf
from data.dataset_posReg import ImuDataset
from model.net import AGGRU_1, AGGRU_2, AGGRU_3

learning_rate = 2e-4
epochs = 300
checkpoint_interval = 10
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
log_on = True
sigma = 0.04   # 高斯噪声的标准差

if __name__ == '__main__':
    batch_size = 64
    pretrain_epoch = 0

    # DataSet & DataLoader Setting
    # train_data_folder = ['data/dataset_work/DIP_IMU/train']
    
    # 采用分开赋值的方法构造验证集和训练集
    train_percent = 0.9
    train_data_folder = ["F:\CodeForPaper\Dataset\AMASS\HumanEva\pt","F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_6\Detail"]
    # val_data_folder = ["data/dataset_work/DIP_IMU/train"]
    custom_dataset = ImuDataset(train_data_folder) #加载和处理这两个数据集
    train_size = int(len(custom_dataset) * train_percent)
    val_size = int(len(custom_dataset)) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    # train_dataset = ImuDataset(train_data_folder)
    # val_dataset = ImuDataset(val_data_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    rnn2 = AGGRU_2(6*12, 256, 23*3).to(device)  # 12 = 3+6+3

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn2.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # init TensorBoard Summary Writer
    writer = SummaryWriter('log/ggip2')  #日志写入目录
    
    # 预训练的设置
    # pretrain_path = 'GGIP/checkpoints/ggip2/epoch_60.pkl'
    # pretrain_data = torch.load(pretrain_path)
    # rnn2.load_state_dict(pretrain_data['model_state_dict'])
    # pretrain_epoch = pretrain_data['epoch'] + 1
    # optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])
    
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        epoch_step = 0
        for batch_idx, data in enumerate(train_loader):   #按batch读取数据
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
            p_leaf = data[2].to(device).float()             # [batch_size, max_seq, 5, 3]
            p_all = data[3].to(device).float()              # [batch_size, max_seq, 23, 3]
            # pose = data[4].to(device).float()               # [batch_size, max_seq, 15, 6]
            # r_root = data[5].to(device).float()             # [batch_size, max_seq, 9]
            # tran = data[6].to(device).float()               # [batch_size, max_seq, 3]
            
            # PIP 训练(need to be List)
            p_leaf_modify = p_leaf.view(-1, p_leaf.shape[1], 15)
            noise = sigma * torch.randn(p_leaf_modify.shape).to(device).float()   # 为了鲁棒添加的高斯噪声，标准差为0.4
            p_leaf_noise =  p_leaf_modify + noise
            p_leaf = p_leaf_noise.view(p_leaf_noise.shape[0], p_leaf_noise.shape[1], 5, 3)

            p_leaf = torch.cat((p_leaf.new_zeros(p_leaf.shape[0], p_leaf.shape[1], 1, 3), p_leaf), -2)
            x = torch.cat((acc, ori, p_leaf), -1)   #[n,t,72]
            input = x.view(x.shape[0], x.shape[1], -1)
            target = p_all.view(-1, p_all.shape[1], 69)     #所有关节-目标的形状为 [batch_size * max_seq, 69]，这里的 69 是每个关节的 3 个维度（位置）
            
            rnn2.train()
            logits = rnn2(input)
            optimizer.zero_grad()
            
            # 改了mseLoss为mean之后的计算（纯粹为了比较）
            loss = criterion(logits, target).to(device)
            if log_on:
                writer.add_scalar('mse_step/train', loss, epoch_step)
            train_loss_list.append(loss)
            
            loss.backward()
            optimizer.step()
            epoch_step += 1
            
            if (batch_idx * batch_size) % 200 == 0:   #每200个batch打印一次
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f})'.format(
                        epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), loss))
        # 计算当前epoch的平均损失
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        if log_on:
            writer.add_scalar('mse/train', avg_train_loss, epoch)

        test_loss = 0
        val_seq_length = 0
        
        rnn2.eval()
        with torch.no_grad():
            epoch_step = 0
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()                # [batch_size, max_seq, 18]
                ori_val = data_val[1].to(device).float()                # [batch_size, max_seq, 54]
                p_leaf_val = data_val[2].to(device).float()             # [batch_size, max_seq, 5, 3]
                p_all_val = data_val[3].to(device).float()              # [batch_size, max_seq, 23, 3]
                # pose_val = data_val[4].to(device).float()               # [batch_size, max_seq, 15, 6]
                # r_root_val = data_val[5].to(device).float()             # [batch_size, max_seq, 9]
                # tran_val = data_val[6].to(device).float()               # [batch_size, max_seq, 3]
                
                # PIP 训练(need to be List)
                p_leaf_val = torch.cat((p_leaf_val.new_zeros(p_leaf_val.shape[0], p_leaf_val.shape[1], 1, 3), p_leaf_val), -2)
                x_val = torch.cat((acc_val, ori_val, p_leaf_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_all_val.view(-1, p_all_val.shape[1], 69)       # [batch_size, max_seq, 15]
                
                logits_val = rnn2(input_val)

                # 损失计算
                loss_val = criterion(logits_val, target_val).to(device)
                if log_on:
                    writer.add_scalar('mse_step/val', loss_val, epoch_step)
                    
                epoch_step += 1
                val_loss_list.append(loss_val)
                
            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            if log_on:
                writer.add_scalar('mse/val', avg_val_loss, epoch)
                
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
            checkpoint = {"model_state_dict": rnn2.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch+pretrain_epoch}
            path_checkpoint = "GGIP/checkpoints/ggip2/epoch_{}.pkl".format(epoch+pretrain_epoch)
            torch.save(checkpoint, path_checkpoint)
            
    # checkpoint = {"model_state_dict": rnn1.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "epoch": epochs+pretrain_epoch-1}
    # path_checkpoint = "./checkpoints/trial0129/rnn1/checkpoint_final_{}_epoch_{}.pkl".format(epochs+pretrain_epoch-1, avg_val_loss)
    # torch.save(checkpoint, path_checkpoint)