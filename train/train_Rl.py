import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset

from torch.utils.tensorboard import SummaryWriter

import config as conf
from data.dataset_posReg import ImuDataset
from model.net import AGGRU_1, AGGRU_2, AGGRU_3

learning_rate = 1e-4
epochs = 200
checkpoint_interval = 4
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
log_on = True

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

    rnn1 = AGGRU_1(6*12, 256, 5*3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn1.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # init TensorBoard Summary Writer
    writer = SummaryWriter('GGIP/log/ggip1')
    
    # 预训练的设置
    pretrain_path = 'GGIP/checkpoints/ggip1/epoch_190.pkl'
    pretrain_data = torch.load(pretrain_path)
    rnn1.load_state_dict(pretrain_data['model_state_dict'])
    pretrain_epoch = pretrain_data['epoch'] + 1
    optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])
    
    train_loss_list = []
    val_loss_list = []

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
            p_leaf = data[2].to(device).float()             # [batch_size, max_seq, 5, 3]
            # p_all = data[3].to(device).float()              # [batch_size, max_seq, 23, 3]
            # pose = data[4].to(device).float()               # [batch_size, max_seq, 15, 6]
            # r_root = data[5].to(device).float()             # [batch_size, max_seq, 9]
            # tran = data[6].to(device).float()               # [batch_size, max_seq, 3]
            
            # PIP 训练(need to be List)
            x = torch.cat((acc, ori), -1)   
            input = x.view(x.shape[0], x.shape[1], -1) #[n,t,72]
            target = p_leaf.view(-1, p_leaf.shape[1], 15)     
            
            rnn1.train()
            logits = rnn1(input)
            optimizer.zero_grad()
            
            # 改了mseLoss为mean之后的计算（纯粹为了比较）
            loss = criterion(logits, target).to(device)
            if log_on:
                writer.add_scalar('mse_step/train', loss, epoch_step)
            train_loss_list.append(loss)
            
            loss.backward()
            optimizer.step()
            epoch_step += 1
            
            if (batch_idx * batch_size) % 200 == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f})'.format(
                        epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), loss))

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        if log_on:
            writer.add_scalar('mse/train', avg_train_loss, epoch)

        test_loss = 0
        val_seq_length = 0
        
        rnn1.eval()
        with torch.no_grad():
            epoch_step = 0
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()                # [batch_size, max_seq, 18]
                ori_val = data_val[1].to(device).float()                # [batch_size, max_seq, 54]
                p_leaf_val = data_val[2].to(device).float()             # [batch_size, max_seq, 5, 3]
                # p_all_val = data_val[3].to(device).float()              # [batch_size, max_seq, 23, 3]
                # pose_val = data_val[4].to(device).float()               # [batch_size, max_seq, 15, 6]
                # r_root_val = data_val[5].to(device).float()             # [batch_size, max_seq, 9]
                # tran_val = data_val[6].to(device).float()               # [batch_size, max_seq, 3]
                
                # PIP 训练(need to be List)
                x_val = torch.cat((acc_val, ori_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15)       # [batch_size, max_seq, 15]
                
                logits_val = rnn1(input_val)

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
            checkpoint = {"model_state_dict": rnn1.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch+pretrain_epoch}
            path_checkpoint = "GGIP/checkpoints/ggip1/epoch_{}.pkl".format(epoch+pretrain_epoch)
            torch.save(checkpoint, path_checkpoint)
            
    # checkpoint = {"model_state_dict": rnn1.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "epoch": epochs+pretrain_epoch-1}
    # path_checkpoint = "./checkpoints/trial0129/rnn1/checkpoint_final_{}_epoch_{}.pkl".format(epochs+pretrain_epoch-1, avg_val_loss)
    # torch.save(checkpoint, path_checkpoint)