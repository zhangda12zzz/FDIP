import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

import config as conf
from dataset_batch import ImuDataset
from articulate.utils.torch import *

learning_rate = 0.00001
epochs = 5
checkpoint_interval = 1
sigma = 0.04   # tp推荐的是0.04，但是这岂不是直接比数据还大了
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    batch_size = 256
    pretrain_epoch = 0

    # 采用分开赋值的方法构造验证集和训练集
    train_data_folder = ["data/dataset_work/DIP_IMU/train"]
    # train_data_folder = ["data/dataset_work/AMASS/train"]
    val_data_folder = ["data/dataset_work/TotalCapture/train"]
    train_dataset = ImuDataset(train_data_folder)
    val_dataset = ImuDataset(val_data_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 采用划分方法构造训练集+验证集
    # train_percent = 0.8
    # train_data_folder = ["data/dataset_work/AMASS/train","data/dataset_work/DIP_IMU/train", "data/dataset_work/TotalCapture/train"]
    # custom_dataset = ImuDataset(train_data_folder)
    # train_size = int(len(custom_dataset) * train_percent)
    # val_size = int(len(custom_dataset)) - train_size
    # train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    rnn2 = RNN(input_size=72 + conf.joint_set.n_leaf * 3,
                        output_size=conf.joint_set.n_full * 3,
                        hidden_size=256,
                        num_rnn_layer=2,
                        bidirectional=True,
                        dropout=0.4).float().to(device)
    # rnn2 = RNNTP(conf.joint_set.n_leaf * 3 + 72, conf.joint_set.n_full * 3, 256).float().to(device)

    criterion = nn.MSELoss(reduction="sum").to(device)
    # criterion = my_mse_loss().to(device)
    optimizer = optim.Adam(rnn2.parameters(), lr=learning_rate)
    
    # writer = SummaryWriter('runs/trial4/rnn2_PIP')

    val_best_loss = 0.024627
    
    # 预训练的设置
    pretrain_path = 'checkpoints/trial0129/rnn2/best_118_epoch_0.021980210397439654.pkl'  # amasstrial中的final     11.0607 评价：虽然训练loss没上面低，但是验证loss相似
    pretrain_data = torch.load(pretrain_path)
    rnn2.load_state_dict(pretrain_data['model_state_dict'])
    pretrain_epoch = pretrain_data['epoch']+1
    optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])
    
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
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
            acc = data[0].to(device).float()                # [batch_size, max_seq, 18]  batch_size=1
            ori = data[1].to(device).float()                # [batch_size, max_seq, 54]
            p_leaf = data[2].to(device).float()             # [batch_size, max_seq, 5, 3]
            p_all = data[3].to(device).float()              # [batch_size, max_seq, 23, 3]
            # pose = data[4].to(device).float()               # [batch_size, max_seq, 15, 6]
            # r_root = data[5].to(device).float()             # [batch_size, max_seq, 9]
            # tran = data[6].to(device).float()               # [batch_size, max_seq, 3]
            
            # input = torch.cat((acc, ori), -1).squeeze(0)                   # [batch_size, max_seq, 72]
            # target = p_leaf.view(-1, p_leaf.shape[1], 15).squeeze(0)       # [batch_size, max_seq, 15]
            
            p_leaf_modify = p_leaf.view(-1, p_leaf.shape[1], 15)
            noise = sigma * torch.randn(p_leaf_modify.shape).to(device).float()   # 为了鲁棒添加的高斯噪声，标准差为0.4
            p_leaf_noise =  p_leaf_modify + noise
            
            # PIP
            input = list(torch.cat((p_leaf_noise, acc, ori), -1))      # [max_seq, 15+72=87]
            target = p_all.view(-1, p_all.shape[1], 69)          # [max_seq, 69]
            
            # Transpose
            # input = torch.cat((p_leaf_noise, acc, ori), -1).squeeze(0)       # [max_seq, 15+72=87]
            # target = p_all.view(-1, p_all.shape[1], 69).squeeze(0)           # [max_seq, 69]

            rnn2.train()
            
            logits = rnn2(input)
            optimizer.zero_grad()
            
            # 损失计算
            loss = criterion(torch.stack(logits), target).to(device)
            # seq_len = p_leaf.shape[1]
            # loss = loss_mat / seq_len
             
            # loss.backward()
            # optimizer.step()
            
            # if (batch_idx * batch_size) % 200 == 0:
            #         print('Train Epoch: {} [{}/{}]\tLoss: {:.6f})'.format(
            #             epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), loss.item()))
            a_loss = loss.item() / (batch_size * target.shape[1])   # 计算平均一帧的mseLoss
            train_loss_list.append(a_loss)
            
            loss.backward()
            optimizer.step()
            
            if (batch_idx * batch_size) % 200 == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f})'.format(
                        epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), a_loss))

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        # writer.add_scalar('rnn2_mse/train', avg_train_loss, epoch)

        # test_loss = 0
        val_seq_length = 0
        
        rnn2.eval()
        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()                # [batch_size, max_seq, 18]
                ori_val = data_val[1].to(device).float()                # [batch_size, max_seq, 54]
                p_leaf_val = data_val[2].to(device).float()             # [batch_size, max_seq, 5, 3]
                p_all_val = data_val[3].to(device).float()              # [batch_size, max_seq, 23, 3]
                # pose_val = data_val[4].to(device).float()               # [batch_size, max_seq, 15, 6]
                # r_root_val = data_val[5].to(device).float()             # [batch_size, max_seq, 9]
                # tran_val = data_val[6].to(device).float()               # [batch_size, max_seq, 3]
                
                p_leaf_modify_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15)
                noise_val = sigma * torch.randn(p_leaf_modify_val.shape).to(device).float()   # 为了鲁棒添加的高斯噪声，标准差为0.4
                p_leaf_noise_val =  p_leaf_modify_val + noise_val

                # PIP
                input_val = list(torch.cat((p_leaf_noise_val, acc_val, ori_val), -1))       # [max_seq, 15+72=87]
                target_val = p_all_val.view(-1, p_all_val.shape[1], 69)               # [max_seq, 69]                 
                
                # Transpose
                # input_val = torch.cat((p_leaf_noise_val, acc_val, ori_val), -1).squeeze(0)       # [max_seq, 15+72=87]
                # target_val = p_all_val.view(-1, p_all_val.shape[1], 69).squeeze(0)           # [max_seq, 69]                 
                
                logits_val = rnn2(input_val)
                # 损失计算
                loss_val = criterion(torch.stack(logits_val), target_val).to(device)
                # val_seq_length += p_leaf_val.shape[1]
                # test_loss += loss_mat_val
                a_loss_val = loss_val.item() / (batch_size * target_val.shape[1])
                val_loss_list.append(a_loss_val)
            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            # writer.add_scalar('rnn2_mse/val', avg_val_loss, epoch)
                

            # test_loss /= val_seq_length
            # print('\nVAL set: Average loss: {:.4f} \n'.format(test_loss))

            if avg_val_loss < val_best_loss:
                val_best_loss = avg_val_loss
                print('\nval loss reset')
                checkpoint = {"model_state_dict": rnn2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch+pretrain_epoch}
                if epoch > 0:
                    path_checkpoint = "./checkpoints/trial0129/rnn2/best_{}_epoch_{}.pkl".format(epoch+pretrain_epoch, avg_val_loss)
                    torch.save(checkpoint, path_checkpoint)

        

        if epoch % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": rnn2.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch+pretrain_epoch}
            path_checkpoint = "./checkpoints/trial0129/rnn2/checkpoint_{}_epoch_{}.pkl".format(epoch+pretrain_epoch, avg_val_loss)
            torch.save(checkpoint, path_checkpoint)
            
    # checkpoint = {"model_state_dict": rnn2.state_dict(),
    #                   "optimizer_state_dict": optimizer.state_dict(),
    #                   "epoch": epochs+pretrain_epoch-1}
    # path_checkpoint = "./checkpoints/trial0129/rnn2/rnn2__checkpoint_final_{}_epoch_{}.pkl".format(epochs+pretrain_epoch-1, avg_val_loss)
    # torch.save(checkpoint, path_checkpoint)