import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import config as conf
from dataset import ImuDataset
from articulate.utils.torch import *

learning_rate = 0.0002
epochs = 25
checkpoint_interval = 5
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    batch_size = 1
    pretrain_epoch = 0

    # 采用划分方法构造训练集+验证集
    train_percent = 0.8
    train_data_folder = ["data/dataset_work/AMASS/train","data/dataset_work/DIP_IMU/train", "data/dataset_work/TotalCapture/train"]
    custom_dataset = ImuDataset(train_data_folder)
    train_size = int(len(custom_dataset) * train_percent)
    val_size = int(len(custom_dataset)) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    rnn1_pip_dict = 'data/weights/rnnSon/pip_rnn1.pt'
    rnn1 = RNNWithInit(input_size=72,
                            output_size=conf.joint_set.n_leaf * 3,
                            hidden_size=256,
                            num_rnn_layer=2,
                            dropout=0.4).float().to(device)
    rnn1.load_state_dict(torch.load(rnn1_pip_dict))

    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = optim.Adam(rnn1.parameters(), lr=learning_rate)
    
    val_best_loss = 10000.0
    
    
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
            acc = data[0].to(device).float()                # [batch_size, max_seq, 18]
            ori = data[1].to(device).float()                # [batch_size, max_seq, 54]
            p_leaf = data[2].to(device).float()             # [batch_size, max_seq, 5, 3]
            # p_all = data[3].to(device).float()              # [batch_size, max_seq, 23, 3]
            # pose = data[4].to(device).float()               # [batch_size, max_seq, 15, 6]
            # r_root = data[5].to(device).float()             # [batch_size, max_seq, 9]
            # tran = data[6].to(device).float()               # [batch_size, max_seq, 3]
            
            # PIP 训练(need to be List)
            x = list(torch.cat((acc, ori), -1))
            lj_init = list(p_leaf[:,0].view(-1, 15))
            input = list(zip(x, lj_init))   #[1,72+15]
            target = list(p_leaf.view(-1, p_leaf.shape[1], 15))       # [batch_size, max_seq, 15]
            
            # Transpose 训练(need to be Tensor)
            # input = torch.cat((acc, ori), -1).squeeze(0)                   # [batch_size, max_seq, 72]
            # target = p_leaf.view(-1, p_leaf.shape[1], 15).squeeze(0)       # [batch_size, max_seq, 15]
            
            rnn1.eval()
            logits = rnn1(input)
            
            # 损失计算
            # PIP使用target[0]，transpose使用target
            # if(batch_size == 1):
            loss_mat = criterion(logits[0], target[0]).to(device)   # loss_mat = criterion(logits[0], target).to(device)
            seq_len = acc.shape[1]
            loss = loss_mat / seq_len
            
            if (batch_idx * batch_size) % 200 == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f})'.format(
                        epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), loss.item()))
                    
        test_loss = 0
        val_seq_length = 0
        
        rnn1.eval()
        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()                # [batch_size, max_seq, 18]
                ori_val = data_val[1].to(device).float()                # [batch_size, max_seq, 54]
                p_leaf_val = data_val[2].to(device).float()             # [batch_size, max_seq, 5, 3]
                # p_all_val = data_val[3].to(device).float()              # [batch_size, max_seq, 23, 3]
                # pose_val = data_val[4].to(device).float()               # [batch_size, max_seq, 15, 6]
                # r_root_val = data_val[5].to(device).float()             # [batch_size, max_seq, 9]
                # tran_val = data_val[6].to(device).float()               # [batch_size, max_seq, 3]
                
                # PIP 训练(need to be List)
                x_val = list(torch.cat((acc_val, ori_val), -1))
                lj_init_val = list(p_leaf_val[:,0].view(-1, 15))
                input_val = list(zip(x_val, lj_init_val))   #[1,72+15]
                target_val = list(p_leaf_val.view(-1, p_leaf_val.shape[1], 15))       # [batch_size, max_seq, 15]
                
                # Transpose 训练(need to be Tensor)
                # input_val = torch.cat((acc_val, ori_val), -1).squeeze(0)                   # [batch_size, max_seq, 72]
                # target_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15).squeeze(0)       # [batch_size, max_seq, 15]
            
                logits_val = rnn1(input_val)

                # 损失计算
                # if(batch_size == 1):
                loss_mat_val = criterion(logits_val[0], target_val[0]).to(device)   # loss_mat_val = criterion(logits_val[0], target_val).to(device)
                val_seq_length += acc_val.shape[1]
                test_loss += loss_mat_val

            test_loss /= val_seq_length
            print('\nVAL set: Average loss: {:.4f} \n'.format(test_loss))
            
            # 0.7089

