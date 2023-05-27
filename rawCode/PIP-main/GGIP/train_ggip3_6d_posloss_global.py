import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset

from torch.utils.tensorboard import SummaryWriter

import articulate as art
import config as conf
from GGIP.dataset_ggip import ImuDataset
# from articulate.utils.torch import *
from GGIP.ggip_net import AGGRU_1, AGGRU_2, AGGRU_3, GGIP

learning_rate = 2e-4
epochs = 401
checkpoint_interval = 40
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
log_on = True
sigma = 0.025
smpl_model_func = art.ParametricModel(conf.paths.smpl_file)


def reduced_local_to_global(batch, seq, glb_reduced_pose, r6d=False, root_rotation=None, shape=None):
    if r6d:
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(batch, -1, conf.joint_set.n_reduced, 3, 3)
    else:
        glb_reduced_pose = glb_reduced_pose.view(batch, -1, conf.joint_set.n_reduced, 3, 3)
    global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(batch, glb_reduced_pose.shape[1], 24, 1, 1)
    global_full_pose[:, :, conf.joint_set.reduced] = glb_reduced_pose
    
    pose = global_full_pose.clone()
    for i in range(global_full_pose.shape[0]):
        pose[i] = smpl_model_func.inverse_kinematics_R(global_full_pose[i]).view(-1, 24, 3, 3) # 到这一步变成了相对父节点的相对坐标
    pose[:, :, conf.joint_set.ignored] = torch.eye(3, device=pose.device)
    
    if root_rotation is not None:
        pose[:, :, 0:1] = root_rotation.view(batch, -1, 1, 3, 3)       # 第一个是全局根节点方向
    
    pose = pose.view(-1,24,3,3).contiguous() #[n,t,24,3,3]->[nt,24,3,3]
    if shape is not None:
        shape = shape.repeat(1,seq,1) #[n,1,10]
        shape = shape.view(-1,10).contiguous()
    
    _,joints_pos = smpl_model_func.forward_kinematics(pose, shape=shape)
    return pose, joints_pos # pose是smpl参数【24维度】，joints_pos是全局关节位置【24维度】


if __name__ == '__main__':
    batch_size = 64
    pretrain_epoch = 0

    # DataSet & DataLoader Setting
    # train_data_folder = ['data/dataset_work/DIP_IMU/train']
    
    # 采用分开赋值的方法构造验证集和训练集
    train_percent = 0.9
    train_data_folder = ["data/dataset_work/AMASS/train", "data/dataset_work/DIP_IMU/train"]
    # train_data_folder = ["data/dataset_work/DIP_IMU/train"]
    custom_dataset = ImuDataset(train_data_folder)
    train_size = int(len(custom_dataset) * train_percent)
    val_size = int(len(custom_dataset)) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    # train_dataset = ImuDataset(train_data_folder)
    # val_dataset = ImuDataset(val_data_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    rnn3 = AGGRU_3(24*3+16*12, 256, 15*6).to(device)
    ggip = GGIP()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn3.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # init TensorBoard Summary Writer
    writer = SummaryWriter('GGIP/log/ggip3_r6d_lossopt_global_spatical')
    
    # 预训练的设置
    # pretrain_path = 'GGIP/checkpoints/ggip3_r6d_lossopt_global/epoch_20.pkl'
    # pretrain_data = torch.load(pretrain_path)
    # rnn3.load_state_dict(pretrain_data['model_state_dict'])
    # pretrain_epoch = pretrain_data['epoch'] + 1
    # optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])
    
    train_loss_list = []
    train_loss_r6d_list = []
    train_loss_consis_list = []
    val_loss_list = []
    val_loss_r6d_list = []

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
            acc = data[0].float()                # [batch_size, max_seq, 18]
            ori = data[1].float()                # [batch_size, max_seq, 54]
            p_all = data[3].float()              # [batch_size, max_seq, 23, 3]
            pose = data[5].float()               # [batch_size, max_seq, 15, 6]
            pose_loc = data[4].float()
            beta = data[7]
            
            if beta.equal(torch.zeros(beta.shape[0],beta.shape[1],10)):
                beta = None
            
            n,t,_,_ = acc.shape
            
            # PIP 训练(need to be List)
            p_all_modify = p_all.view(-1, p_all.shape[1], 69)
            noise = sigma * torch.randn(p_all_modify.shape).float()   # 为了鲁棒添加的高斯噪声，标准差为0.4
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
            input = torch.concat((full_acc, full_ori, p_all), dim=-1).to(device)
            
            target = pose.view(-1, 15, 3, 3)
            target_r6d = art.math.rotation_matrix_to_r6d(target).view(-1,15,6)    #[ntv,6]
            target_r6d = target_r6d.view(n,t,90).to(device)
            _ ,target_pos = reduced_local_to_global(n, t, target, shape=beta)  
            # target_pos不等于p_all_modify哦，根节点不同会影响关节内容，但是在训练和测试阶段我们直接扔掉根节点
            # 只要target和logit逐渐收敛，之后后续添加根节点也能成功估计位置
            
            
            rnn3.train()
            logits = rnn3(input)
            optimizer.zero_grad()
            
            # 改了mseLoss为mean之后的计算（纯粹为了比较）
            _, logits_pos = reduced_local_to_global(n, t, logits, r6d=True, shape=beta)
            # target6d_pose_loc, target6d_pos = reduced_local_to_global(n, t, target_r6d, r6d=True, shape=beta)
            
            # TODO: target_pose_loc == logits_pose_loc ？
            # TODO: target_r6d == pose_loc.view(-1,15,6)?   YES!
            # TODO: target_r6d == logits
            logits_pos_used = logits_pos[:,conf.joint_set.reduced_pos] * 100
            target_pos_used = target_pos[:,conf.joint_set.reduced_pos].to(device) * 100
            loss = criterion(logits_pos_used, target_pos_used)
            
            loss_consis = criterion(logits_pos_used[1:], logits_pos_used[:-1])
            
            loss_r6d = criterion(logits, target_r6d)
            loss_all = loss / 100.0 + loss_r6d + loss_consis / 100.0
            
            if log_on:
                writer.add_scalar('mse_pos_step/train', loss, epoch_step)
            train_loss_list.append(loss.item())
            train_loss_r6d_list.append(loss_r6d.item())
            train_loss_consis_list.append(loss_consis.item())
            
            loss_all.backward()
            optimizer.step()
            epoch_step += 1
            
            if (batch_idx * batch_size) % 200 == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tr6dLoss: {:.6f})'.format(
                        epoch+pretrain_epoch, batch_idx * batch_size, len(train_loader.dataset), loss, loss_r6d))

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_train_loss_r6d = sum(train_loss_r6d_list) / len(train_loss_r6d_list)
        avg_train_loss_consis = sum(train_loss_consis_list) / len(train_loss_consis_list)
        if log_on:
            writer.add_scalar('mse_pos/train', avg_train_loss, epoch)
            writer.add_scalar('mse_consis/train', avg_train_loss_consis, epoch)
            writer.add_scalar('mse/train', avg_train_loss_r6d, epoch)

        test_loss = 0
        val_seq_length = 0
        
        rnn3.eval()
        with torch.no_grad():
            epoch_step = 0
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].float()                # [batch_size, max_seq, 18]
                ori_val = data_val[1].float()                # [batch_size, max_seq, 54]
                p_all_val = data_val[3].float()              # [batch_size, max_seq, 23, 3]
                pose_val = data_val[5].float()               # [batch_size, max_seq, 15, 6]
                beta_val = data_val[7]
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
                input_val = torch.concat((full_acc_val, full_ori_val, p_all_val), dim=-1).to(device)
                
                target_val = pose_val.view(-1, 15, 3, 3)
                target_r6d_val = art.math.rotation_matrix_to_r6d(target_val).view(-1,15,6)    #[ntv,6]
                target_r6d_val = target_r6d_val.view(n,t,90).to(device)
                
                logits_val = rnn3(input_val)
            
                # 损失计算
                loss_val = criterion(logits_val, target_r6d_val)
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
            checkpoint = {"model_state_dict": rnn3.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch+pretrain_epoch}
            path_checkpoint = "GGIP/checkpoints/ggip3_r6d_lossopt_global_spatical/epoch_{}.pkl".format(epoch+pretrain_epoch)
            torch.save(checkpoint, path_checkpoint)
            
    # checkpoint = {"model_state_dict": rnn1.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "epoch": epochs+pretrain_epoch-1}
    # path_checkpoint = "./checkpoints/trial0129/rnn1/checkpoint_final_{}_epoch_{}.pkl".format(epochs+pretrain_epoch-1, avg_val_loss)
    # torch.save(checkpoint, path_checkpoint)