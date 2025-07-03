"""
体来说是一个 AGGRU_1 模型，用于处理 IMU（惯性测量单元）数据。目标是基于 IMU 传感器数据（加速度计和方向数据）
预测人体叶关节的相对位置。脚本包括数据加载、模型训练、验证和检查点保存。

主要组成部分：
环境设置：

脚本将 CUDA 设备设置为 '0'，以便在 GPU 可用时进行加速，否则默认使用 CPU。
导入库：

导入了必要的库，如 torch、torch.nn、torch.optim 和 torch.utils.data。
从本地文件导入了自定义模块，如 config、ImuDataset 和 AGGRU_1。
超参数设置：

learning_rate：设置为 1e-4。
epochs：总训练轮数设置为 200。
checkpoint_interval：每 4 个 epoch 保存一次检查点。
device：自动检测并设置为 GPU（如果可用）。
log_on：一个标志，用于启用 TensorBoard 日志记录。
数据加载：

从指定的文件夹（train_data_folder）加载数据集。
使用 90-10 的比例将数据集拆分为训练集和验证集。
为训练集和验证集创建了 DataLoader 对象。
模型初始化：

使用特定的输入、隐藏和输出维度初始化 AGGRU_1 模型。
模型被移动到指定的设备（GPU 或 CPU）。
损失函数和优化器：

使用均方误差（MSE）作为损失函数。
使用 Adam 优化器进行训练，指定了学习率和 betas 参数。
TensorBoard 日志记录：

初始化了一个 SummaryWriter，用于记录训练和验证的指标，以便在 TensorBoard 中进行可视化。
预训练：

可以使用检查点文件（pretrain_path）对模型进行预训练。模型的状态和优化器的状态从该检查点加载。
训练循环：

训练循环遍历指定的 epoch 数量。
对于每个数据批次，模型通过以下步骤进行训练：
输入数据（加速度计和方向数据）被拼接并传递给模型。
计算模型的输出与目标（叶关节的相对位置）之间的损失。
反向传播并更新模型参数。
记录训练损失并定期打印训练进度。
验证循环：

每个 epoch 结束后，模型在验证集上进行评估。
计算验证损失并记录到 TensorBoard 中。
检查点保存：

每隔 checkpoint_interval 个 epoch，保存模型的检查点，包括模型的状态、优化器的状态和当前的 epoch 数。
日志记录：

训练和验证的损失被记录到 TensorBoard 中，以便后续分析和可视化。
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset

from torch.utils.tensorboard import SummaryWriter     # 用于可视化训练过程，日志记录

import config as conf
from data.dataset_posReg import ImuDataset
from model.net import AGGRU_1, AGGRU_2, AGGRU_3

import time
from model.architecture import GAN_model_GIP
from data.dataset_poseReg import ImuMotionData
import option_parser
from option_parser import try_mkdir
import articulate as art


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(r'F:\CodeForPaper\Ka_GAIP\data\SMPLmodel\basicmodel_m_lbs_10_207_0_v1.0.0.pkl', joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        r'''
        返回【蒙面关节全局角度误差】【关节全局角度误差】【关节位置错误】【顶点位置错误】*100 【预测运动抖动】/100
        '''

        pose_p = art.math.r6d_to_rotation_matrix(pose_p.clone()).view(-1, 24, 3, 3)
        pose_t = art.math.r6d_to_rotation_matrix(pose_t.clone()).view(-1, 24, 3, 3)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))

learning_rate = 1e-4

checkpoint_interval = 10
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
log_on = True

if __name__ == '__main__':
    batch_size = 64
    pretrain_epoch = 0
    epochs = 1         #200

    # 采用分开赋值的方法构造验证集和训练集
    train_percent = 0.9
    train_data_folder = [
        "F:\CodeForPaper\Dataset\AMASS\HumanEva\pt",
        "F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_6\Detail"
    ]
    custom_dataset = ImuDataset(train_data_folder)

    train_size = int(len(custom_dataset) * train_percent)
    val_size = int(len(custom_dataset)) - train_size

    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])    # 随机分割数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  #封装数据集

    train_batches_count = len(train_loader)
    print(f"Number of batches in train_loader: {train_batches_count}\n")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    rnn1 = AGGRU_1(6*9, 256, 5*3).to(device)    # 输入维度为6*9（6个IMU，9个特征(3+6)），隐藏层维度为256，输出维度为5*3（5个关节，3个维度）

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn1.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # init TensorBoard Summary Writer
    writer = SummaryWriter('log/ggip1')


    """
    预训练模型加载
    """
    # pretrain_path = 'GGIP/checkpoints/ggip1/epoch_190.pkl'
    # pretrain_data = torch.load(pretrain_path)
    # rnn1.load_state_dict(pretrain_data['model_state_dict'])
    # pretrain_epoch = pretrain_data['epoch'] + 1
    # optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])

    save_dir = "GGIP/checkpoints/ggip1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        print('\n===== AGGRU_1 Training Epoch: {} ============================================================'.format(epoch + pretrain_epoch + 1))
        train_loss_list = []
        val_loss_list = []

        epoch_step = 1
        for batch_idx, data in enumerate(train_loader):
            '''
            data include:
                > sequence length  (int)
                > 归一化后的 acc （6*3）                          
                > 归一化后的 ori （6*9）                          
                > out_rot_6d  6D 旋转  （6*6）    
                > 叶关节和根的相对位置 p_leaf （5*3）               
                > 所有关节和根的相对位置 p_all （23*3）                               
                > out_pos   pose         （15*3*3）     
                > 所有非根关节相对于根关节的 6D 旋转 pose （15*6）  out_pos_6d    
                > 根关节旋转 p_root （9）（就是ori） 
                > 根关节位置 tran (3)              
            '''
            acc = data[0].to(device).float()                # [batch_size, max_seq, 18]
            ori_6d = data[2].to(device).float()                # [batch_size, max_seq, 54]
            p_leaf = data[3].to(device).float()             # [batch_size, max_seq, 5, 3]

            # PIP 训练(need to be List)    拼接为[batch_size, max_seq, 54]
            x = torch.cat((acc, ori_6d), -1)
            input = x.view(x.shape[0], x.shape[1], -1) #[n,t,54]   6*(3+6)
            target = p_leaf.view(-1, p_leaf.shape[1], 15)   #输出展平

            rnn1.train()
            logits = rnn1(input)
            optimizer.zero_grad()

            # MSE
            loss = torch.sqrt(criterion(logits, target).to(device))
            if log_on:
                writer.add_scalar('mse_step/train', loss, epoch_step)
            train_loss_list.append(loss)

            loss.backward()
            optimizer.step()
            epoch_step += 1

            log_interval = len(train_loader) // 10  # 每10%打印一次
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + pretrain_epoch + 1, min((batch_idx + 1) * batch_size, len(train_loader.dataset)), len(train_loader.dataset), loss))

        #每个epoch结束后，进行平均训练损失，进行一次验证集的测试。
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        print('Average Training Loss: {:.6f}'.format(avg_train_loss))

        if log_on:
            writer.add_scalar('mse/train', avg_train_loss, epoch+1)



        # test_loss = 0
        # val_seq_length = 0

        rnn1.eval()
        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()                # [batch_size, max_seq, 18]
                ori_val = data_val[2].to(device).float()                # [batch_size, max_seq, 54]
                p_leaf_val = data_val[3].to(device).float()             # [batch_size, max_seq, 5, 3]

                # PIP 训练(need to be List)
                x_val = torch.cat((acc_val, ori_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15)       # [batch_size, max_seq, 15]
                
                logits_val = rnn1(input_val)

                # 损失计算MSE
                loss_val = torch.sqrt(criterion(logits_val, target_val).to(device))

                val_loss_list.append(loss_val)
            #计算并记录平均验证损失
            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            print('Val：Average Validation Loss: {:.6f}\n'.format(avg_val_loss))
            if log_on:
                writer.add_scalar('mse/val', avg_val_loss, epoch+1)

        # 保存检查点
        if (epoch+1+pretrain_epoch) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": rnn1.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch+1+pretrain_epoch}
            path_checkpoint = "./GGIP/checkpoints/ggip1/epoch_{}.pkl".format(epoch+1+pretrain_epoch)
            torch.save(checkpoint, path_checkpoint)

    # 预测整个训练集并保存（保持原始顺序）
    print("Saving predictions for the entire training dataset...")

    # 使用原始数据集（非分割后的训练集），确保顺序一致
    all_predictions = []

    rnn1.eval()
    with torch.no_grad():
        for data in train_loader:
            acc = data[0].to(device).float()
            ori_6d = data[2].to(device).float()
            x = torch.cat((acc, ori_6d), -1)
            input = x.view(x.shape[0], x.shape[1], -1)

            logits = rnn1(input)
            zeros = torch.zeros(logits.shape[:-1] + (3,), device=logits.device)
            logits_extended = torch.cat([logits, zeros], dim=-1)  # 在最后一个维度拼接3个0
            logits_extended = logits_extended.view(*logits.shape[:-1], 6, 3)
            all_predictions.append(logits_extended)

    all_predictions = torch.cat(all_predictions, dim=0)

    print(all_predictions.shape, all_predictions.dtype, all_predictions[0].shape,all_predictions[0].dtype)

    print("Saving predictions for the entire validate dataset...")

    # 使用原始数据集（非分割后的训练集），确保顺序一致
    all_predictions_val = []

    rnn1.eval()
    with torch.no_grad():
        for data in val_loader:
            acc = data[0].to(device).float()
            ori_6d = data[2].to(device).float()
            x = torch.cat((acc, ori_6d), -1)
            input = x.view(x.shape[0], x.shape[1], -1)

            logits = rnn1(input)
            zeros = torch.zeros(logits.shape[:-1] + (3,), device=logits.device)
            logits_extended = torch.cat([logits, zeros], dim=-1)  # 在最后一个维度拼接3个0
            logits_extended = logits_extended.view(*logits.shape[:-1], 6, 3)
            all_predictions_val.append(logits_extended)

    all_predictions_val = torch.cat(all_predictions_val, dim=0)
    print(all_predictions_val.shape, all_predictions_val.dtype, all_predictions_val[0].shape,all_predictions_val[0].dtype)



    # ==============================================================

    # 关闭 TensorBoard Writer
    # checkpoint = {"model_state_dict": rnn1.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "epoch": epochs+pretrain_epoch-1}
    # path_checkpoint = "./checkpoints/trial0129/rnn1/checkpoint_final_{}_epoch_{}.pkl".format(epochs+pretrain_epoch-1, avg_val_loss)
    # torch.save(checkpoint, path_checkpoint)



    """
    AGGRU_2 ==========================================================================================
    """
    epochs = 1 #300
    #sigma = 0.04  # 高斯噪声的标准差

    rnn2 = AGGRU_2(6 * 12, 256, 23 * 3).to(device)  # 12 = 3+6+3

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn2.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # init TensorBoard Summary Writer
    writer = SummaryWriter('log/ggip2')  # 日志写入目录

    # 预训练的设置
    # pretrain_path = 'GGIP/checkpoints/ggip2/epoch_60.pkl'
    # pretrain_data = torch.load(pretrain_path)
    # rnn2.load_state_dict(pretrain_data['model_state_dict'])
    # pretrain_epoch = pretrain_data['epoch'] + 1
    # optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])



    for epoch in range(epochs):

        train_loss_list = []
        val_loss_list = []

        print('\n===== AGGRU_2 Training Epoch: {} ============================================================'.format(epoch + pretrain_epoch + 1))
        epoch_step = 0
        for batch_idx, data in enumerate(train_loader):  # 按batch读取数据
            epoch_step += 1
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
            acc = data[0].to(device).float()  # [batch_size, max_seq, 18]
            ori_6d = data[2].to(device).float()  # [batch_size, max_seq, 54]
            p_all = data[4].to(device).float()  # [batch_size, max_seq, 23, 3]

            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            p_leaf_modify = all_predictions[batch_start:batch_end]

            x = torch.cat((acc, ori_6d, p_leaf_modify), -1)
            input = x.view(x.shape[0], x.shape[1], -1)
            target = p_all.view(-1, p_all.shape[1],
                                69)  # 所有关节-目标的形状为 [batch_size * max_seq, 69]，这里的 69 是每个关节的 3 个维度（位置）

            rnn2.train()
            logits = rnn2(input)
            optimizer.zero_grad()

            # 改了mseLoss为mean之后的计算（纯粹为了比较）
            loss = torch.sqrt(criterion(logits, target).to(device))
            if log_on:
                writer.add_scalar('mse_step/train', loss, epoch_step)
            train_loss_list.append(loss)

            loss.backward()
            optimizer.step()

            log_interval = len(train_loader) // 10  # 每10%打印一次
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + pretrain_epoch + 1, min((batch_idx + 1) * batch_size, len(train_loader.dataset)),
                    len(train_loader.dataset), loss))

        #每个epoch结束后，进行平均训练损失，进行一次验证集的测试。
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        print('AGGRU_2: Average Training Loss: {:.6f}'.format(avg_train_loss))


        # 计算当前epoch的平均损失
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        if log_on:
            writer.add_scalar('mse/train', avg_train_loss, epoch+1)

        rnn2.eval()    # 进入评估模式
        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()  # [batch_size, max_seq, 18]
                ori_val = data_val[2].to(device).float()  # [batch_size, max_seq, 54]
                p_all_val = data_val[4].to(device).float()  # [batch_size, max_seq, 23, 3]

                batch_start = batch_idx_val
                batch_end = (batch_idx_val + 1)
                p_leaf_modify_val = all_predictions_val[batch_start:batch_end]

                x_val = torch.cat((acc_val, ori_val, p_leaf_modify_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_all_val.view(-1, p_all_val.shape[1], 69)  # [batch_size, max_seq, 15]

                logits_val = rnn2(input_val)

                # 损失计算
                loss_val = torch.sqrt(criterion(logits_val, target_val).to(device))
                if log_on:
                    writer.add_scalar('mse_step/val', loss_val, epoch_step)

                val_loss_list.append(loss_val)

            #计算并记录平均验证损失
            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            print('Val：Average Validation Loss: {:.6f}\n'.format(avg_val_loss))
            if log_on:
                writer.add_scalar('mse/val', avg_val_loss, epoch+1)

            # if avg_val_loss < val_best_loss:
            #     val_best_loss = avg_val_loss
            #     print('\nval loss reset')
            #     checkpoint = {"model_state_dict": rnn1.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "epoch": epoch+pretrain_epoch}
            #     if epoch > 0:
            #         path_checkpoint = "./checkpoints/trial0129/rnn1/best__{}_epoch_{}.pkl".format(epoch+pretrain_epoch, avg_val_loss)
            #         torch.save(checkpoint, path_checkpoint)

        if (epoch+1 + pretrain_epoch) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": rnn2.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch+1 + pretrain_epoch}
            path_checkpoint = "GGIP/checkpoints/ggip2/epoch_{}.pkl".format(epoch+1 + pretrain_epoch)
            torch.save(checkpoint, path_checkpoint)


    # 预测整个训练集并保存（保持原始顺序）
    print("AGGRU_2: Saving predictions for the entire training dataset...")

    # 使用原始数据集（非分割后的训练集），确保顺序一致
    all_predictions_2 = []

    rnn2.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            acc = data[0].to(device).float()
            ori_6d = data[2].to(device).float()

            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            p_leaf_modify_val = all_predictions[batch_start:batch_end]

            x_val = torch.cat((acc, ori_6d, p_leaf_modify_val), -1)
            input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)

            logits = rnn2(input_val)
            zeros = torch.zeros(logits.shape[:-1] + (3,), device=logits.device)
            logits_extended = torch.cat([logits, zeros], dim=-1)  # 在最后一个维度拼接3个0
            logits_extended = logits_extended.view(*logits.shape[:-1], 24, 3)
            all_predictions_2.append(logits_extended)

    all_predictions_2 = torch.cat(all_predictions_2, dim=0)

    print(all_predictions_2.shape, all_predictions_2.dtype, all_predictions_2[0].shape,all_predictions_2[0].dtype)

    # 预测整个训练集并保存（保持原始顺序）
    print("AGGRU_2: Saving predictions for the entire validate dataset...")

    # 使用原始数据集（非分割后的训练集），确保顺序一致
    all_predictions_val_2 = []

    rnn2.eval()
    with torch.no_grad():
        for batch_idx_val, data in enumerate(val_loader):
            acc = data[0].to(device).float()
            ori_6d = data[2].to(device).float()

            batch_start = batch_idx_val
            batch_end = batch_idx_val + 1
            p_leaf_modify_val = all_predictions_val[batch_start:batch_end]

            x_val = torch.cat((acc, ori_6d, p_leaf_modify_val), -1)
            input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)

            logits = rnn2(input_val)
            zeros = torch.zeros(logits.shape[:-1] + (3,), device=logits.device)
            logits_extended = torch.cat([logits, zeros], dim=-1)  # 在最后一个维度拼接3个0
            logits_extended = logits_extended.view(*logits.shape[:-1], 24, 3)
            all_predictions_val_2.append(logits_extended)

    all_predictions_val_2 = torch.cat(all_predictions_val_2, dim=0)    #列表中的张量拼接成完整的张量
    print(all_predictions_val_2.shape, all_predictions_val_2.dtype, all_predictions_val_2[0].shape,all_predictions_val_2[0].dtype)

    """
    AGGRU_3 ==========================================================================================
    """

    evaluator = PoseEvaluator()
    #
    # # log_path = os.path.join(args.save_dir, 'logs_CIP/') # './pretrained/logs/'
    # try_mkdir(args.save_dir)
    # # try_mkdir(log_path)
    #
    # with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
    #     para_file.write(' '.join(sys.argv))  # 存储相关参数

    # dataset = ImuMotionData(args)
    # data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    #
    # args.epoch_begin = 301
    # model.load(epoch=300, suffix='train/checkpoints/expe_pretrainCompare/GAIP/k6')

    # 只加载判别器
    # model.models.discriminator.load_state_dict(torch.load(os.path.join('GGIP/checkpoints/trial9_gan_all123_ganpose/topology/500', 'discriminator.pt'),
    #                                                  map_location=device))

    # model.setup()
    #

    to15Joints = [1, 2, 7, 12, 3, 8, 13, 15, 16, 19, 24, 20, 25, 21, 26]  # 按照smpl原本的标准关节顺序定义的15个躯干节点
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19]  # 没有头，但是包含了根节点
    sip_err = 0
    ang_err = 0
    jerk_err = 0

    epochs = 1 #300
    #sigma = 0.04  # 高斯噪声的标准差

    rnn3 = AGGRU_3(6 * 9 + 24 * 3, 256, 24 * 6).to(device)  # 12 = 3+6+3

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn2.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # init TensorBoard Summary Writer
    writer = SummaryWriter('log/ggip3')  # 日志写入目录

    # 预训练的设置
    # pretrain_path = 'GGIP/checkpoints/ggip2/epoch_60.pkl'
    # pretrain_data = torch.load(pretrain_path)
    # rnn2.load_state_dict(pretrain_data['model_state_dict'])
    # pretrain_epoch = pretrain_data['epoch'] + 1
    # optimizer.load_state_dict(pretrain_data['optimizer_state_dict'])

    start_time = time.time()

    for epoch in range(epochs):

        train_loss_list = []
        val_loss_list = []

        print('\n===== AGGRU_3 Training Epoch: {} ============================================================='.format(epoch + pretrain_epoch + 1))
        epoch_step = 0
        for batch_idx, data in enumerate(train_loader):  # 按batch读取数据
            epoch_step += 1
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
            acc = data[0].to(device).float()  # [batch_size, max_seq, 6, 3]
            ori_6d = data[2].to(device).float()  # [batch_size, max_seq, 6, 6]
            pose_6d = data[6].to(device).float()  # [batch_size, max_seq, 24, 6]

            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            p_all_pos = all_predictions_2[batch_start:batch_end]

            x = torch.cat((acc, ori_6d), -1)
            input = x.view(x.shape[0], x.shape[1], -1)
            p_all_modify = p_all_pos.view(x.shape[0], x.shape[1], -1)
            input = torch.cat((input, p_all_modify), -1)
            target = pose_6d.view(pose_6d.shape[0], pose_6d.shape[1], 144)

            rnn3.train()
            logits = rnn3(input)
            optimizer.zero_grad()

            loss = torch.sqrt(criterion(logits, target).to(device))
            if log_on:
                writer.add_scalar('mse_step/train', loss, epoch_step)
            train_loss_list.append(loss)
            offline_errs = []
            offline_errs.append(evaluator.eval(logits, target))
            offline_err = torch.stack(offline_errs).mean(dim=0)

            sip_err = offline_err[0, 0]
            ang_err = offline_err[1, 0]
            jerk_err = offline_err[4, 0]

            loss.backward()
            optimizer.step()

            log_interval = len(train_loader) // 10  # 每10%打印一次
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + pretrain_epoch + 1, min((batch_idx + 1) * batch_size, len(train_loader.dataset)),
                    len(train_loader.dataset), loss))
                print('sip_err:', sip_err, 'ang_err:', ang_err, 'jerk_err:', jerk_err)

        #每个epoch结束后，进行平均训练损失，进行一次验证集的测试。
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        print('AGGRU_3: Average Training Loss: {:.6f}'.format(avg_train_loss))


        # 计算当前epoch的平均损失
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        if log_on:
            writer.add_scalar('mse/train', avg_train_loss, epoch+1)

        rnn3.eval()    # 进入评估模式
        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()  # [batch_size, max_seq, 6, 3]
                ori_val = data_val[2].to(device).float()  # [batch_size, max_seq, 6, 6]
                pose_6d_val = data_val[6].to(device).float()  # [batch_size, max_seq, 24, 6]

                batch_start = batch_idx_val
                batch_end = (batch_idx_val + 1)
                p_all_val = all_predictions_val_2[batch_start:batch_end]

                x = torch.cat((acc_val, ori_val), -1)
                input_val = x.view(x.shape[0], x.shape[1], -1)
                p_all_modify_val = p_all_val.view(x.shape[0], x.shape[1], -1)
                input_val = torch.cat((input_val, p_all_modify_val), -1)
                target_val = pose_6d_val.view(pose_6d_val.shape[0], pose_6d_val.shape[1], 144)

                logits_val = rnn3(input_val)

                # 损失计算
                loss_val = torch.sqrt(criterion(logits_val, target_val).to(device))
                if log_on:
                    writer.add_scalar('mse_step/val', loss_val, epoch_step)

                val_loss_list.append(loss_val)

            #计算并记录平均验证损失
            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            print('Val：Average Validation Loss: {:.6f}\n'.format(avg_val_loss))
            if log_on:
                writer.add_scalar('mse/val', avg_val_loss, epoch+1)

            # if avg_val_loss < val_best_loss:
            #     val_best_loss = avg_val_loss
            #     print('\nval loss reset')
            #     checkpoint = {"model_state_dict": rnn1.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "epoch": epoch+pretrain_epoch}
            #     if epoch > 0:
            #         path_checkpoint = "./checkpoints/trial0129/rnn1/best__{}_epoch_{}.pkl".format(epoch+pretrain_epoch, avg_val_loss)
            #         torch.save(checkpoint, path_checkpoint)

        if (epoch+1 + pretrain_epoch) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": rnn2.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch+1 + pretrain_epoch}
            path_checkpoint = "GGIP/checkpoints/ggip2/epoch_{}.pkl".format(epoch+1 + pretrain_epoch)
            torch.save(checkpoint, path_checkpoint)


    end_tiem = time.time()
    print('training time', end_tiem - start_time)

    # # 预测整个训练集并保存（保持原始顺序）
    # print("AGGRU_3: Saving predictions for the entire training dataset...")
    #
    # # 使用原始数据集（非分割后的训练集），确保顺序一致
    # all_predictions_2 = []
    #
    # rnn2.eval()
    # with torch.no_grad():
    #     for batch_idx, data in enumerate(train_loader):
    #         acc = data[0].to(device).float()
    #         ori_6d = data[2].to(device).float()
    #
    #         batch_start = batch_idx * batch_size
    #         batch_end = (batch_idx + 1) * batch_size
    #         p_leaf_modify_val = all_predictions[batch_start:batch_end]
    #
    #         x_val = torch.cat((acc, ori_6d, p_leaf_modify_val), -1)
    #         input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
    #
    #         logits = rnn2(input_val)
    #         zeros = torch.zeros(logits.shape[:-1] + (3,), device=logits.device)
    #         logits_extended = torch.cat([logits, zeros], dim=-1)  # 在最后一个维度拼接3个0
    #         logits_extended = logits_extended.view(*logits.shape[:-1], 24, 3)
    #         all_predictions_2.append(logits_extended)
    #
    # all_predictions_2 = torch.cat(all_predictions_2, dim=0)
    #
    # print(all_predictions_2.shape, all_predictions_2.dtype, all_predictions_2[0].shape,all_predictions_2[0].dtype)
    #
    # # 预测整个训练集并保存（保持原始顺序）
    # print("AGGRU_3: Saving predictions for the entire validate dataset...")
    #
    # # 使用原始数据集（非分割后的训练集），确保顺序一致
    # all_predictions_val_2 = []
    #
    # rnn2.eval()
    # with torch.no_grad():
    #     for batch_idx_val, data in enumerate(val_loader):
    #         acc = data[0].to(device).float()
    #         ori_6d = data[2].to(device).float()
    #
    #         batch_start = batch_idx_val
    #         batch_end = batch_idx_val + 1
    #         p_leaf_modify_val = all_predictions_val[batch_start:batch_end]
    #
    #         x_val = torch.cat((acc, ori_6d, p_leaf_modify_val), -1)
    #         input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
    #
    #         logits = rnn2(input_val)
    #         zeros = torch.zeros(logits.shape[:-1] + (3,), device=logits.device)
    #         logits_extended = torch.cat([logits, zeros], dim=-1)  # 在最后一个维度拼接3个0
    #         logits_extended = logits_extended.view(*logits.shape[:-1], 24, 3)
    #         all_predictions_val_2.append(logits_extended)
    #
    # all_predictions_val_2 = torch.cat(all_predictions_val_2, dim=0)    #列表中的张量拼接成完整的张量
    # print(all_predictions_val_2.shape, all_predictions_val_2.dtype, all_predictions_val_2[0].shape,all_predictions_val_2[0].dtype)
