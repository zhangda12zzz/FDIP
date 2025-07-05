import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 假设模型和数据集定义在这些模块中
# 请确保这些导入路径是正确的
from data.dataset_posReg import ImuDataset
from model.net_zd import FDIP_1, FDIP_2, FDIP_3
from evaluator import PoseEvaluator

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置使用的GPU
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5      # 正则化参数，控制权重衰减
BATCH_SIZE = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOG_ENABLED = True
TRAIN_PERCENT = 0.9
BATCH_SIZE_VAL = 32
SEED = 42

# --- Paths ---
# 请确保这些路径在您的环境中是正确的
TRAIN_DATA_FOLDERS = [
    os.path.join("D:\\", "Dataset", "AMASS", "HumanEva", "pt"),
    os.path.join("D:\\", "Dataset", "DIPIMUandOthers", "DIP_6", "Detail")
]
CHECKPOINT_DIR = os.path.join("GGIP", "checkpoints")
LOG_DIR = "log"


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU
        # 确保CUDA卷积操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # True可能加速但引入不确定性
    print(f"Random seed set to {seed}")


class EarlyStopping:
    """
    如果验证损失在给定的耐心期后没有改善，则提前停止训练（仍会进行下一阶段训练）。
    MODIFIED: 现在可以保存和加载优化器状态和轮数。
    """

    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience    # 允许验证损失不改善的轮数上限
        self.verbose = verbose
        self.counter = 0      # 用于记录连续没有改善的轮数
        self.best_score = None  # 记录最佳分数（-val_loss）【最佳=最大】
        self.early_stop = False
        self.val_loss_min = np.Inf  # 记录最低验证损失
        self.delta = delta     # 定义"显著改善"的最小阈值
        self.path = path
        self.best_epoch = 0  # 记录达到最低验证损失的轮数

    def __call__(self, val_loss, model, optimizer, epoch):
        score = -val_loss  # 将损失转换为分数，越大越好
        if self.best_score is None:  # 第一次调用
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:  # 分数没有改善
            self.counter += 1
            if self.verbose:    # 打印信息
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # 达到耐心上限
                self.early_stop = True
        else:  # 分数有所改善
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0  # 重置计数器

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """当验证损失减少时保存模型、优化器和轮数。"""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving checkpoint to {self.path}...')

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,  # 当前轮数
            'model_state_dict': model.state_dict(),  # 模型参数
            'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
            'val_loss_min': val_loss,  # 当前最佳验证损失
            'best_score': self.best_score,  # 当前最佳分数
            'early_stopping_counter': self.counter  # 早停计数器状态
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch


def create_directories():
    """创建必要的目录。"""
    dirs = [
        os.path.join(CHECKPOINT_DIR, "ggip1"),
        os.path.join(CHECKPOINT_DIR, "ggip2"),
        os.path.join(CHECKPOINT_DIR, "ggip3"),
        LOG_DIR
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Directories created or already exist.")


def load_data(train_percent=TRAIN_PERCENT):
    """加载并分割数据集。"""
    print("Loading dataset...")
    try:
        # 确保ImuDataset可以处理文件夹列表
        custom_dataset = ImuDataset(TRAIN_DATA_FOLDERS)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your dataset paths and ImuDataset class are correct.")
        sys.exit(1)

    total_size = len(custom_dataset)
    train_size = int(total_size * train_percent)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))

    train_dataset = Subset(custom_dataset, train_indices)
    val_dataset = Subset(custom_dataset, val_indices)

    # 数据加载多进程设置：    推荐在Windows上将num_workers设置为0，除非您确定您的设置可以处理多进程
    # Linux/macOS通常可以设置为4或更高，取决于CPU核心数
    num_workers = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False,
        pin_memory=True, num_workers=num_workers
    )
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    return train_loader, val_loader


def train_fdip_1(model, optimizer, scheduler, train_loader, val_loader, epochs, early_stopper, start_epoch=0):
    """训练 FDIP_1 模型，支持从指定轮数开始训练"""
    print("\n=============================== Starting FDIP_1 Training =============================")
    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip1')) if LOG_ENABLED else None

    # 从 start_epoch 开始循环，end_epoch 为 epochs - 1
    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1  # 用于日志显示，从1开始
        model.train()
        train_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"FDIP_1 Epoch {current_epoch}/{epochs}", leave=True)
        for data in epoch_pbar:
            acc = data[0].to(DEVICE, non_blocking=True).float()
            ori_6d = data[2].to(DEVICE, non_blocking=True).float()
            p_leaf = data[3].to(DEVICE, non_blocking=True).float()

            x = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            target = p_leaf.view(-1, p_leaf.shape[1], 15)  # 5个叶节点，每个3D位置

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = torch.sqrt(criterion(logits, target))  # 使用RMSE

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss encountered at FDIP_1 Epoch {current_epoch}, skipping batch.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data_val in val_loader:
                acc_val = data_val[0].to(DEVICE, non_blocking=True).float()
                ori_val = data_val[2].to(DEVICE, non_blocking=True).float()
                p_leaf_val = data_val[3].to(DEVICE, non_blocking=True).float()

                x_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                target_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15)
                logits_val = model(x_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val))
                val_losses.append(loss_val.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0  # 处理空列表情况
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0  # 处理空列表情况
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'FDIP_1 Epoch {current_epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}')

        if LOG_ENABLED and writer:
            writer.add_scalars('loss/fdip1', {'train': avg_train_loss, 'val': avg_val_loss}, current_epoch)
            writer.add_scalar('learning_rate/fdip1', current_lr, current_epoch)

        scheduler.step()  # 学习率调度器步进

        # 检查早停
        early_stopper(avg_val_loss, model, optimizer, current_epoch)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {current_epoch} for FDIP_1.")
            break

    # 训练结束后，加载最佳模型的状态
    print(
        f"FDIP_1 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    if os.path.exists(early_stopper.path):
        best_checkpoint = torch.load(early_stopper.path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        print(f"Warning: Best model checkpoint not found at {early_stopper.path}. Using last epoch's model.")

    if writer: writer.close()
    print("======================== FDIP_1 Training Finished ==========================================")
    return model


def train_fdip_2(model1, model2, optimizer, scheduler, train_loader, val_loader, epochs, early_stopper, start_epoch=0):
    """训练 FDIP_2 模型，支持从指定轮数开始训练"""
    print("\n====================== Starting FDIP_2 Training (Online Inference) =========================")
    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip2')) if LOG_ENABLED else None

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        model1.eval()  # model1在FDIP_2训练时不进行训练，只做前向推断
        model2.train()
        train_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"FDIP_2 Epoch {current_epoch}/{epochs}", leave=True)
        for data in epoch_pbar:
            acc = data[0].to(DEVICE, non_blocking=True).float()
            ori_6d = data[2].to(DEVICE, non_blocking=True).float()
            p_all = data[4].to(DEVICE, non_blocking=True).float()  # 24个关节的3D位置

            with torch.no_grad():  # FDIP_1 的推断不计算梯度
                input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                p_leaf_logits = model1(input1)
                # p_leaf_logits是5个叶节点的预测位置，需要拼接根节点（第0个节点）的0位置
                # 形状: [B, S, 5*3] -> [B, S, 6, 3]
                zeros = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)  # 根节点的3D位置是0
                p_leaf_pred = torch.cat(
                    [zeros, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 5, 3)], dim=2)

            # FDIP_2 的输入是acc, ori_6d和p_leaf_pred
            # 这里的p_leaf_pred形状是[B, S, 6, 3]，与acc和ori_6d的节点维度对齐
            # 拼接时需要展平
            x2 = torch.cat((acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], -1)), -1).view(
                acc.shape[0], acc.shape[1], -1)

            # target是所有24个关节的3D位置，根关节位置补0 (这是针对模型输出设计的，根关节位置为0)
            target = torch.cat([torch.zeros_like(p_all[:, :, 0:1, :]), p_all], dim=2).view(p_all.shape[0],
                                                                                           p_all.shape[1], -1)  # 24*3

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model2(x2)
                loss = torch.sqrt(criterion(logits, target))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss encountered at FDIP_2 Epoch {current_epoch}, skipping batch.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证阶段
        model2.eval()
        val_losses = []
        with torch.no_grad():
            for data_val in val_loader:
                acc_val, ori_val, p_all_val = [d.to(DEVICE, non_blocking=True).float() for d in
                                               (data_val[0], data_val[2], data_val[4])]
                input1_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                p_leaf_logits_val = model1(input1_val)
                zeros_val = torch.zeros(p_leaf_logits_val.shape[0], p_leaf_logits_val.shape[1], 3, device=DEVICE)
                p_leaf_pred_val = torch.cat(
                    [zeros_val, p_leaf_logits_val.view(p_leaf_logits_val.shape[0], p_leaf_logits_val.shape[1], 5, 3)],
                    dim=2)

                x2_val = torch.cat(
                    (acc_val, ori_val, p_leaf_pred_val.view(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], -1)),
                    -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                target_val = torch.cat([torch.zeros_like(p_all_val[:, :, 0:1, :]), p_all_val], dim=2).view(
                    p_all_val.shape[0],
                    p_all_val.shape[1], -1)
                logits_val = model2(x2_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val))
                val_losses.append(loss_val.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'FDIP_2 Epoch {current_epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}')

        if LOG_ENABLED and writer:
            writer.add_scalars('loss/fdip2', {'train': avg_train_loss, 'val': avg_val_loss}, current_epoch)
            writer.add_scalar('learning_rate/fdip2', current_lr, current_epoch)

        scheduler.step()
        early_stopper(avg_val_loss, model2, optimizer, current_epoch)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {current_epoch} for FDIP_2.")
            break

    print(
        f"FDIP_2 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    if os.path.exists(early_stopper.path):
        best_checkpoint = torch.load(early_stopper.path)
        model2.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        print(f"Warning: Best model checkpoint not found at {early_stopper.path}. Using last epoch's model.")

    if writer: writer.close()
    print("=========================== FDIP_2 Training Finished ==================================")
    return model2


def train_fdip_3(model1, model2, model3, optimizer, scheduler, train_loader, val_loader, epochs, early_stopper,
                 start_epoch=0):
    """训练 FDIP_3 模型，支持从指定轮数开始训练"""
    print("\n======================== Starting FDIP_3 Training (Online Inference)====================")
    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip3')) if LOG_ENABLED else None

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        model1.eval()  # 不训练
        model2.eval()  # 不训练
        model3.train()
        train_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"FDIP_3 Epoch {current_epoch}/{epochs}", leave=True)
        for data in epoch_pbar:
            acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                       (data[0], data[2], data[6])]  # pose_6d_gt是24个关节的6D姿态

            with torch.no_grad():  # FDIP_1 和 FDIP_2 的推断不计算梯度
                input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                p_leaf_logits = model1(input1)
                zeros = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
                p_leaf_pred = torch.cat(
                    [zeros, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 5, 3)], dim=2)

                input2 = torch.cat((acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], -1)),
                                   -1).view(acc.shape[0], acc.shape[1], -1)
                p_all_pos_flattened = model2(input2)  # FDIP_2 输出的所有24个关节的3D位置，展平

            input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)  # FDIP_3 的一部分输入

            # 目标是所有24个关节的6D姿态，展平
            target = pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], -1)  # 24*6 = 144

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                # FDIP_3 的输入是原始IMU数据和FDIP_2预测的所有关节位置
                logits = model3(input_base, p_all_pos_flattened)
                loss = torch.sqrt(criterion(logits, target))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss encountered at FDIP_3 Epoch {current_epoch}, skipping batch.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证阶段
        model3.eval()
        val_losses = []
        with torch.no_grad():
            for data_val in val_loader:
                acc_val, ori_val, pose_6d_gt_val = [d.to(DEVICE, non_blocking=True).float() for d in
                                                    (data_val[0], data_val[2], data_val[6])]
                input1_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                p_leaf_logits_val = model1(input1_val)
                zeros_val = torch.zeros(p_leaf_logits_val.shape[0], p_leaf_logits_val.shape[1], 3, device=DEVICE)
                p_leaf_pred_val = torch.cat(
                    [zeros_val, p_leaf_logits_val.view(p_leaf_logits_val.shape[0], p_leaf_logits_val.shape[1], 5, 3)],
                    dim=2)

                input2_val = torch.cat(
                    (acc_val, ori_val, p_leaf_pred_val.view(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], -1)),
                    -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                p_all_pos_flattened_val = model2(input2_val)
                input_base_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)

                target_val = pose_6d_gt_val.view(pose_6d_gt_val.shape[0], pose_6d_gt_val.shape[1], -1)
                logits_val = model3(input_base_val, p_all_pos_flattened_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val))
                val_losses.append(loss_val.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'FDIP_3 Epoch {current_epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}')

        if LOG_ENABLED and writer:
            writer.add_scalars('loss/fdip3', {'train': avg_train_loss, 'val': avg_val_loss}, current_epoch)
            writer.add_scalar('learning_rate/fdip3', current_lr, current_epoch)

        scheduler.step()
        early_stopper(avg_val_loss, model3, optimizer, current_epoch)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {current_epoch} for FDIP_3.")
            break

    print(
        f"FDIP_3 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    if os.path.exists(early_stopper.path):
        best_checkpoint = torch.load(early_stopper.path)
        model3.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        print(f"Warning: Best model checkpoint not found at {early_stopper.path}. Using last epoch's model.")

    if writer: writer.close()
    print("================================ FDIP_3 Training Finished =======================================")
    return model3


def evaluate_pipeline(model1, model2, model3, data_loader):
    print("\n============================ Evaluating Complete Pipeline ======================================")
    evaluator = PoseEvaluator()
    model1.eval(); model2.eval(); model3.eval()
    all_errs_list = []
    with torch.no_grad():
        for data_val in tqdm(data_loader, desc="Evaluating Pipeline"):
            acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                       (data_val[0], data_val[2], data_val[6])]
            input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            p_leaf_logits = model1(input1)
            zeros1 = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
            p_leaf_pred = torch.cat([zeros1, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 5, 3)], dim=2)
            input2 = torch.cat((acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], -1)), -1).view(acc.shape[0], acc.shape[1], -1)
            p_all_pos_flattened = model2(input2)
            input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            pose_pred = model3(input_base, p_all_pos_flattened)
            pose_gt = pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], -1)
            errs = evaluator.eval(pose_pred, pose_gt)
            all_errs_list.append(errs.cpu())

    if all_errs_list:
        all_errs = torch.cat(all_errs_list, dim=1)  # 形状 [5, total_frames]
        avg_errs = all_errs.mean(dim=1)
        print("Complete Pipeline Evaluation Results:")
        print(f"SIP Error (deg): {avg_errs[0].item():.4f}")
        print(f"Angular Error (deg): {avg_errs[1].item():.4f}")
        print(f"Positional Error (cm): {avg_errs[2].item():.4f}")
        print(f"Mesh Error (cm): {avg_errs[3].item():.4f}")
        print(f"Jitter Error (100m/s^3): {avg_errs[4].item():.4f}")

        # 添加小提琴图
        import seaborn as sns
        import matplotlib.pyplot as plt
        error_names = ["SIP Error (deg)", "Angular Error (deg)", "Positional Error (cm)", "Mesh Error (cm)", "Jitter Error (100m/s^3)"]
        for i, name in enumerate(error_names):
            plt.figure(figsize=(8, 6))  # 设置图表大小
            sns.violinplot(data=all_errs[i].numpy(), color='skyblue', inner='box')  # 绘制小提琴图，包含箱线图
            plt.title(f"{name} Distribution")
            plt.xlabel("Error Type")
            plt.ylabel("Error Value")
            plt.savefig(f"{name.replace(' ', '_')}_violin.png")
            plt.close()
        print("Error distribution plots saved as PNG files.")
    else:
        print("No evaluation results generated.")


def main():
    """主函数，运行完整的训练和评估流程，并支持从断点恢复。"""
    set_seed(SEED)
    print("==================== Starting Full Training Pipeline =====================")
    total_start_time = time.time()

    create_directories()
    train_loader, val_loader = load_data(train_percent=TRAIN_PERCENT)

    patience = 15
    max_epochs = 200  # 设置一个较高的上限，由早停来决定最佳轮数

    # --- 阶段 1: FDIP_1 ---
    print("\n--- Initializing Stage 1: FDIP_1 ---")
    model1 = FDIP_1(input_dim=6 * 9, output_dim=5 * 3).to(DEVICE)  # 6个IMU，每个9维(acc, ori_6d)。输出5个叶节点的3D位置。
    optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)   #自适应学习率优化算法（每个参数的学习率）
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max_epochs, eta_min=1e-6)   #优化，全局基准学习率（起点）

    checkpoint_path1 = os.path.join(CHECKPOINT_DIR, 'ggip1', 'best_model_fdip1.pth')  # FDIP_1 模型最佳检查点（
    early_stopper1 = EarlyStopping(patience=patience, path=checkpoint_path1, verbose=True)   # 早停
    start_epoch1 = 0

    if os.path.exists(checkpoint_path1):   # 恢复训练
        print(f"Found checkpoint for FDIP_1. Resuming training from: {checkpoint_path1}")
        checkpoint = torch.load(checkpoint_path1, map_location=DEVICE)  # 加载到当前设备
        model1.load_state_dict(checkpoint['model_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch1 = checkpoint['epoch']  # 从上次保存的epoch+1开始
        early_stopper1.val_loss_min = checkpoint['val_loss_min']
        early_stopper1.best_score = checkpoint['best_score']
        early_stopper1.counter = checkpoint.get('early_stopping_counter', 0)  # 兼容旧检查点
        # 调整调度器以匹配恢复的轮数
        for _ in range(start_epoch1):  # scheduler.step() 会在每次epoch结束时调用，所以需要步进start_epoch次
            scheduler1.step()
        print(f"Resuming from Epoch {start_epoch1 + 1}. Best validation loss so far: {early_stopper1.val_loss_min:.6f}")
    else:
        print("No checkpoint found for FDIP_1. Starting training from scratch.")

    model1 = train_fdip_1(
        model=model1,
        optimizer=optimizer1,
        scheduler=scheduler1,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        early_stopper=early_stopper1,
        start_epoch=start_epoch1
    )

    # --- 阶段 2: FDIP_2 ---
    print("\n--- Initializing Stage 2: FDIP_2 ---")
    model2 = FDIP_2(input_dim=6 * 12, output_dim=24 * 3).to(DEVICE)  # 6个IMU(acc, ori_6d) + 6个节点的3D位置。输出24个关节的3D位置。
    optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max_epochs, eta_min=1e-6)

    checkpoint_path2 = os.path.join(CHECKPOINT_DIR, 'ggip2', 'best_model_fdip2.pth')
    early_stopper2 = EarlyStopping(patience=patience, path=checkpoint_path2, verbose=True)
    start_epoch2 = 0

    if os.path.exists(checkpoint_path2):
        print(f"Found checkpoint for FDIP_2. Resuming training from: {checkpoint_path2}")
        checkpoint = torch.load(checkpoint_path2, map_location=DEVICE)
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch2 = checkpoint['epoch']
        early_stopper2.val_loss_min = checkpoint['val_loss_min']
        early_stopper2.best_score = checkpoint['best_score']
        early_stopper2.counter = checkpoint.get('early_stopping_counter', 0)
        for _ in range(start_epoch2):
            scheduler2.step()
        print(f"Resuming from Epoch {start_epoch2 + 1}. Best validation loss so far: {early_stopper2.val_loss_min:.6f}")
    else:
        print("No checkpoint found for FDIP_2. Starting training from scratch.")

    model2 = train_fdip_2(
        model1=model1,  # FDIP_1 作为预训练或固定模型输入
        model2=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        early_stopper=early_stopper2,
        start_epoch=start_epoch2
    )

    # --- 阶段 3: FDIP_3 ---
    print("\n--- Initializing Stage 3: FDIP_3 ---")
    model3 = FDIP_3(input_dim=6 * 9, output_dim=24 * 6, num_nodes=24).to(
        DEVICE)  # 6个IMU的9维输入。同时需要FDIP_2输出的24个关节的3D位置。输出24个关节的6D姿态。
    optimizer3 = optim.Adam(model3.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=max_epochs, eta_min=1e-6)

    checkpoint_path3 = os.path.join(CHECKPOINT_DIR, 'ggip3', 'best_model_fdip3.pth')
    early_stopper3 = EarlyStopping(patience=patience, path=checkpoint_path3, verbose=True)
    start_epoch3 = 0

    if os.path.exists(checkpoint_path3):
        print(f"Found checkpoint for FDIP_3. Resuming training from: {checkpoint_path3}")
        checkpoint = torch.load(checkpoint_path3, map_location=DEVICE)
        model3.load_state_dict(checkpoint['model_state_dict'])
        optimizer3.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch3 = checkpoint['epoch']
        early_stopper3.val_loss_min = checkpoint['val_loss_min']
        early_stopper3.best_score = checkpoint['best_score']
        early_stopper3.counter = checkpoint.get('early_stopping_counter', 0)
        for _ in range(start_epoch3):
            scheduler3.step()
        print(f"Resuming from Epoch {start_epoch3 + 1}. Best validation loss so far: {early_stopper3.val_loss_min:.6f}")
    else:
        print("No checkpoint found for FDIP_3. Starting training from scratch.")

    model3 = train_fdip_3(
        model1=model1,  # FDIP_1 作为预训练或固定模型输入
        model2=model2,  # FDIP_2 作为预训练或固定模型输入
        model3=model3,
        optimizer=optimizer3,
        scheduler=scheduler3,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        early_stopper=early_stopper3,
        start_epoch=start_epoch3
    )

    print("\nAll training stages complete!")
    total_end_time = time.time()
    print(f"Total training time: {(total_end_time - total_start_time) / 3600:.2f} hours")

    # --- 评估阶段 ---
    # 评估逻辑使用最终加载的最佳模型，无需更改
    evaluate_pipeline(model1, model2, model3, val_loader)

    print("\nTraining and evaluation finished successfully!")


if __name__ == '__main__':
    main()

