import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import seaborn as sns     # 基于matplotlib的高级可视化库，用于创建统计图表
import matplotlib.pyplot as plt   # 绘图库
from torch.cuda.amp import autocast, GradScaler     #float16
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter          # 训练指标写入TensorBoard日志
from tqdm import tqdm
import json
import pickle    # 序列化
import pandas as pd   # 表格数据分析
from datetime import datetime
from articulate.math import r6d_to_rotation_matrix
from data.dataset_posReg import ImuDataset
from model.net_zd import FDIP_1, FDIP_2, FDIP_3
from evaluator import PoseEvaluator, PerFramePoseEvaluator   # 整体、逐帧姿态评估
import gc
import argparse

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置使用的GPU
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3                       # 正则化参数，控制权重衰减
BATCH_SIZE = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOG_ENABLED = True
TRAIN_PERCENT = 0.9
BATCH_SIZE_VAL = 32
SEED = 42

PATIENCE = 20
MAX_EPOCHS = 150
DELTA = 0

# --- Paths ---
# 请确保这些路径在您的环境中是正确的
TRAIN_DATA_FOLDERS = [
    os.path.join("D:\\", "Dataset", "TotalCapture_Real_60FPS", "KaPt", "split_actions"),
    os.path.join("D:\\", "Dataset", "DIPIMUandOthers", "DIP_6", "Detail"),
    os.path.join("D:\\", "Dataset", "AMASS", "DanceDB", "pt"),
    os.path.join("D:\\", "Dataset", "AMASS", "HumanEva", "pt"),

]
VAL_DATA_FOLDERS = [
    os.path.join("D:\\", "Dataset", "SingleOne",  "pt"),
]

TIMESTAMP = None
CHECKPOINT_DIR = None
LOG_DIR = "log"
LOG_RUN_DIR = None


class DataNormalizer:
    """数据标准化器，确保训练集和验证集使用相同的归一化参数"""
    def __init__(self):
        self.stats = {}
        self.fitted = False

    def fit(self, data_loader, device=DEVICE):
        """在训练集上拟合统计量"""
        print("Computing normalization statistics from training data...")

        # 初始化累积变量
        acc_sum = torch.zeros(6, 3, device=device)  # 6个传感器，3轴加速度
        ori_sum = torch.zeros(6, 6, device=device)  # 6个传感器，6D方向
        acc_sq_sum = torch.zeros(6, 3, device=device)
        ori_sq_sum = torch.zeros(6, 6, device=device)

        total_samples = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(data_loader, desc="Computing stats")):
                acc = data[0].to(device, non_blocking=True).float()  # [B, S, 6, 3]
                ori_6d = data[2].to(device, non_blocking=True).float()  # [B, S, 6, 6]

                batch_size, seq_len = acc.shape[:2]

                # 重塑为 [B*S, 6, 3] 和 [B*S, 6, 6]
                acc_flat = acc.view(-1, 6, 3)
                ori_flat = ori_6d.view(-1, 6, 6)

                # 累积统计量
                acc_sum += acc_flat.sum(dim=0)
                ori_sum += ori_flat.sum(dim=0)
                acc_sq_sum += (acc_flat ** 2).sum(dim=0)
                ori_sq_sum += (ori_flat ** 2).sum(dim=0)

                total_samples += batch_size * seq_len

                # 为了节省内存，只使用部分数据计算统计量
                if batch_idx >= 100:  # 使用前100个batch计算统计量
                    break

        # 计算均值和标准差
        self.stats['acc_mean'] = acc_sum / total_samples
        self.stats['ori_mean'] = ori_sum / total_samples
        self.stats['acc_std'] = torch.sqrt(acc_sq_sum / total_samples - self.stats['acc_mean'] ** 2)
        self.stats['ori_std'] = torch.sqrt(ori_sq_sum / total_samples - self.stats['ori_mean'] ** 2)

        # 防止标准差为0
        self.stats['acc_std'] = torch.clamp(self.stats['acc_std'], min=1e-6)
        self.stats['ori_std'] = torch.clamp(self.stats['ori_std'], min=1e-6)

        self.fitted = True

        # 打印统计信息
        print("Normalization statistics computed:")
        print(f"  Acc mean: {self.stats['acc_mean'].mean().item():.6f}")
        print(f"  Acc std: {self.stats['acc_std'].mean().item():.6f}")
        print(f"  Ori mean: {self.stats['ori_mean'].mean().item():.6f}")
        print(f"  Ori std: {self.stats['ori_std'].mean().item():.6f}")

    def transform_batch(self, acc, ori_6d):
        """对单个batch进行标准化"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        # 标准化
        acc_norm = (acc - self.stats['acc_mean'].unsqueeze(0).unsqueeze(0)) / self.stats['acc_std'].unsqueeze(
            0).unsqueeze(0)
        ori_norm = (ori_6d - self.stats['ori_mean'].unsqueeze(0).unsqueeze(0)) / self.stats['ori_std'].unsqueeze(
            0).unsqueeze(0)

        return acc_norm, ori_norm

    def save_stats(self, path):
        """保存归一化统计量"""
        if self.fitted:
            torch.save(self.stats, path)
            print(f"Normalization stats saved to: {path}")

    def load_stats(self, path):
        """加载归一化统计量"""
        self.stats = torch.load(path)
        self.fitted = True
        print(f"Normalization stats loaded from: {path}")


def create_directories():
    """创建必要的目录，log内部带时间戳子文件夹。"""
    dirs = [
        os.path.join(CHECKPOINT_DIR, "ggip1"),
        os.path.join(CHECKPOINT_DIR, "ggip2"),
        os.path.join(CHECKPOINT_DIR, "ggip3"),
        LOG_DIR,
        LOG_RUN_DIR,
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Directories created with timestamp {TIMESTAMP}:")
    for dir_path in dirs:
        print(f"  - {dir_path}")

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


def setup_directories_and_paths(args):
    """根据命令行参数设置全局路径变量"""
    global TIMESTAMP, CHECKPOINT_DIR, LOG_RUN_DIR

    # 确定时间戳和检查点目录
    if args.resume:
        TIMESTAMP = args.resume
        CHECKPOINT_DIR = os.path.join("GGIP", f"checkpoints_{TIMESTAMP}")
        print(f"Resuming training from timestamp: {TIMESTAMP}")
    elif args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
        TIMESTAMP = os.path.basename(CHECKPOINT_DIR).replace("checkpoints_", "")
        print(f"Using checkpoint directory: {CHECKPOINT_DIR}")
    else:
        TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        CHECKPOINT_DIR = os.path.join("GGIP", f"checkpoints_{TIMESTAMP}")
        print(f"Starting new training with timestamp: {TIMESTAMP}")

    # 设置日志目录
    LOG_RUN_DIR = os.path.join(LOG_DIR, TIMESTAMP)

    # 验证检查点目录是否存在（仅对续训情况）
    if not os.path.exists(CHECKPOINT_DIR) and (args.resume or args.checkpoint_dir):
        print(f"Error: Checkpoint directory {CHECKPOINT_DIR} does not exist!")
        sys.exit(1)

    print(f"Global paths set:")
    print(f"  - TIMESTAMP: {TIMESTAMP}")
    print(f"  - CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"  - LOG_RUN_DIR: {LOG_RUN_DIR}")

def clear_memory():
    """清理GPU和CPU内存"""
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

def cleanup_training_objects(*objects):
    """清理训练相关对象"""
    for obj in objects:
        if obj is not None:
            del obj
    clear_memory()


class EarlyStopping:
    """
    如果验证损失在给定的耐心期后没有改善，则提前停止训练（仍会进行下一阶段训练）。
    MODIFIED: 现在可以保存和加载优化器状态和轮数。
    """

    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience    # 允许验证损失不改善的轮数上限
        self.verbose = verbose      # 控制是否打印详细信息
        self.counter = 0            # 用于记录连续没有改善的轮数
        self.best_score = None      # 记录最佳分数（-val_loss）【最佳=最大】
        self.early_stop = False
        self.val_loss_min = np.Inf  # 记录最低验证损失
        self.delta = delta          # 定义"显著改善"的最小阈值
        self.path = path
        self.best_epoch = 0         # 记录达到最低验证损失的轮数

    def __call__(self, val_loss, model, optimizer, epoch):

        if not np.isfinite(val_loss):
            if self.verbose:
                print(f"Warning: Validation loss is {val_loss} at epoch {epoch}, skipping EarlyStopping.")
            return

        score = -val_loss            # 将损失转换为分数，越大越好
        if self.best_score is None:  # 第一次调用
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:  # 分数没有改善
            self.counter += 1
            if self.verbose:                        # 打印信息
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:       # 达到耐心上限
                self.early_stop = True
        else:                                       # 分数有所改善
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0                        # 重置计数器

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """当验证损失减少时保存模型、优化器和轮数。"""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving checkpoint to {self.path}...')

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,                                   # 当前轮数
            'model_state_dict': model.state_dict(),           # 模型参数
            'optimizer_state_dict': optimizer.state_dict(),   # 优化器状态
            'val_loss_min': val_loss,                         # 当前最佳验证损失
            'best_score': self.best_score,                    # 当前最佳分数
            'early_stopping_counter': self.counter            # 早停计数器状态
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch


def load_data_unified_split(train_percent=0.8, val_percent=0.2, seed=None):
    """
    统一加载所有数据集，然后随机划分为训练集和验证集
    这样可以确保训练集和验证集来自相同的数据分布

    参数:
        train_percent: 训练集比例
        val_percent: 验证集比例
        seed: 随机种子，用于确保可重现的划分
    """
    print("Loading unified dataset with consistent split method...")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    try:
        all_data_folders = TRAIN_DATA_FOLDERS + VAL_DATA_FOLDERS
        print(f"Combining datasets from {len(all_data_folders)} folders:")
        for folder in all_data_folders:
            print(f"  - {folder}")

        unified_dataset = ImuDataset(all_data_folders)
        total_size = len(unified_dataset)
        print(f"Total unified dataset size: {total_size} samples")

        train_size = int(total_size * train_percent)
        val_size = total_size - train_size  # 剩余的都给验证集

        print(f"Dataset split:")
        print(f"  - Training: {train_size} samples ({train_size / total_size * 100:.1f}%)")
        print(f"  - Validation: {val_size} samples ({val_size / total_size * 100:.1f}%)")

        from torch.utils.data import random_split

        train_dataset, val_dataset = random_split(
            unified_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed) if seed else None
        )

    except Exception as e:
        print(f"Error loading unified dataset: {e}")
        sys.exit(1)

    num_workers = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    print(f"Data loaders created successfully!")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def check_data_distribution(train_loader, val_loader, num_samples=5):
    """
    检查训练集和验证集的数据分布一致性

    参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_samples: 用于统计的样本批次数
    """
    print("\n=== Data Distribution Analysis ===")

    def compute_stats(data_loader, name, max_batches=num_samples):
        """计算数据集的基本统计信息"""
        stats = {
            'acc_mean': [],
            'acc_std': [],
            'ori_mean': [],
            'ori_std': [],
            'pos_mean': [],
            'pos_std': []
        }

        count = 0
        with torch.no_grad():
            for data in data_loader:
                if count >= max_batches:
                    break

                acc = data[0].float()
                ori = data[2].float()
                pos = data[3].float()

                stats['acc_mean'].append(acc.mean().item())
                stats['acc_std'].append(acc.std().item())
                stats['ori_mean'].append(ori.mean().item())
                stats['ori_std'].append(ori.std().item())
                stats['pos_mean'].append(pos.mean().item())
                stats['pos_std'].append(pos.std().item())

                count += 1

        # 计算平均值
        for key in stats:
            stats[key] = np.mean(stats[key])

        return stats

    # 计算训练集和验证集统计
    print("Computing training set statistics...")
    train_stats = compute_stats(train_loader, "Train")

    print("Computing validation set statistics...")
    val_stats = compute_stats(val_loader, "Validation")

    # 打印对比结果
    print(f"\nDistribution Comparison (based on {num_samples} batches):")
    print(f"{'Metric':<15} {'Train':<12} {'Validation':<12} {'Difference':<12}")
    print("-" * 55)

    for key in train_stats:
        train_val = train_stats[key]
        val_val = val_stats[key]
        diff = abs(train_val - val_val)
        print(f"{key:<15} {train_val:<12.6f} {val_val:<12.6f} {diff:<12.6f}")

    # 计算总体相似度得分
    total_diff = sum([abs(train_stats[key] - val_stats[key]) for key in train_stats])
    print(f"\nTotal Difference Score: {total_diff:.6f} (lower is better)")

    if total_diff < 0.1:
        print("✓ Data distributions appear consistent!")
    elif total_diff < 0.5:
        print("⚠ Data distributions have minor differences")
    else:
        print("❌ Data distributions have significant differences")

    return train_stats, val_stats


def load_data_separate():
    """
    分别加载训练集和验证集，避免数据泄漏
    """
    print("Loading separate train and validation datasets...")

    try:
        # 加载训练数据集
        print("Loading training dataset...")
        train_dataset = ImuDataset(TRAIN_DATA_FOLDERS)
        print(f"Training dataset loaded: {len(train_dataset)} samples")

        # 加载验证数据集
        print("Loading validation dataset...")
        val_dataset = ImuDataset(VAL_DATA_FOLDERS)
        print(f"Validation dataset loaded: {len(val_dataset)} samples")

    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure your dataset paths and ImuDataset class are correct.")
        sys.exit(1)

    # 数据加载器设置
    num_workers = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    print(f"Data loaders created successfully!")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def load_data_legacy(train_percent=0.9):
    """
    原始的数据加载方式（作为备选方案）
    从同一数据集中按比例划分训练和验证集
    """
    print("Loading dataset with legacy split method...")
    try:
        custom_dataset = ImuDataset(TRAIN_DATA_FOLDERS)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    total_size = len(custom_dataset)
    train_size = int(total_size * train_percent)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))

    train_dataset = Subset(custom_dataset, train_indices)
    val_dataset = Subset(custom_dataset, val_indices)

    num_workers = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False,
        pin_memory=True, num_workers=num_workers
    )
    print(f"Legacy dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    return train_loader, val_loader

def train_fdip_1(model, optimizer, scheduler, train_loader, val_loader, epochs, early_stopper, start_epoch=0):
    """训练 FDIP_1 模型，支持从指定轮数开始训练"""
    print("\n=============================== Starting FDIP_1 Training =============================")
    criterion = nn.MSELoss()
    scaler = GradScaler()
    # 创建SummaryWriter实例
    writer = SummaryWriter(os.path.join(LOG_RUN_DIR, 'ggip1')) if LOG_ENABLED else None

    # 从 start_epoch 开始循环，end_epoch 为 epochs - 1
    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1  # 用于日志显示，从1开始
        model.train()
        train_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"FDIP_1 Epoch {current_epoch}/{epochs}", leave=True)
        # 每个批次 out_acc, out_ori, out_rot_6d, out_leaf_pos, out_all_pos,  out_pose, out_pose_6d, out_shape
        for data in epoch_pbar:
            acc = data[0].to(DEVICE, non_blocking=True).float()
            ori_6d = data[2].to(DEVICE, non_blocking=True).float()
            p_leaf = data[3].to(DEVICE, non_blocking=True).float()

            x = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            target = p_leaf.view(-1, p_leaf.shape[1], 15)                       # 5个叶节点，每个3D位置

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = torch.sqrt(criterion(logits, target))                    # 使用RMSE

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

                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    print(
                        f"Warning: NaN/Inf loss encountered at FDIP_1 Epoch {current_epoch}, skipping validation batch.")
                    continue

                val_losses.append(loss_val.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0      # 处理空列表情况
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0            # 处理空列表情况
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'FDIP_1 Epoch {current_epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}')

        if LOG_ENABLED and writer:
            writer.add_scalars('loss/fdip1', {'train': avg_train_loss, 'val': avg_val_loss}, current_epoch)
            writer.add_scalar('learning_rate/fdip1', current_lr, current_epoch)

        scheduler.step()                                                     # 学习率调度器步进

        # 检查早停
        early_stopper(avg_val_loss, model, optimizer, current_epoch)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {current_epoch} for FDIP_1.")
            break

        # 🔥 每个epoch结束后清理内存
        torch.cuda.empty_cache()

    # 训练结束后，加载最佳模型的状态
    print(
        f"FDIP_1 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    if os.path.exists(early_stopper.path):
        best_checkpoint = torch.load(early_stopper.path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        print(f"Warning: Best model checkpoint not found at {early_stopper.path}. Using last epoch's model.")

    if writer:
        writer.close()
        del writer  # 🔥 清理writer

    # 🔥 清理训练过程中的临时变量
    del criterion, scaler
    torch.cuda.empty_cache()

    print("======================== FDIP_1 Training Finished ==========================================")
    return model


def train_fdip_2(model1, model2, optimizer, scheduler, train_loader, val_loader, epochs, early_stopper, start_epoch=0):
    """训练 FDIP_2 模型，支持从指定轮数开始训练"""
    print("\n====================== Starting FDIP_2 Training (Online Inference) =========================")
    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_RUN_DIR, 'ggip2')) if LOG_ENABLED else None

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
                    [zeros, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], -1)], dim=2)

            # FDIP_2 的输入是acc, ori_6d和p_leaf_pred
            # 这里的p_leaf_pred形状是[B, S, 6, 3]，与acc和ori_6d的节点维度对齐
            # 拼接时需要展平
            x2 = torch.cat([acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 6, 3)],dim=-1).view(
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
                zeros_val = torch.zeros(p_leaf_logits_val.shape[0], p_leaf_logits_val.shape[1],1, 3, device=DEVICE)
                p_leaf_pred_val = torch.cat(
                    [zeros_val, p_leaf_logits_val.view(p_leaf_logits_val.shape[0], p_leaf_logits_val.shape[1], 5, 3)],
                    dim=2)

                x2_val = torch.cat(
                    (acc_val, ori_val, p_leaf_pred_val),
                    -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                target_val = torch.cat([torch.zeros_like(p_all_val[:, :, 0:1, :]), p_all_val], dim=2).view(
                    p_all_val.shape[0],
                    p_all_val.shape[1], -1)
                logits_val = model2(x2_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val))

                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    print(
                        f"Warning: NaN/Inf loss encountered at FDIP_2 Epoch {current_epoch}, skipping validation batch.")
                    continue

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

        # 🔥 每个epoch结束后清理内存
        torch.cuda.empty_cache()

    print(
        f"FDIP_2 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")


    if os.path.exists(early_stopper.path):
        best_checkpoint = torch.load(early_stopper.path)
        model2.load_state_dict(best_checkpoint['model_state_dict'])
        del best_checkpoint  # 🔥 清理checkpoint
    else:
        print(f"Warning: Best model checkpoint not found at {early_stopper.path}. Using last epoch's model.")

    if writer:
        writer.close()
        del writer

    # 🔥 清理训练过程中的临时变量
    del criterion, scaler
    torch.cuda.empty_cache()

    print("=========================== FDIP_2 Training Finished ==================================")
    return model2


def train_fdip_3(model1, model2, model3, optimizer, scheduler, train_loader, val_loader, epochs, early_stopper,
                 start_epoch=0):
    """训练 FDIP_3 模型，支持从指定轮数开始训练"""
    print("\n======================== Starting FDIP_3 Training (Online Inference)====================")
    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_RUN_DIR, 'ggip3')) if LOG_ENABLED else None

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
                zeros = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 1, 3, device=DEVICE)
                p_leaf_pred = torch.cat(
                    [zeros, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 5, 3)], dim=2)

                input2 = torch.cat((acc, ori_6d, p_leaf_pred),-1).view(acc.shape[0], acc.shape[1], -1)
                p_all_pos_flattened = model2(input2)  # FDIP_2 输出的所有24个关节的3D位置，展平

            input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)  # FDIP_3 的一部分输入

            # 目标是所有24个关节的6D姿态，展平
            target = pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], -1)  # 24*6 = 144

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                # FDIP_3 的输入是原始IMU数据和FDIP_2预测的所有关节位置
                pose_pred_flat = model3(input_base, p_all_pos_flattened)

                # 重塑为便于计算损失的格式
                batch_size, seq_len = pose_pred_flat.shape[:2]
                pose_pred = pose_pred_flat.view(batch_size, seq_len, 24, 6)
                pose_gt = pose_6d_gt.view(batch_size, seq_len, 24, 6)

                # 使用改进的损失函数
                loss = rotation_matrix_loss(pose_pred, pose_gt)

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
                zeros_val = torch.zeros(p_leaf_logits_val.shape[0], p_leaf_logits_val.shape[1],3, device=DEVICE)
                p_leaf_pred_val = torch.cat(
                    [zeros_val, p_leaf_logits_val],
                    dim=2)

                input2_val = torch.cat(
                    (acc_val, ori_val, p_leaf_pred_val.view(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], 6, 3)),
                    -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                p_all_pos_flattened_val = model2(input2_val)
                input_base_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)

                target_val = pose_6d_gt_val.view(pose_6d_gt_val.shape[0], pose_6d_gt_val.shape[1], -1)
                logits_val = model3(input_base_val, p_all_pos_flattened_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val))

                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    print(
                        f"Warning: NaN/Inf loss encountered at FDIP_1 Epoch {current_epoch}, skipping validation batch.")
                    continue

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

        # 🔥 每个epoch结束后清理内存
        torch.cuda.empty_cache()

    print(
        f"FDIP_3 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    if os.path.exists(early_stopper.path):
        best_checkpoint = torch.load(early_stopper.path)
        model3.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        print(f"Warning: Best model checkpoint not found at {early_stopper.path}. Using last epoch's model.")

    if writer:
        writer.close()
        del writer  # 🔥 清理writer

    # 🔥 清理训练过程中的临时变量
    del criterion, scaler
    torch.cuda.empty_cache()

    print("================================ FDIP_3 Training Finished =======================================")
    return model3


def clean_filename(filename):
    """清理文件名，移除Windows中不允许的字符"""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def evaluate_pipeline(model1, model2, model3, data_loader):
    print("\n============================ Evaluating Complete Pipeline ======================================")

    # 🔥 评估前清理内存
    clear_memory()

    eval_results_dir = os.path.join("GGIP", f"evaluate_pipeline_{TIMESTAMP}")
    eval_plots_dir = os.path.join(eval_results_dir, "plots")
    eval_data_dir = os.path.join(eval_results_dir, "data")

    # 创建目录
    os.makedirs(eval_results_dir, exist_ok=True)
    os.makedirs(eval_plots_dir, exist_ok=True)
    os.makedirs(eval_data_dir, exist_ok=True)

    try:
        evaluator = PerFramePoseEvaluator()
        model1.eval()
        model2.eval()
        model3.eval()

        all_errors = {
            "pos_err": [],
            "mesh_err": [],
            "angle_err": [],
            "jitter_err": []
        }

        print("Running model evaluation...")
        with torch.no_grad():
            for data_val in tqdm(data_loader, desc="Evaluating Pipeline"):
                try:
                    # --- 模型前向传播 ---
                    acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                               (data_val[0], data_val[2], data_val[6])]

                    input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                    p_leaf_logits = model1(input1)

                    zeros1 = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
                    p_leaf_pred = torch.cat([zeros1, p_leaf_logits], dim=2)

                    input2 = torch.cat(
                        (acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 6, 3)),
                        -1).view(acc.shape[0], acc.shape[1], -1)
                    p_all_pos_flattened = model2(input2)
                    input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                    pose_pred_flat = model3(input_base, p_all_pos_flattened)

                    batch_size, seq_len = pose_pred_flat.shape[:2]
                    pose_pred = pose_pred_flat.view(batch_size, seq_len, 24, 6)

                    errs_dict = evaluator.eval(pose_pred, pose_6d_gt)

                    for key in all_errors.keys():
                        if errs_dict[key].numel() > 0:
                            all_errors[key].append(errs_dict[key].flatten().cpu())

                except Exception as e:
                    print(f"Warning: Error processing batch in evaluation: {e}")
                    continue

        # 🔥 评估前清理内存
        clear_memory()

        # --- 汇总结果 ---
        if all_errors["mesh_err"]:
            print("Processing evaluation results...")

            # 拼接所有误差数据
            final_errors = {key: torch.cat(val, dim=0) for key, val in all_errors.items() if val}
            avg_errors = {key: val.mean().item() for key, val in final_errors.items()}

            # 打印结果
            print("\nComplete Pipeline Evaluation Results (Mean):")
            print(f"  - Positional Error (cm):      {avg_errors.get('pos_err', 'N/A'):.4f}")
            print(f"  - Mesh Error (cm):            {avg_errors.get('mesh_err', 'N/A'):.4f}")
            print(f"  - Angular Error (deg):        {avg_errors.get('angle_err', 'N/A'):.4f}")
            print(f"  - Jitter Error (cm/s²):       {avg_errors.get('jitter_err', 'N/A'):.4f}")

            # --- 保存数据到文件 ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            try:
                # 1. 保存原始误差数据 (pickle格式，保持完整的tensor数据)
                raw_data_path = os.path.join(eval_data_dir, f"raw_errors_{timestamp}.pkl")
                with open(raw_data_path, 'wb') as f:
                    pickle.dump(final_errors, f)
                print(f"Raw error data saved to: {raw_data_path}")

                # 2. 保存统计结果 (JSON格式，易于阅读)
                stats_data = {
                    "timestamp": timestamp,
                    "evaluation_results": {
                        "mean_errors": avg_errors,
                        "sample_counts": {key: len(val) for key, val in final_errors.items()},
                        "std_errors": {key: val.std().item() for key, val in final_errors.items()},
                        "min_errors": {key: val.min().item() for key, val in final_errors.items()},
                        "max_errors": {key: val.max().item() for key, val in final_errors.items()}
                    },
                    "units": {  # 添加单位信息
                        "pos_err": "cm",
                        "mesh_err": "cm",
                        "angle_err": "degrees",
                        "jitter_err": "cm/s²"
                    }
                }

                stats_path = os.path.join(eval_data_dir, f"evaluation_stats_{timestamp}.json")
                with open(stats_path, 'w') as f:
                    json.dump(stats_data, f, indent=2)
                print(f"Statistics saved to: {stats_path}")

                # 3. 保存为CSV格式 (方便Excel打开)
                csv_data = []
                for key, values in final_errors.items():
                    for value in values.numpy():
                        csv_data.append({
                            'metric': key,
                            'value': value,
                            'timestamp': timestamp
                        })

                if csv_data:
                    df = pd.DataFrame(csv_data)
                    csv_path = os.path.join(eval_data_dir, f"evaluation_data_{timestamp}.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"CSV data saved to: {csv_path}")

            except Exception as e:
                print(f"Warning: Error saving data files: {e}")

            # --- 生成并保存图表 ---
            print("\nSaving error distribution plots...")

            # 修正后的单位映射
            error_names_map = {
                "pos_err": "Positional Error (cm)",
                "mesh_err": "Mesh Error (cm)",
                "angle_err": "Angular Error (deg)",
                "jitter_err": "Jitter Error (cm/s²)"  # 修正单位
            }

            # 对应的y轴标签
            ylabel_map = {
                "pos_err": "Error (cm)",
                "mesh_err": "Error (cm)",
                "angle_err": "Error (degrees)",
                "jitter_err": "Error (cm/s²)"  # 修正y轴标签
            }

            for key, full_name in error_names_map.items():
                if key in final_errors:
                    try:
                        plt.figure(figsize=(8, 6))
                        sns.violinplot(data=final_errors[key].numpy(), color='skyblue', inner='box')

                        # 使用正确的标题和y轴标签
                        plt.title(f"{full_name} Distribution", fontsize=14, fontweight='bold')
                        plt.ylabel(ylabel_map[key], fontsize=12)  # 使用具体的单位标签
                        plt.xlabel("Distribution", fontsize=12)

                        # 添加统计信息到图上
                        mean_val = final_errors[key].mean().item()
                        std_val = final_errors[key].std().item()
                        plt.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}',
                                 transform=plt.gca().transAxes,
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                        # 清理文件名（移除特殊字符）
                        clean_name = clean_filename(
                            full_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_'))
                        filename = f"{clean_name}_violin_{timestamp}.png"
                        filepath = os.path.join(eval_plots_dir, filename)

                        plt.savefig(filepath, bbox_inches='tight', dpi=300)
                        plt.close()
                        print(f"  - Saved: {filepath}")

                    except Exception as e:
                        print(f"Warning: Error saving plot for {key}: {e}")
                        plt.close()
                        continue

            # --- 保存汇总报告 ---
            try:
                report_path = os.path.join(eval_results_dir, f"evaluation_report_{timestamp}.txt")
                with open(report_path, 'w') as f:
                    f.write("=== GGIP Pipeline Evaluation Report ===\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Total samples evaluated: {len(final_errors.get('mesh_err', []))}\n\n")

                    f.write("Mean Errors:\n")
                    unit_labels = {"pos_err": "cm", "mesh_err": "cm", "angle_err": "deg", "jitter_err": "cm/s²"}
                    for key, value in avg_errors.items():
                        unit = unit_labels.get(key, "")
                        f.write(f"  - {key}: {value:.4f} {unit}\n")

                    f.write("\nStandard Deviations:\n")
                    for key, val in final_errors.items():
                        unit = unit_labels.get(key, "")
                        f.write(f"  - {key}: {val.std().item():.4f} {unit}\n")

                    f.write("\nData Files Generated:\n")
                    f.write(f"  - Raw data: raw_errors_{timestamp}.pkl\n")
                    f.write(f"  - Statistics: evaluation_stats_{timestamp}.json\n")
                    f.write(f"  - CSV data: evaluation_data_{timestamp}.csv\n")
                    f.write(f"  - Plots: Located in plots/ subdirectory\n")

                print(f"Evaluation report saved to: {report_path}")

            except Exception as e:
                print(f"Warning: Error saving evaluation report: {e}")

        else:
            print("No evaluation results were generated.")

            # 即使没有结果也保存一个空报告
            try:
                empty_report_path = os.path.join(eval_results_dir,
                                                 f"evaluation_report_empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(empty_report_path, 'w') as f:
                    f.write("=== GGIP Pipeline Evaluation Report ===\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                    f.write("Status: No evaluation results were generated.\n")
                    f.write("Possible reasons: Empty dataset, evaluation errors, or model issues.\n")
                print(f"Empty evaluation report saved to: {empty_report_path}")
            except Exception as e:
                print(f"Warning: Error saving empty report: {e}")

    except Exception as e:
        print(f"Critical error in evaluation pipeline: {e}")
        print("Continuing with main program execution...")

        # 保存错误报告
        try:
            error_report_path = os.path.join(eval_results_dir,
                                             f"evaluation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(error_report_path, 'w') as f:
                f.write("=== GGIP Pipeline Evaluation Error Report ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Error Type: {type(e).__name__}\n")
            print(f"Error report saved to: {error_report_path}")
        except:
            print("Could not save error report")

    print(f"\nEvaluation completed. Results saved in: {eval_results_dir}")


def rotation_matrix_loss(pred_6d, target_6d):
    """
    将6D表示转换为旋转矩阵后计算Frobenius范数损失

    参数:
        pred_6d: 预测的6D旋转表示，形状为[B, S, J, 6]
        target_6d: 目标6D旋转表示，形状为[B, S, J, 6]
    返回:
        旋转矩阵空间中的损失
    """
    # 展平以便批量处理
    batch_size, seq_len, joints, _ = pred_6d.shape
    pred_6d_flat = pred_6d.reshape(-1, 6)
    target_6d_flat = target_6d.reshape(-1, 6)

    # 使用您提供的函数转换为旋转矩阵
    pred_rotmat = r6d_to_rotation_matrix(pred_6d_flat)  # 输出形状 [B*S*J, 3, 3]
    target_rotmat = r6d_to_rotation_matrix(target_6d_flat)

    # 计算Frobenius范数 (矩阵元素间的欧几里德距离)
    loss = torch.mean(torch.norm(pred_rotmat - target_rotmat, dim=(-2, -1)))

    return loss

def parse_args():
    parser = argparse.ArgumentParser(description='FDIP Training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint directory (e.g., 20250804_143022)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Specific checkpoint directory path')
    return parser.parse_args()
# python train.py --resume 20250804_143022
# python train.py --checkpoint_dir GGIP/checkpoints_20250804_143022


def main():
    """主函数，运行完整的训练和评估流程，并支持从断点恢复。"""
    set_seed(SEED)
    print("==================== Starting Full Training Pipeline =====================")

    args = parse_args()
    setup_directories_and_paths(args)
    create_directories()
    try:
        # train_loader, val_loader = load_data_separate()
        train_loader, val_loader = load_data_unified_split(
            train_percent=0.8,
            val_percent=0.2,
            seed=SEED
        )
        print("✓ Using separate train/validation datasets - No data leakage risk!")

        print("✓ Using unified dataset with consistent split!")
        # 检查数据分布一致性
        check_data_distribution(train_loader, val_loader)

    except Exception as e:
        print(f"❌ Failed to load separate datasets: {e}")
        print("⚠️  Falling back to legacy split method (may have data leakage risk)")
        train_loader, val_loader = load_data_legacy(train_percent=0.9)

    patience = PATIENCE
    max_epochs = MAX_EPOCHS

    total_start_time = time.time()

    # --- 阶段 1: FDIP_1 ---
    print("\n--- Initializing Stage 1: FDIP_1 ---")
    model1 = FDIP_1(input_dim=6 * 9, output_dim=5 * 3).to(DEVICE)
    checkpoint_path1 = os.path.join(CHECKPOINT_DIR, 'ggip1', 'best_model_fdip1.pth')
    # 定义阶段1的完成标记文件路径
    completion_marker1 = os.path.join(CHECKPOINT_DIR, 'ggip1', 'fdip1_completed.marker')

    # 检查阶段1是否已经完成
    if os.path.exists(completion_marker1):
        print("Stage 1 (FDIP_1) already completed. Loading best model and skipping training.")
        if os.path.exists(checkpoint_path1):
            checkpoint = torch.load(checkpoint_path1, map_location=DEVICE)
            model1.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded best model for FDIP_1 from {checkpoint_path1}")
            del checkpoint  # 清理checkpoint
        else:
            print(f"Error: Completion marker found, but checkpoint file {checkpoint_path1} is missing!")
            print("Please resolve this inconsistency or remove the marker file to re-train.")
            sys.exit(1)  # 终止程序，因为状态不一致
    else:
        # 优化器决定如何根据梯度更新参数、调度器决定何时调整学习率大小
        optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max_epochs, eta_min=1e-6)
        early_stopper1 = EarlyStopping(patience=patience, path=checkpoint_path1, verbose=True)
        start_epoch1 = 0

        if os.path.exists(checkpoint_path1):
            print(f"Found checkpoint for FDIP_1. Resuming training from: {checkpoint_path1}")
            checkpoint = torch.load(checkpoint_path1, map_location=DEVICE)
            model1.load_state_dict(checkpoint['model_state_dict'])
            optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch1 = checkpoint['epoch']
            early_stopper1.val_loss_min = checkpoint['val_loss_min']
            early_stopper1.best_score = checkpoint['best_score']
            early_stopper1.counter = checkpoint.get('early_stopping_counter', 0)
            for _ in range(start_epoch1):
                scheduler1.step()
            print(
                f"Resuming from Epoch {start_epoch1 + 1}. Best validation loss so far: {early_stopper1.val_loss_min:.6f}")
            del checkpoint  # 清理checkpoint
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

        # 训练成功结束后，创建完成标记文件
        with open(completion_marker1, 'w') as f:
            f.write(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Best model saved at epoch {early_stopper1.best_epoch} with val_loss {early_stopper1.val_loss_min:.6f}\n")
        print(f"Stage 1 (FDIP_1) marked as completed.")

        # 🔥 清理Stage 1的训练对象，释放内存
        cleanup_training_objects(optimizer1, scheduler1, early_stopper1)
        print("Stage 1 training objects cleaned up.")

    # --- 阶段 2: FDIP_2 ---
    print("\n--- Initializing Stage 2: FDIP_2 ---")
    model2 = FDIP_2(input_dim=6 * 12, output_dim=24 * 3).to(DEVICE)
    checkpoint_path2 = os.path.join(CHECKPOINT_DIR, 'ggip2', 'best_model_fdip2.pth')
    completion_marker2 = os.path.join(CHECKPOINT_DIR, 'ggip2', 'fdip2_completed.marker')

    if os.path.exists(completion_marker2):
        print("Stage 2 (FDIP_2) already completed. Loading best model and skipping training.")
        if os.path.exists(checkpoint_path2):
            checkpoint = torch.load(checkpoint_path2, map_location=DEVICE)
            model2.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded best model for FDIP_2 from {checkpoint_path2}")
            del checkpoint  # 清理checkpoint
        else:
            print(f"Error: Completion marker found, but checkpoint file {checkpoint_path2} is missing!")
            sys.exit(1)
    else:
        optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max_epochs, eta_min=1e-6)
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
            print(
                f"Resuming from Epoch {start_epoch2 + 1}. Best validation loss so far: {early_stopper2.val_loss_min:.6f}")
            del checkpoint  # 清理checkpoint
        else:
            print("No checkpoint found for FDIP_2. Starting training from scratch.")

        model2 = train_fdip_2(
            model1=model1,
            model2=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=max_epochs,
            early_stopper=early_stopper2,
            start_epoch=start_epoch2
        )

        with open(completion_marker2, 'w') as f:
            f.write(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Best model saved at epoch {early_stopper2.best_epoch} with val_loss {early_stopper2.val_loss_min:.6f}\n")
        print(f"Stage 2 (FDIP_2) marked as completed.")

        # 🔥 清理Stage 2的训练对象，释放内存
        cleanup_training_objects(optimizer2, scheduler2, early_stopper2)
        print("Stage 2 training objects cleaned up.")

    # --- 阶段 3: FDIP_3 ---
    print("\n--- Initializing Stage 3: FDIP_3 ---")
    model3 = FDIP_3(input_dim=288, output_dim=24 * 6).to(DEVICE)
    checkpoint_path3 = os.path.join(CHECKPOINT_DIR, 'ggip3', 'best_model_fdip3.pth')
    completion_marker3 = os.path.join(CHECKPOINT_DIR, 'ggip3', 'fdip3_completed.marker')

    if os.path.exists(completion_marker3):
        print("Stage 3 (FDIP_3) already completed. Loading best model and skipping training.")
        if os.path.exists(checkpoint_path3):
            checkpoint = torch.load(checkpoint_path3, map_location=DEVICE)
            model3.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded best model for FDIP_3 from {checkpoint_path3}")
            del checkpoint  # 清理checkpoint
        else:
            print(f"Error: Completion marker found, but checkpoint file {checkpoint_path3} is missing!")
            sys.exit(1)
    else:
        optimizer3 = optim.Adam(model3.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=max_epochs, eta_min=1e-6)
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
            print(
                f"Resuming from Epoch {start_epoch3 + 1}. Best validation loss so far: {early_stopper3.val_loss_min:.6f}")
            del checkpoint  # 清理checkpoint
        else:
            print("No checkpoint found for FDIP_3. Starting training from scratch.")

        model3 = train_fdip_3(
            model1=model1,
            model2=model2,
            model3=model3,
            optimizer=optimizer3,
            scheduler=scheduler3,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=max_epochs,
            early_stopper=early_stopper3,
            start_epoch=start_epoch3
        )

        with open(completion_marker3, 'w') as f:
            f.write(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Best model saved at epoch {early_stopper3.best_epoch} with val_loss {early_stopper3.val_loss_min:.6f}\n")
        print(f"Stage 3 (FDIP_3) marked as completed.")

        # 🔥 清理Stage 3的训练对象，释放内存
        cleanup_training_objects(optimizer3, scheduler3, early_stopper3)
        print("Stage 3 training objects cleaned up.")

    print("\nAll training stages complete!")
    total_end_time = time.time()
    print(f"Total training time: {(total_end_time - total_start_time) / 3600:.2f} hours")

    # 最终评估前再次清理内存
    clear_memory()
    evaluate_pipeline(model1, model2, model3, val_loader)

    print("\nTraining and evaluation finished successfully!")


if __name__ == '__main__':
    main()
