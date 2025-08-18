import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import pickle
import pandas as pd
from datetime import datetime
from articulate.math import r6d_to_rotation_matrix
from data.dataset_posReg import ImuDataset
from model.net_zd import FDIP_1, FDIP_2, FDIP_3
from evaluator import PoseEvaluator, PerFramePoseEvaluator
import gc
import argparse

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
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
TRAIN_DATA_FOLDERS = [
    os.path.join("D:\\", "Dataset", "TotalCapture_Real_60FPS", "KaPt", "split_actions"),
    os.path.join("D:\\", "Dataset", "DIPIMUandOthers", "DIP_6", "Detail"),
    os.path.join("D:\\", "Dataset", "AMASS", "DanceDB", "pt"),
    os.path.join("D:\\", "Dataset", "AMASS", "HumanEva", "pt"),
]
VAL_DATA_FOLDERS = [
    os.path.join("D:\\", "Dataset", "SingleOne", "pt"),
]

TIMESTAMP = None
CHECKPOINT_DIR = None
LOG_DIR = "log"
LOG_RUN_DIR = None
TRAINING_MODE = None  # 新增：训练模式标识


# ===== 改进的模型类 =====

class FDIP_2_Residual(nn.Module):
    """改进的FDIP_2，加入残差连接和门控机制"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.backbone = FDIP_2(input_dim, output_dim)

        # 残差连接层
        self.residual_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # 主分支输出
        main_output = self.backbone(x)

        # 残差分支
        # 对序列维度进行池化以减少计算量
        x_pooled = x.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        residual = self.residual_proj(x_pooled)

        # 门控融合
        gate_weight = self.gate(main_output)
        output = gate_weight * main_output + (1 - gate_weight) * residual

        # 层归一化
        output = self.layer_norm(output)

        return output


class FDIP_3_Residual(nn.Module):
    """改进的FDIP_3，加入残差连接和注意力机制"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.backbone = FDIP_3(input_dim, output_dim)

        # 特征维度
        self.feature_dim = 256

        # 多头注意力，用于融合IMU和位置信息
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)

        # 投影层
        self.imu_proj = nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.pos_proj = nn.Sequential(
            nn.Linear(72, self.feature_dim),  # 24*3 -> feature_dim
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, output_dim)
        )

        # 融合权重（可学习）
        self.fusion_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, imu_input, pos_input):
        # 原始输出
        main_output = self.backbone(imu_input, pos_input)

        # 注意力增强分支
        imu_features = self.imu_proj(imu_input)  # [B, S, feature_dim]
        pos_features = self.pos_proj(pos_input)  # [B, S, feature_dim]

        # 自注意力 - IMU特征
        attended_imu, _ = self.attention(imu_features, imu_features, imu_features)
        attended_imu = self.norm1(attended_imu + imu_features)

        # 交叉注意力 - IMU与位置特征
        cross_attended, _ = self.attention(attended_imu, pos_features, pos_features)
        enhanced_features = self.norm2(cross_attended + attended_imu)

        # 输出投影
        attention_output = self.output_proj(enhanced_features)

        # 自适应融合
        fusion_weight = torch.sigmoid(self.fusion_weight)
        final_output = fusion_weight * main_output + (1 - fusion_weight) * attention_output

        return final_output


# ===== 改进的早停类 =====

class MultiModelEarlyStopping:
    """支持多模型的早停机制"""

    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_epoch = 0

    def __call__(self, val_loss, models, optimizer, epoch):
        if not np.isfinite(val_loss):
            if self.verbose:
                print(f"Warning: Validation loss is {val_loss} at epoch {epoch}, skipping EarlyStopping.")
            return

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, optimizer, epoch):
        """保存多模型检查点"""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving checkpoint to {self.path}...')

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model1_state_dict': models[0].state_dict(),
            'model2_state_dict': models[1].state_dict(),
            'model3_state_dict': models[2].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss_min': val_loss,
            'best_score': self.best_score,
            'early_stopping_counter': self.counter,
            'training_mode': TRAINING_MODE  # 新增：记录训练模式
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch


# ===== 工具函数 =====

def create_directories():
    """创建必要的目录，根据训练模式区分"""
    if TRAINING_MODE == "joint":
        dirs = [
            os.path.join(CHECKPOINT_DIR, "joint_e2e_training"),  # 联合训练专用
            os.path.join(CHECKPOINT_DIR, "joint_e2e_logs"),  # 联合训练日志专用
            LOG_DIR,
            LOG_RUN_DIR,
        ]
    else:  # sequential模式
        dirs = [
            os.path.join(CHECKPOINT_DIR, "sequential_stage1"),  # 分阶段训练
            os.path.join(CHECKPOINT_DIR, "sequential_stage2"),
            os.path.join(CHECKPOINT_DIR, "sequential_stage3"),
            LOG_DIR,
            LOG_RUN_DIR,
        ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Directories created with timestamp {TIMESTAMP} (Mode: {TRAINING_MODE}):")
    for dir_path in dirs:
        print(f"  - {dir_path}")


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def setup_directories_and_paths(args):
    """设置全局路径变量，包含训练模式区分"""
    global TIMESTAMP, CHECKPOINT_DIR, LOG_RUN_DIR, TRAINING_MODE

    # 确定训练模式
    TRAINING_MODE = "joint" if args.use_joint_training else "sequential"

    if args.resume:
        TIMESTAMP = args.resume
        # 根据恢复的目录结构判断训练模式
        base_checkpoint_dir = os.path.join("GGIP", f"checkpoints_{TIMESTAMP}")
        if os.path.exists(os.path.join(base_checkpoint_dir, "joint_e2e_training")):
            TRAINING_MODE = "joint"
        elif os.path.exists(os.path.join(base_checkpoint_dir, "sequential_stage1")):
            TRAINING_MODE = "sequential"

        CHECKPOINT_DIR = base_checkpoint_dir
        print(f"Resuming training from timestamp: {TIMESTAMP} (Mode: {TRAINING_MODE})")

    elif args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
        TIMESTAMP = os.path.basename(CHECKPOINT_DIR).replace("checkpoints_", "")
        # 根据目录结构判断训练模式
        if os.path.exists(os.path.join(CHECKPOINT_DIR, "joint_e2e_training")):
            TRAINING_MODE = "joint"
        elif os.path.exists(os.path.join(CHECKPOINT_DIR, "sequential_stage1")):
            TRAINING_MODE = "sequential"
        print(f"Using checkpoint directory: {CHECKPOINT_DIR} (Mode: {TRAINING_MODE})")

    else:
        # 新训练：在时间戳中包含训练模式标识
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "joint" if TRAINING_MODE == "joint" else "seq"
        TIMESTAMP = f"{base_timestamp}_{mode_suffix}"
        CHECKPOINT_DIR = os.path.join("GGIP", f"checkpoints_{TIMESTAMP}")
        print(f"Starting new {TRAINING_MODE} training with timestamp: {TIMESTAMP}")

    # 设置日志目录，也包含模式标识
    LOG_RUN_DIR = os.path.join(LOG_DIR, f"{TIMESTAMP}_{TRAINING_MODE}")

    if not os.path.exists(CHECKPOINT_DIR) and (args.resume or args.checkpoint_dir):
        print(f"Error: Checkpoint directory {CHECKPOINT_DIR} does not exist!")
        sys.exit(1)

    print(f"Global paths set:")
    print(f"  - TRAINING_MODE: {TRAINING_MODE}")
    print(f"  - TIMESTAMP: {TIMESTAMP}")
    print(f"  - CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"  - LOG_RUN_DIR: {LOG_RUN_DIR}")


def clear_memory():
    """清理GPU和CPU内存"""
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


def cleanup_training_objects(*objects):
    """清理训练相关对象"""
    for obj in objects:
        if obj is not None:
            del obj
    clear_memory()


def load_data_unified_split(train_percent=0.8, val_percent=0.2, seed=None):
    """统一加载所有数据集，然后随机划分"""
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
        val_size = total_size - train_size

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
    """检查训练集和验证集的数据分布一致性"""
    print("\n=== Data Distribution Analysis ===")

    def compute_stats(data_loader, name, max_batches=num_samples):
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

        for key in stats:
            stats[key] = np.mean(stats[key])

        return stats

    print("Computing training set statistics...")
    train_stats = compute_stats(train_loader, "Train")

    print("Computing validation set statistics...")
    val_stats = compute_stats(val_loader, "Validation")

    print(f"\nDistribution Comparison (based on {num_samples} batches):")
    print(f"{'Metric':<15} {'Train':<12} {'Validation':<12} {'Difference':<12}")
    print("-" * 55)

    for key in train_stats:
        train_val = train_stats[key]
        val_val = val_stats[key]
        diff = abs(train_val - val_val)
        print(f"{key:<15} {train_val:<12.6f} {val_val:<12.6f} {diff:<12.6f}")

    total_diff = sum([abs(train_stats[key] - val_stats[key]) for key in train_stats])
    print(f"\nTotal Difference Score: {total_diff:.6f} (lower is better)")

    if total_diff < 0.1:
        print("✓ Data distributions appear consistent!")
    elif total_diff < 0.5:
        print("⚠ Data distributions have minor differences")
    else:
        print("❌ Data distributions have significant differences")

    return train_stats, val_stats


def rotation_matrix_loss(pred_6d, target_6d):
    """将6D表示转换为旋转矩阵后计算Frobenius范数损失"""
    batch_size, seq_len, joints, _ = pred_6d.shape
    pred_6d_flat = pred_6d.reshape(-1, 6)
    target_6d_flat = target_6d.reshape(-1, 6)

    pred_rotmat = r6d_to_rotation_matrix(pred_6d_flat)
    target_rotmat = r6d_to_rotation_matrix(target_6d_flat)

    loss = torch.mean(torch.norm(pred_rotmat - target_rotmat, dim=(-2, -1)))
    return loss


# ===== 端到端联合训练函数 =====

def train_end_to_end_joint(model1, model2, model3, optimizer, scheduler, train_loader, val_loader, epochs,
                           early_stopper, start_epoch=0):
    """端到端联合训练所有三个模型"""
    print("\n====================== Starting End-to-End Joint Training =========================")
    print(f"🚀 Using Joint E2E Training Mode - All models trained simultaneously")
    print(f"📊 Checkpoint saving to: {early_stopper.path}")

    criterion = nn.MSELoss()
    scaler = GradScaler()
    # 使用专门的联合训练日志目录
    writer = SummaryWriter(os.path.join(LOG_RUN_DIR, 'joint_e2e_logs')) if LOG_ENABLED else None

    # 定义多任务损失权重
    loss_weights = {
        'leaf_pos': 1.0,
        'all_pos': 1.5,
        'pose_6d': 3.0
    }

    print(f"📈 Loss weights: {loss_weights}")

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1

        model1.train()
        model2.train()
        model3.train()

        train_losses = {'total': [], 'leaf_pos': [], 'all_pos': [], 'pose_6d': []}
        epoch_pbar = tqdm(train_loader, desc=f"🔄 Joint E2E Epoch {current_epoch}/{epochs}", leave=True)

        for data in epoch_pbar:
            acc = data[0].to(DEVICE, non_blocking=True).float()
            ori_6d = data[2].to(DEVICE, non_blocking=True).float()
            p_leaf_gt = data[3].to(DEVICE, non_blocking=True).float()
            p_all_gt = data[4].to(DEVICE, non_blocking=True).float()
            pose_6d_gt = data[6].to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # === 级联前向传播 ===
                # FDIP_1: 预测叶节点位置
                input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                p_leaf_pred = model1(input1)

                # FDIP_2: 预测所有关节位置
                zeros = torch.zeros(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 3, device=DEVICE)
                p_leaf_with_root = torch.cat([zeros, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], -1)],
                                             dim=2)
                input2 = torch.cat(
                    [acc, ori_6d, p_leaf_with_root.view(p_leaf_with_root.shape[0], p_leaf_with_root.shape[1], 6, 3)],
                    dim=-1).view(acc.shape[0], acc.shape[1], -1)
                p_all_pred = model2(input2)

                # FDIP_3: 预测6D姿态
                input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                pose_6d_pred = model3(input_base, p_all_pred)

                # === 计算多任务损失 ===
                # 叶节点位置损失
                loss_leaf = torch.sqrt(criterion(p_leaf_pred, p_leaf_gt.view(-1, p_leaf_gt.shape[1], 15)))

                # 所有关节位置损失
                p_all_target = torch.cat([torch.zeros_like(p_all_gt[:, :, 0:1, :]), p_all_gt], dim=2).view(
                    p_all_gt.shape[0], p_all_gt.shape[1], -1)
                loss_all_pos = torch.sqrt(criterion(p_all_pred, p_all_target))

                # 6D姿态损失
                batch_size, seq_len = pose_6d_pred.shape[:2]
                pose_pred_reshaped = pose_6d_pred.view(batch_size, seq_len, 24, 6)
                loss_pose = rotation_matrix_loss(pose_pred_reshaped, pose_6d_gt)

                # 加权总损失
                total_loss = (loss_weights['leaf_pos'] * loss_leaf +
                              loss_weights['all_pos'] * loss_all_pos +
                              loss_weights['pose_6d'] * loss_pose)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss at epoch {current_epoch}, skipping batch.")
                continue

            scaler.scale(total_loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for model in [model1, model2, model3] for p in model.parameters()],
                                           max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # 记录损失
            train_losses['total'].append(total_loss.item())
            train_losses['leaf_pos'].append(loss_leaf.item())
            train_losses['all_pos'].append(loss_all_pos.item())
            train_losses['pose_6d'].append(loss_pose.item())

            epoch_pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'leaf': f"{loss_leaf.item():.4f}",
                'pos': f"{loss_all_pos.item():.4f}",
                'pose': f"{loss_pose.item():.4f}"
            })

        # === 验证阶段 ===
        model1.eval()
        model2.eval()
        model3.eval()
        val_losses = {'total': [], 'leaf_pos': [], 'all_pos': [], 'pose_6d': []}

        with torch.no_grad():
            for data_val in val_loader:
                acc_val = data_val[0].to(DEVICE, non_blocking=True).float()
                ori_val = data_val[2].to(DEVICE, non_blocking=True).float()
                p_leaf_gt_val = data_val[3].to(DEVICE, non_blocking=True).float()
                p_all_gt_val = data_val[4].to(DEVICE, non_blocking=True).float()
                pose_6d_gt_val = data_val[6].to(DEVICE, non_blocking=True).float()

                # 验证前向传播
                input1_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                p_leaf_pred_val = model1(input1_val)

                zeros_val = torch.zeros(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], 3, device=DEVICE)
                p_leaf_with_root_val = torch.cat(
                    [zeros_val, p_leaf_pred_val.view(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], -1)], dim=2)
                input2_val = torch.cat([acc_val, ori_val, p_leaf_with_root_val.view(p_leaf_with_root_val.shape[0],
                                                                                    p_leaf_with_root_val.shape[1], 6,
                                                                                    3)], dim=-1).view(acc_val.shape[0],
                                                                                                      acc_val.shape[1],
                                                                                                      -1)
                p_all_pred_val = model2(input2_val)

                input_base_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                pose_6d_pred_val = model3(input_base_val, p_all_pred_val)

                # 验证损失计算
                loss_leaf_val = torch.sqrt(
                    criterion(p_leaf_pred_val, p_leaf_gt_val.view(-1, p_leaf_gt_val.shape[1], 15)))
                p_all_target_val = torch.cat([torch.zeros_like(p_all_gt_val[:, :, 0:1, :]), p_all_gt_val], dim=2).view(
                    p_all_gt_val.shape[0], p_all_gt_val.shape[1], -1)
                loss_all_pos_val = torch.sqrt(criterion(p_all_pred_val, p_all_target_val))

                batch_size_val, seq_len_val = pose_6d_pred_val.shape[:2]
                pose_pred_reshaped_val = pose_6d_pred_val.view(batch_size_val, seq_len_val, 24, 6)
                loss_pose_val = rotation_matrix_loss(pose_pred_reshaped_val, pose_6d_gt_val)

                total_loss_val = (loss_weights['leaf_pos'] * loss_leaf_val +
                                  loss_weights['all_pos'] * loss_all_pos_val +
                                  loss_weights['pose_6d'] * loss_pose_val)

                if not torch.isnan(total_loss_val) and not torch.isinf(total_loss_val):
                    val_losses['total'].append(total_loss_val.item())
                    val_losses['leaf_pos'].append(loss_leaf_val.item())
                    val_losses['all_pos'].append(loss_all_pos_val.item())
                    val_losses['pose_6d'].append(loss_pose_val.item())

        # 打印训练结果
        avg_train_losses = {k: np.mean(v) if v else 0.0 for k, v in train_losses.items()}
        avg_val_losses = {k: np.mean(v) if v else 0.0 for k, v in val_losses.items()}
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\n🔄 Joint E2E Epoch {current_epoch}/{epochs} | LR: {current_lr:.6f}')
        print(
            f'  📈 Train - Total: {avg_train_losses["total"]:.6f}, Leaf: {avg_train_losses["leaf_pos"]:.6f}, Pos: {avg_train_losses["all_pos"]:.6f}, Pose: {avg_train_losses["pose_6d"]:.6f}')
        print(
            f'  📉 Val   - Total: {avg_val_losses["total"]:.6f}, Leaf: {avg_val_losses["leaf_pos"]:.6f}, Pos: {avg_val_losses["all_pos"]:.6f}, Pose: {avg_val_losses["pose_6d"]:.6f}')

       # 计算损失比率
        if avg_train_losses["total"] > 0:
           loss_ratio = avg_val_losses["total"] / avg_train_losses["total"]
           print(f'  📊 Loss Ratio (Val/Train): {loss_ratio:.3f}')

        if LOG_ENABLED and writer:
           for loss_type in train_losses.keys():
               writer.add_scalars(f'joint_e2e_loss/{loss_type}', {
                   'train': avg_train_losses[loss_type],
                   'val': avg_val_losses[loss_type]
               }, current_epoch)
           writer.add_scalar('joint_e2e_learning_rate', current_lr, current_epoch)
           writer.add_scalar('joint_e2e_loss_ratio', loss_ratio if avg_train_losses["total"] > 0 else 0, current_epoch)

        scheduler.step()
        early_stopper(avg_val_losses['total'], [model1, model2, model3], optimizer, current_epoch)
        if early_stopper.early_stop:
           print(f"🛑 Early stopping triggered at epoch {current_epoch} for Joint E2E Training.")
           break

        torch.cuda.empty_cache()

        # 加载最佳模型
        if os.path.exists(early_stopper.path):
            print(f"✅ Loading best joint E2E model from epoch {early_stopper.best_epoch}")
        checkpoint = torch.load(early_stopper.path, map_location=DEVICE)
        model1.load_state_dict(checkpoint['model1_state_dict'])
        model2.load_state_dict(checkpoint['model2_state_dict'])
        model3.load_state_dict(checkpoint['model3_state_dict'])
        del checkpoint

        if writer:
            writer.close()
        del writer

        del criterion, scaler
        torch.cuda.empty_cache()

        print("======================== End-to-End Joint Training Finished =======================================")
        return model1, model2, model3


# ===== 评估函数 =====
def evaluate_pipeline(model1, model2, model3, data_loader):
   print("\n============================ Evaluating Complete Pipeline ======================================")
   print(f"📊 Evaluation for {TRAINING_MODE} training mode")

   # 评估前清理内存
   clear_memory()

   # 根据训练模式创建不同的评估目录
   eval_results_dir = os.path.join("GGIP", f"evaluate_{TRAINING_MODE}_pipeline_{TIMESTAMP}")
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

       print(f"Running {TRAINING_MODE} model evaluation...")
       with torch.no_grad():
           for data_val in tqdm(data_loader, desc=f"Evaluating {TRAINING_MODE.upper()} Pipeline"):
               try:
                   # 模型前向传播
                   acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                              (data_val[0], data_val[2], data_val[6])]

                   # 级联推理
                   input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                   p_leaf_logits = model1(input1)

                   zeros1 = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
                   p_leaf_pred = torch.cat(
                       [zeros1, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], -1)], dim=2)

                   input2 = torch.cat(
                       [acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 6, 3)],
                       dim=-1).view(acc.shape[0], acc.shape[1], -1)
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

       # 评估结果处理
       clear_memory()

       if all_errors["mesh_err"]:
           print("Processing evaluation results...")

           # 拼接所有误差数据
           final_errors = {key: torch.cat(val, dim=0) for key, val in all_errors.items() if val}
           avg_errors = {key: val.mean().item() for key, val in final_errors.items()}

           # 打印结果
           print(f"\n🎯 {TRAINING_MODE.upper()} Pipeline Evaluation Results (Mean):")
           print(f"  - Positional Error (cm):      {avg_errors.get('pos_err', 'N/A'):.4f}")
           print(f"  - Mesh Error (cm):            {avg_errors.get('mesh_err', 'N/A'):.4f}")
           print(f"  - Angular Error (deg):        {avg_errors.get('angle_err', 'N/A'):.4f}")
           print(f"  - Jitter Error (cm/s²):       {avg_errors.get('jitter_err', 'N/A'):.4f}")

           # 保存数据到文件
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

           try:
               # 1. 保存原始误差数据
               raw_data_path = os.path.join(eval_data_dir, f"{TRAINING_MODE}_raw_errors_{timestamp}.pkl")
               with open(raw_data_path, 'wb') as f:
                   pickle.dump(final_errors, f)
               print(f"Raw error data saved to: {raw_data_path}")

               # 2. 保存统计结果
               stats_data = {
                   "timestamp": timestamp,
                   "training_mode": TRAINING_MODE,
                   "evaluation_results": {
                       "mean_errors": avg_errors,
                       "sample_counts": {key: len(val) for key, val in final_errors.items()},
                       "std_errors": {key: val.std().item() for key, val in final_errors.items()},
                       "min_errors": {key: val.min().item() for key, val in final_errors.items()},
                       "max_errors": {key: val.max().item() for key, val in final_errors.items()}
                   },
                   "units": {
                       "pos_err": "cm",
                       "mesh_err": "cm",
                       "angle_err": "degrees",
                       "jitter_err": "cm/s²"
                   }
               }

               stats_path = os.path.join(eval_data_dir, f"{TRAINING_MODE}_evaluation_stats_{timestamp}.json")
               with open(stats_path, 'w') as f:
                   json.dump(stats_data, f, indent=2)
               print(f"Statistics saved to: {stats_path}")

               # 3. 保存为CSV格式
               csv_data = []
               for key, values in final_errors.items():
                   for value in values.numpy():
                       csv_data.append({
                           'training_mode': TRAINING_MODE,
                           'metric': key,
                           'value': value,
                           'timestamp': timestamp
                       })

               if csv_data:
                   df = pd.DataFrame(csv_data)
                   csv_path = os.path.join(eval_data_dir, f"{TRAINING_MODE}_evaluation_data_{timestamp}.csv")
                   df.to_csv(csv_path, index=False)
                   print(f"CSV data saved to: {csv_path}")

           except Exception as e:
               print(f"Warning: Error saving data files: {e}")

       else:
           print("No evaluation results were generated.")

   except Exception as e:
       print(f"Critical error in evaluation pipeline: {e}")

   print(f"\n✅ {TRAINING_MODE.upper()} evaluation completed. Results saved in: {eval_results_dir}")


def clean_filename(filename):
   """清理文件名，移除Windows中不允许的字符"""
   invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
   for char in invalid_chars:
       filename = filename.replace(char, '_')
   return filename


def parse_args():
   parser = argparse.ArgumentParser(description='Improved FDIP Training with Joint Learning')
   parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint directory (e.g., 20250804_143022_joint)')
   parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Specific checkpoint directory path')
   parser.add_argument('--use_joint_training', action='store_true', default=True,
                       help='Use end-to-end joint training (default: True)')
   parser.add_argument('--use_residual', action='store_true', default=True,
                       help='Use residual connections in models (default: True)')
   parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size (default: 64)')
   parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
   parser.add_argument('--max_epochs', type=int, default=150,
                       help='Maximum training epochs (default: 150)')
   parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')

   return parser.parse_args()


# ===== 主训练函数 =====

def main():
   """改进的主函数，支持端到端联合训练"""
   args = parse_args()

   global BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE
   BATCH_SIZE = args.batch_size
   LEARNING_RATE = args.learning_rate
   MAX_EPOCHS = args.max_epochs
   PATIENCE = args.patience

   set_seed(SEED)
   print("==================== Starting Improved Training Pipeline =====================")
   print(f"🚀 Training mode: {'Joint End-to-End' if args.use_joint_training else 'Sequential Stages'}")
   print(f"🔧 Residual connections: {'Enabled' if args.use_residual else 'Disabled'}")

   setup_directories_and_paths(args)
   create_directories()

   # 数据加载
   try:
       train_loader, val_loader = load_data_unified_split(
           train_percent=0.8,
           val_percent=0.2,
           seed=SEED
       )
       print("✅ Using unified dataset with consistent split!")
       check_data_distribution(train_loader, val_loader)

   except Exception as e:
       print(f"❌ Failed to load unified datasets: {e}")
       sys.exit(1)

   total_start_time = time.time()

   if args.use_joint_training:
       print(f"\n🎯 === Using End-to-End Joint Training Mode ===")

       # 初始化模型
       model1 = FDIP_1(input_dim=6 * 9, output_dim=5 * 3).to(DEVICE)

       if args.use_residual:
           model2 = FDIP_2_Residual(input_dim=6 * 12, output_dim=24 * 3).to(DEVICE)
           model3 = FDIP_3_Residual(input_dim=6 * 9, output_dim=24 * 6).to(DEVICE)
           print("✅ Using residual-enhanced models")
       else:
           model2 = FDIP_2(input_dim=6 * 12, output_dim=24 * 3).to(DEVICE)
           model3 = FDIP_3(input_dim=6 * 9, output_dim=24 * 6).to(DEVICE)
           print("✅ Using original models")

       # 检查点路径 - 使用联合训练专用路径
       checkpoint_path = os.path.join(CHECKPOINT_DIR, 'joint_e2e_training', 'best_joint_e2e_model.pth')
       completion_marker = os.path.join(CHECKPOINT_DIR, 'joint_e2e_training', 'joint_e2e_completed.marker')

       # 检查是否已完成训练
       if os.path.exists(completion_marker) and not args.resume:
           print("🎉 Joint E2E training already completed. Loading best models and skipping training.")
           if os.path.exists(checkpoint_path):
               checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
               model1.load_state_dict(checkpoint['model1_state_dict'])
               model2.load_state_dict(checkpoint['model2_state_dict'])
               model3.load_state_dict(checkpoint['model3_state_dict'])
               print(f"✅ Successfully loaded best joint E2E models from {checkpoint_path}")
               del checkpoint
           else:
               print(f"❌ Error: Completion marker found, but checkpoint file {checkpoint_path} is missing!")
               sys.exit(1)
       else:
           # 设置联合优化器 - 对不同模块使用不同学习率
           param_groups = [
               {'params': model1.parameters(), 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
               {'params': model2.parameters(), 'lr': LEARNING_RATE * 0.8, 'weight_decay': WEIGHT_DECAY},
               {'params': model3.parameters(), 'lr': LEARNING_RATE * 0.6, 'weight_decay': WEIGHT_DECAY},
           ]
           optimizer = optim.AdamW(param_groups)
           scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
               optimizer, T_0=20, T_mult=2, eta_min=1e-6
           )

           early_stopper = MultiModelEarlyStopping(
               patience=PATIENCE,
               path=checkpoint_path,
               verbose=True
           )

           start_epoch = 0

           # 检查是否有断点可以恢复
           if os.path.exists(checkpoint_path):
               print(f"🔄 Found joint E2E checkpoint. Resuming training from: {checkpoint_path}")
               checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
               model1.load_state_dict(checkpoint['model1_state_dict'])
               model2.load_state_dict(checkpoint['model2_state_dict'])
               model3.load_state_dict(checkpoint['model3_state_dict'])
               optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
               start_epoch = checkpoint['epoch']
               early_stopper.val_loss_min = checkpoint['val_loss_min']
               early_stopper.best_score = checkpoint['best_score']
               early_stopper.counter = checkpoint.get('early_stopping_counter', 0)

               # 恢复调度器状态
               for _ in range(start_epoch):
                   scheduler.step()

               print(f"🎯 Resuming from Epoch {start_epoch + 1}. Best validation loss: {early_stopper.val_loss_min:.6f}")
               del checkpoint
           else:
               print("🆕 No joint E2E checkpoint found. Starting training from scratch.")

           # 执行联合训练
           model1, model2, model3 = train_end_to_end_joint(
               model1, model2, model3, optimizer, scheduler,
               train_loader, val_loader, MAX_EPOCHS, early_stopper, start_epoch
           )

           # 训练完成标记
           with open(completion_marker, 'w') as f:
               f.write(f"Joint E2E training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
               f.write(f"Training mode: {TRAINING_MODE}\n")
               f.write(f"Residual connections: {'Enabled' if args.use_residual else 'Disabled'}\n")
               f.write(
                   f"Best model saved at epoch {early_stopper.best_epoch} with val_loss {early_stopper.val_loss_min:.6f}\n")
           print(f"🏁 Joint E2E training marked as completed.")

           # 清理训练对象
           cleanup_training_objects(optimizer, scheduler, early_stopper)

   else:
       print("\n⚠️ === Using Sequential Stage Training Mode ===")
       print("Sequential training not implemented in this version. Please use joint training.")
       return

   print(f"\n🎉 All {TRAINING_MODE} training complete!")
   total_end_time = time.time()
   print(f"⏱️  Total training time: {(total_end_time - total_start_time) / 3600:.2f} hours")

   clear_memory()
   evaluate_pipeline(model1, model2, model3, val_loader)
   print(f"\n🏆 Improved {TRAINING_MODE} training and evaluation finished successfully!")


if __name__ == '__main__':
   main()

