import os
import sys

import inspect

# === getargspec兼容性修复 ===
if not hasattr(inspect, 'getargspec'):
    def getargspec_compat(func):
        """兼容性包装器，替代已弃用的getargspec"""
        try:
            sig = inspect.signature(func)
            args = list(sig.parameters.keys())
            defaults = []
            varargs = None
            keywords = None

            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    varargs = param.name
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    keywords = param.name
                elif param.default != inspect.Parameter.empty:
                    defaults.append(param.default)

            # 构造与getargspec相同的返回格式
            from collections import namedtuple
            ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
            return ArgSpec(
                args=args,
                varargs=varargs,
                keywords=keywords,
                defaults=tuple(defaults) if defaults else None
            )
        except Exception as e:
            print(f"Warning: getargspec compatibility failed for {func}: {e}")
            from collections import namedtuple
            ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
            return ArgSpec(args=[], varargs=None, keywords=None, defaults=None)


    inspect.getargspec = getargspec_compat
    print("✅ Applied inspect.getargspec compatibility patch for Python 3.11+")


import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.amp import autocast, GradScaler
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
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 5e-3
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
    os.path.join("F:\\", "IMUdata", "TotalCapture_Real_60FPS", "KaPt"),
    os.path.join("F:\\", "IMUdata", "DIPIMUandOthers", "DIP_6"),
    os.path.join("F:\\", "IMUdata", "AMASS", "DanceDB", "pt"),
    os.path.join("F:\\", "IMUdata", "AMASS", "HumanEva", "pt"),
]
VAL_DATA_FOLDERS = [
    os.path.join("F:\\", "IMUdata", "SingleOne", "pt"),
]

TIMESTAMP = None
CHECKPOINT_DIR = None
LOG_DIR = "log"
LOG_RUN_DIR = None
TRAINING_MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='Improved FDIP Training with Acceleration Normalization Only')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint directory (e.g., 20250804_143022_joint)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Specific checkpoint directory path')
    parser.add_argument('--use_joint_training', action='store_true', default=True,
                        help='Use end-to-end joint training (default: True)')
    parser.add_argument('--use_residual', action='store_true', default=False,
                        help='Use residual connections in models (default: False)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Maximum training epochs (default: 150)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')

    # 新增参数：续训时是否重置最佳损失记录
    parser.add_argument('--reset_best_on_resume', action='store_true', default=False,
                        help='Reset best validation loss when resuming training. Useful when loss weights are changed.')

    return parser.parse_args()


class AccelerationNormalizer:
    """只对加速度数据进行标准化的类"""

    def __init__(self):
        self.acc_mean = None
        self.acc_std = None
        self.fitted = False

    def fit(self, train_loader, max_batches=50):
        """在训练数据上计算加速度统计量"""
        print("Computing acceleration normalization statistics...")
        acc_vals = []

        batch_count = 0
        for data in tqdm(train_loader, desc="Computing acc stats"):
            if batch_count >= max_batches:
                break

            try:
                acc = data[0].float()  # 只取加速度数据

                # 基础数据质量检查
                if not torch.isnan(acc).any() and not torch.isinf(acc).any() and acc.abs().max() < 100:
                    acc_vals.append(acc)
                    batch_count += 1
                else:
                    continue

            except Exception as e:
                print(f"Warning: Error in batch {batch_count}: {e}")
                continue

        if not acc_vals:
            print("Warning: No valid acceleration data found for normalization!")
            return False

        # 合并所有加速度数据
        acc_all = torch.cat(acc_vals, dim=0)  # [Total_samples, seq_len, joint_num, 3]

        # 计算加速度的均值和标准差 - 对每个关节每个坐标轴独立计算
        self.acc_mean = acc_all.mean(dim=(0, 1), keepdim=True)  # [1, 1, joint_num, 3]
        self.acc_std = acc_all.std(dim=(0, 1), keepdim=True) + 1e-8

        # 打印统计信息
        print(f"📊 Acceleration Normalization Statistics:")
        print(f"   - Data shape: {acc_all.shape}")
        print(f"   - Acc range: [{acc_all.min():.3f}, {acc_all.max():.3f}]")
        print(f"   - Mean shape: {self.acc_mean.shape}")
        print(f"   - Global mean: {self.acc_mean.mean():.4f}")
        print(f"   - Global std: {self.acc_std.mean():.4f}")

        # 检查每个关节的统计量
        num_joints = acc_all.shape[2]
        for joint_idx in range(min(num_joints, 3)):  # 只显示前3个关节的信息
            joint_mean = self.acc_mean[0, 0, joint_idx].mean().item()
            joint_std = self.acc_std[0, 0, joint_idx].mean().item()
            print(f"   - Joint {joint_idx}: mean={joint_mean:.4f}, std={joint_std:.4f}")

        self.fitted = True

        # 清理内存
        del acc_vals, acc_all
        torch.cuda.empty_cache()

        return True

    def normalize_acc(self, acc):
        """只标准化加速度数据"""
        if not self.fitted:
            print("Warning: Normalizer not fitted yet, returning original data")
            return acc

        acc_norm = (acc - self.acc_mean.to(acc.device)) / (self.acc_std.to(acc.device) + 1e-8)
        return acc_norm

    def denormalize_acc(self, acc_norm):
        """反标准化加速度数据"""
        if not self.fitted:
            return acc_norm
        return acc_norm * self.acc_std.to(acc_norm.device) + self.acc_mean.to(acc_norm.device)

    def save(self, path):
        """保存标准化参数"""
        if self.fitted:
            torch.save({
                'acc_mean': self.acc_mean,
                'acc_std': self.acc_std,
                'fitted': self.fitted
            }, path)
            print(f"📏 Acceleration normalizer saved to: {path}")

    def load(self, path):
        """加载标准化参数"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.acc_mean = checkpoint['acc_mean']
            self.acc_std = checkpoint['acc_std']
            self.fitted = checkpoint.get('fitted', True)
            print(f"📏 Acceleration normalizer loaded from: {path}")
            return True
        return False


def improved_data_check(acc, ori_6d, pos=None):
    """改进的数据质量检查"""
    # 更宽松的极值检查
    acc_max_threshold = 430.0
    ori_max_threshold = 10.0
    pos_max_threshold = 50.0

    # NaN/Inf检查
    if (torch.isnan(acc).any() or torch.isnan(ori_6d).any() or
            torch.isinf(acc).any() or torch.isinf(ori_6d).any()):
        return False, "NaN/Inf detected in acc/ori"

    if pos is not None:
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            return False, "NaN/Inf in position data"
        if pos.abs().max() > pos_max_threshold:
            return False, f"Extreme pos values: {pos.abs().max():.3f}"

    # 极值检查
    if acc.abs().max() > acc_max_threshold:
        return False, f"Extreme acc values: {acc.abs().max():.3f}"

    if ori_6d.abs().max() > ori_max_threshold:
        return False, f"Extreme ori values: {ori_6d.abs().max():.3f}"

    return True, "OK"


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
            nn.Dropout(0.05),
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

    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt', reset_best=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_epoch = 0
        self.reset_best = reset_best  # 新增：是否重置最佳记录

    def reset_best_values(self, current_epoch):
        """重置最佳值，用于续训时处理损失权重变化的情况"""
        if self.verbose:
            print(f"🔄 Resetting best validation loss due to training configuration changes...")
            print(f"   Previous best: {self.val_loss_min:.6f} at epoch {self.best_epoch}")

        self.best_score = None
        self.val_loss_min = np.inf
        self.counter = 0
        self.best_epoch = current_epoch

        if self.verbose:
            print(f"   Reset complete. Will save next valid validation loss as new best.")

    def __call__(self, val_loss, models, optimizer, epoch, normalizer=None):
        if not np.isfinite(val_loss) or val_loss == 0.0:
            if self.verbose:
                print(f"Warning: Validation loss is {val_loss} at epoch {epoch}, skipping EarlyStopping.")
            return

        score = -val_loss

        # 如果是重置模式且这是第一次调用
        if self.reset_best and self.best_score is None and self.val_loss_min == np.inf:
            if self.verbose:
                print(f"🆕 First validation after reset - accepting current loss {val_loss:.6f} as new best")
            self.best_score = score
            self.save_checkpoint(val_loss, models, optimizer, epoch, normalizer)
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models, optimizer, epoch, normalizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'Current: {val_loss:.6f} vs Best: {self.val_loss_min:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models, optimizer, epoch, normalizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, optimizer, epoch, normalizer=None):
        """保存多模型检查点"""
        if self.verbose:
            improvement = self.val_loss_min - val_loss if self.val_loss_min != np.inf else 0
            print(f'✅ Validation loss {"improved" if improvement > 0 else "updated"} '
                  f'({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving checkpoint to {self.path}...')

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
            'training_mode': TRAINING_MODE,
            'reset_best': self.reset_best  # 保存重置标志
        }
        torch.save(checkpoint, self.path)

        # 单独保存标准化器
        if normalizer is not None and normalizer.fitted:
            norm_path = self.path.replace('.pth', '_normalizer.pth')
            normalizer.save(norm_path)

        self.val_loss_min = val_loss
        self.best_epoch = epoch


# ===== 工具函数 =====

def create_directories():
    """创建必要的目录，根据训练模式区分"""
    if TRAINING_MODE == "joint":
        dirs = [
            os.path.join(CHECKPOINT_DIR, "joint_e2e_training"),
            os.path.join(CHECKPOINT_DIR, "joint_e2e_logs"),
            LOG_DIR,
            LOG_RUN_DIR,
        ]
    else:
        dirs = [
            os.path.join(CHECKPOINT_DIR, "sequential_stage1"),
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

    TRAINING_MODE = "joint" if args.use_joint_training else "sequential"

    if args.resume:
        TIMESTAMP = args.resume
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
        if os.path.exists(os.path.join(CHECKPOINT_DIR, "joint_e2e_training")):
            TRAINING_MODE = "joint"
        elif os.path.exists(os.path.join(CHECKPOINT_DIR, "sequential_stage1")):
            TRAINING_MODE = "sequential"
        print(f"Using checkpoint directory: {CHECKPOINT_DIR} (Mode: {TRAINING_MODE})")

    else:
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "joint" if TRAINING_MODE == "joint" else "seq"
        TIMESTAMP = f"{base_timestamp}_{mode_suffix}"
        CHECKPOINT_DIR = os.path.join("GGIP", f"checkpoints_{TIMESTAMP}")
        print(f"Starting new {TRAINING_MODE} training with timestamp: {TIMESTAMP}")

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
    current_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
    print(f"GPU memory after cleanup: {current_mem:.2f} GB")


def cleanup_training_objects(*objects):
    """清理训练相关对象"""
    for obj in objects:
        if obj is not None:
            del obj
    clear_memory()


def load_data_unified_split(train_percent=0.8, val_percent=0.2, seed=None):
    """统一加载所有数据集，然后随机划分，并计算加速度标准化参数"""
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

    num_workers = 0 if sys.platform == "win32" else 2

    # 创建训练数据加载器（用于计算标准化参数）
    train_loader_for_norm = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 计算统计量时不需要shuffle
        pin_memory=True,
        num_workers=num_workers
    )

    # 计算加速度标准化参数
    print("📊 Computing acceleration normalization statistics...")
    normalizer = AccelerationNormalizer()
    if not normalizer.fit(train_loader_for_norm):
        print("❌ Failed to compute normalization statistics!")
        sys.exit(1)

    # 创建正式的数据加载器
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

    return train_loader, val_loader, normalizer


def check_data_distribution(train_loader, val_loader, normalizer, num_samples=3):
    """检查训练集和验证集的数据分布一致性（包括标准化后的分布）"""
    print("\n=== Data Distribution Analysis ===")

    def compute_stats(data_loader, name, normalizer, max_batches=num_samples):
        stats = {
            'acc_mean': [], 'acc_std': [], 'acc_norm_mean': [], 'acc_norm_std': [],
            'ori_mean': [], 'ori_std': [],
            'pos_mean': [], 'pos_std': []
        }

        count = 0
        with torch.no_grad():
            for data in data_loader:
                if count >= max_batches:
                    break

                try:
                    acc = data[0].float()
                    ori = data[2].float()
                    pos = data[3].float()

                    # 过滤极端值
                    if acc.abs().max() < 100 and ori.abs().max() < 20:
                        # 原始数据统计
                        stats['acc_mean'].append(acc.mean().item())
                        stats['acc_std'].append(acc.std().item())
                        stats['ori_mean'].append(ori.mean().item())
                        stats['ori_std'].append(ori.std().item())
                        stats['pos_mean'].append(pos.mean().item())
                        stats['pos_std'].append(pos.std().item())

                        # 标准化后的加速度统计
                        if normalizer and normalizer.fitted:
                            acc_norm = normalizer.normalize_acc(acc)
                            stats['acc_norm_mean'].append(acc_norm.mean().item())
                            stats['acc_norm_std'].append(acc_norm.std().item())

                    count += 1
                except Exception:
                    count += 1
                    continue

        # 计算平均统计量
        for key in stats:
            if stats[key]:
                stats[key] = np.mean(stats[key])
            else:
                stats[key] = 0.0

        return stats

    print("Computing training set statistics...")
    train_stats = compute_stats(train_loader, "Train", normalizer)

    print("Computing validation set statistics...")
    val_stats = compute_stats(val_loader, "Validation", normalizer)

    print(f"\nDistribution Comparison (based on {num_samples} batches):")
    print(f"{'Metric':<20} {'Train':<12} {'Validation':<12} {'Difference':<12}")
    print("-" * 60)

    for key in train_stats:
        train_val = train_stats[key]
        val_val = val_stats[key]
        diff = abs(train_val - val_val)
        print(f"{key:<20} {train_val:<12.6f} {val_val:<12.6f} {diff:<12.6f}")

    # 重点关注标准化后的加速度分布
    if train_stats['acc_norm_mean'] != 0:
        print(f"\n🔍 Normalized Acceleration Analysis:")
        print(f"   - Train norm acc mean: {train_stats['acc_norm_mean']:.6f} (should be ~0)")
        print(f"   - Train norm acc std:  {train_stats['acc_norm_std']:.6f} (should be ~1)")
        print(f"   - Val norm acc mean:   {val_stats['acc_norm_mean']:.6f} (should be ~0)")
        print(f"   - Val norm acc std:    {val_stats['acc_norm_std']:.6f} (should be ~1)")

    total_diff = sum([abs(train_stats[key] - val_stats[key]) for key in ['acc_mean', 'ori_mean', 'pos_mean']])
    print(f"\nTotal Difference Score: {total_diff:.6f} (lower is better)")

    if total_diff < 0.1:
        print("✓ Data distributions appear consistent!")
    elif total_diff < 0.5:
        print("⚠ Data distributions have minor differences")
    else:
        print("❌ Data distributions have significant differences")

    return train_stats, val_stats


def rotation_matrix_loss_stable(pred_6d, target_6d):
    """数值稳定的旋转矩阵损失"""
    try:
        batch_size, seq_len, joints, _ = pred_6d.shape
        pred_6d_flat = pred_6d.reshape(-1, 6)
        target_6d_flat = target_6d.reshape(-1, 6)

        # 检查输入有效性
        if torch.isnan(pred_6d_flat).any() or torch.isnan(target_6d_flat).any():
            print("rotation is nan")
            return torch.tensor(10.0, device=pred_6d.device, requires_grad=True)

        pred_rotmat = r6d_to_rotation_matrix(pred_6d_flat)
        target_rotmat = r6d_to_rotation_matrix(target_6d_flat)

        # 检查旋转矩阵有效性
        if torch.isnan(pred_rotmat).any() or torch.isnan(target_rotmat).any():
            print("rotation is nan_1")
            return torch.tensor(10.0, device=pred_6d.device, requires_grad=True)

        # 使用更稳定的损失计算
        diff = pred_rotmat - target_rotmat
        loss = torch.mean(torch.norm(diff, dim=(-2, -1)) + 1e-8)

        # 添加数值裁剪
        loss = torch.clamp(loss, 0.0, 100.0)

        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(10.0, device=pred_6d.device, requires_grad=True)

        return loss
    except Exception as e:
        print(f"Error in rotation_matrix_loss: {e}")
        return torch.tensor(10.0, device=pred_6d.device, requires_grad=True)


# ===== 端到端联合训练函数 =====（半精度）
# def train_end_to_end_joint(model1, model2, model3, optimizer, scheduler, train_loader, val_loader,
#                            normalizer, epochs, early_stopper, start_epoch=0):
#     """端到端联合训练所有三个模型（只对加速度数据标准化）"""
#     print("\n====================== Starting End-to-End Joint Training =========================")
#     print(f"🚀 Using Joint E2E Training Mode - All models trained simultaneously")
#     print(f"📊 Checkpoint saving to: {early_stopper.path}")
#     print(f"📏 Data normalization: ONLY Acceleration data")
#
#     criterion = nn.MSELoss(reduction='mean')
#     # scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')  # 修复废弃警告
#     writer = SummaryWriter(os.path.join(LOG_RUN_DIR, 'joint_e2e_logs')) if LOG_ENABLED else None
#
#     # 调整损失权重
#     loss_weights = {
#         'leaf_pos': 0.00,
#         'all_pos': 0.00,
#         'pose_6d': 1
#     }
#
#     print(f"📈 Loss weights: {loss_weights}")
#     print(f"📊 Using acceleration normalization: {normalizer.fitted}")
#
#     for epoch in range(start_epoch, epochs):
#         current_epoch = epoch + 1
#
#         model1.train()
#         model2.train()
#         model3.train()
#
#         train_losses = {'total': [], 'leaf_pos': [], 'all_pos': [], 'pose_6d': []}
#         valid_batches = 0
#         skipped_batches = 0
#
#         epoch_pbar = tqdm(train_loader, desc=f"🔄 Joint E2E Epoch {current_epoch}/{epochs}", leave=True)
#
#         # === 训练循环 ===
#         for batch_idx, data in enumerate(epoch_pbar):
#             try:
#                 acc = data[0].to(DEVICE, non_blocking=True).float()
#                 ori_6d = data[2].to(DEVICE, non_blocking=True).float()  # 不标准化
#                 p_leaf_gt = data[3].to(DEVICE, non_blocking=True).float()  # 不标准化
#                 p_all_gt = data[4].to(DEVICE, non_blocking=True).float()  # 不标准化
#                 pose_6d_gt = data[6].to(DEVICE, non_blocking=True).float()  # 不标准化
#
#                 # 数据质量检查
#                 is_valid, reason = improved_data_check(acc, ori_6d, p_leaf_gt)
#                 if not is_valid:
#                     skipped_batches += 1
#                     if skipped_batches <= 5:  # 只显示前5个警告
#                         print(f"Warning: Skipping batch {batch_idx} - {reason}")
#                     continue
#
#                 # 🔥 只对加速度数据进行标准化
#                 try:
#                     acc_norm = normalizer.normalize_acc(acc)  # 只标准化加速度
#                     # ori_6d, p_leaf_gt, p_all_gt, pose_6d_gt 保持原样
#                 except Exception as e:
#                     print(f"Error in acceleration normalization: {e}")
#                     skipped_batches += 1
#                     continue
#
#                 optimizer.zero_grad(set_to_none=True)
#
#                 with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
#                     # === 使用标准化的加速度数据进行前向传播 ===
#                     # FDIP_1: 预测叶节点位置
#                     input1 = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
#                     p_leaf_pred = model1(input1)
#
#                     # FDIP_2: 预测所有关节位置
#                     zeros = torch.zeros(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 3, device=DEVICE)
#                     p_leaf_with_root = torch.cat(
#                         [zeros, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], -1)], dim=2)
#                     input2 = torch.cat(
#                         [acc_norm, ori_6d,  # 使用标准化的加速度
#                          p_leaf_with_root.view(p_leaf_with_root.shape[0], p_leaf_with_root.shape[1], 6, 3)],
#                         dim=-1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
#                     p_all_pred = model2(input2)
#
#                     # FDIP_3: 预测6D姿态
#                     input_base = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
#                     pose_6d_pred = model3(input_base, p_all_pred)
#
#                     # === 计算损失（使用原始目标数据） ===
#                     # 叶节点位置损失 - 使用原始目标
#                     loss_leaf = criterion(p_leaf_pred, p_leaf_gt.view(-1, p_leaf_gt.shape[1], 15))
#
#                     # 所有关节位置损失 - 使用原始目标
#                     p_all_target = torch.cat([torch.zeros_like(p_all_gt[:, :, 0:1, :]), p_all_gt], dim=2).view(
#                         p_all_gt.shape[0], p_all_gt.shape[1], -1)
#                     loss_all_pos = criterion(p_all_pred, p_all_target)
#
#                     # 6D姿态损失 - 使用原始目标
#                     batch_size, seq_len = pose_6d_pred.shape[:2]
#                     pose_pred_reshaped = pose_6d_pred.view(batch_size, seq_len, 24, 6)
#                     loss_pose = rotation_matrix_loss_stable(pose_pred_reshaped, pose_6d_gt)
#
#                     # 加权总损失
#                     total_loss = (loss_weights['leaf_pos'] * loss_leaf +
#                                   loss_weights['all_pos'] * loss_all_pos +
#                                   loss_weights['pose_6d'] * loss_pose)
#
#                 # 损失有效性检查
#                 if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1000.0:
#                     print(f"Warning: NaN or inf in total_loss: {total_loss}")
#                     skipped_batches += 1
#                     continue
#
#                 scaler.scale(total_loss).backward()
#
#                 # 梯度裁剪
#                 scaler.unscale_(optimizer)
#                 grad_norm = torch.nn.utils.clip_grad_norm_(
#                     [p for model in [model1, model2, model3] for p in model.parameters()],
#                     max_norm=1.0
#                 )
#
#                 if torch.isnan(grad_norm):
#                     print('Warning: NaN gradient norm.')
#                     skipped_batches += 1
#                     continue
#
#                 scaler.step(optimizer)
#                 scaler.update()
#
#                 # 记录损失
#                 train_losses['total'].append(total_loss.item())
#                 train_losses['leaf_pos'].append(loss_leaf.item())
#                 train_losses['all_pos'].append(loss_all_pos.item())
#                 train_losses['pose_6d'].append(loss_pose.item())
#
#                 valid_batches += 1
#
#                 # 更新进度条
#                 if valid_batches % 5 == 0:
#                     epoch_pbar.set_postfix({
#                         'total': f"{total_loss.item():.4f}",
#                         'leaf': f"{loss_leaf.item():.4f}",
#                         'pos': f"{loss_all_pos.item():.4f}",
#                         'pose': f"{loss_pose.item():.4f}",
#                         'valid': f"{valid_batches}",
#                         'skip': f"{skipped_batches}"
#                     })
#
#             except Exception as e:
#                 print(f"Error in training batch {batch_idx}: {e}")
#                 skipped_batches += 1
#                 continue
#
#         # 检查有效batch数量
#         if valid_batches == 0:
#             print(f"❌ No valid training batches in epoch {current_epoch}!")
#             continue
#
#         print(f"📊 Epoch {current_epoch}: Valid batches: {valid_batches}, Skipped: {skipped_batches}")
#
#         # === 验证阶段 ===
#         model1.eval()
#         model2.eval()
#         model3.eval()
#         val_losses = {'total': [], 'leaf_pos': [], 'all_pos': [], 'pose_6d': []}
#         valid_val_batches = 0
#
#         with torch.no_grad():
#             for data_val in val_loader:
#                 try:
#                     acc_val = data_val[0].to(DEVICE, non_blocking=True).float()
#                     ori_val = data_val[2].to(DEVICE, non_blocking=True).float()
#                     p_leaf_gt_val = data_val[3].to(DEVICE, non_blocking=True).float()
#                     p_all_gt_val = data_val[4].to(DEVICE, non_blocking=True).float()
#                     pose_6d_gt_val = data_val[6].to(DEVICE, non_blocking=True).float()
#
#                     # 验证数据质量检查
#                     is_valid, _ = improved_data_check(acc_val, ori_val, p_leaf_gt_val)
#                     if not is_valid:
#                         print("val is_valid")
#                         continue
#
#                     # 验证数据标准化 - 只对加速度
#                     acc_val_norm = normalizer.normalize_acc(acc_val)
#
#                     # 验证前向传播
#                     input1_val = torch.cat((acc_val_norm, ori_val), -1).view(acc_val_norm.shape[0], acc_val_norm.shape[1],
#                                                                              -1)
#                     p_leaf_pred_val = model1(input1_val)
#
#                     zeros_val = torch.zeros(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], 3, device=DEVICE)
#                     p_leaf_with_root_val = torch.cat(
#                         [zeros_val, p_leaf_pred_val.view(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], -1)], dim=2)
#                     input2_val = torch.cat(
#                         [acc_val_norm, ori_val,
#                          p_leaf_with_root_val.view(p_leaf_with_root_val.shape[0], p_leaf_with_root_val.shape[1], 6, 3)],
#                         dim=-1).view(acc_val_norm.shape[0], acc_val_norm.shape[1], -1)
#                     p_all_pred_val = model2(input2_val)
#
#                     input_base_val = torch.cat((acc_val_norm, ori_val), -1).view(acc_val_norm.shape[0],
#                                                                                  acc_val_norm.shape[1], -1)
#                     pose_6d_pred_val = model3(input_base_val, p_all_pred_val)
#
#                     # 验证损失计算 - 使用原始目标
#                     loss_leaf_val = criterion(p_leaf_pred_val, p_leaf_gt_val.view(-1, p_leaf_gt_val.shape[1], 15))
#
#                     p_all_target_val = torch.cat([torch.zeros_like(p_all_gt_val[:, :, 0:1, :]), p_all_gt_val], dim=2).view(
#                         p_all_gt_val.shape[0], p_all_gt_val.shape[1], -1)
#                     loss_all_pos_val = criterion(p_all_pred_val, p_all_target_val)
#
#                     batch_size_val, seq_len_val = pose_6d_pred_val.shape[:2]
#                     pose_pred_reshaped_val = pose_6d_pred_val.view(batch_size_val, seq_len_val, 24, 6)
#                     loss_pose_val = rotation_matrix_loss_stable(pose_pred_reshaped_val, pose_6d_gt_val)
#
#                     total_loss_val = (loss_weights['leaf_pos'] * loss_leaf_val +
#                                       loss_weights['all_pos'] * loss_all_pos_val +
#                                       loss_weights['pose_6d'] * loss_pose_val)
#
#                     if not torch.isnan(total_loss_val) and not torch.isinf(
#                             total_loss_val) and total_loss_val.item() < 1000.0:
#                         val_losses['total'].append(total_loss_val.item())
#                         val_losses['leaf_pos'].append(loss_leaf_val.item())
#                         val_losses['all_pos'].append(loss_all_pos_val.item())
#                         val_losses['pose_6d'].append(loss_pose_val.item())
#                         valid_val_batches += 1
#
#                 except Exception as e:
#                     continue
#
#         print(f"Valid validation batches: {valid_val_batches}/{len(val_loader)}")
#
#         # 计算平均损失
#         avg_train_losses = {k: np.mean(v) if v else float('inf') for k, v in train_losses.items()}
#         avg_val_losses = {k: np.mean(v) if v else float('inf') for k, v in val_losses.items()}
#         current_lr = optimizer.param_groups[0]['lr']
#
#         # 打印训练结果
#         print(f'\n🔄 Joint E2E Epoch {current_epoch}/{epochs} | LR: {current_lr:.6f}')
#         print(f'  📊 Valid batches: Train={valid_batches}, Val={valid_val_batches}')
#         print(
#             f'  📈 Train - Total: {avg_train_losses["total"]:.6f}, Leaf: {avg_train_losses["leaf_pos"]:.6f}, Pos: {avg_train_losses["all_pos"]:.6f}, Pose: {avg_train_losses["pose_6d"]:.6f}')
#         print(
#             f'  📉 Val   - Total: {avg_val_losses["total"]:.6f}, Leaf: {avg_val_losses["leaf_pos"]:.6f}, Pos: {avg_val_losses["all_pos"]:.6f}, Pose: {avg_val_losses["pose_6d"]:.6f}')
#
#         # 计算损失比率
#         loss_ratio = 0.0
#         if avg_train_losses["total"] > 0 and avg_val_losses["total"] < float('inf'):
#             loss_ratio = avg_val_losses["total"] / avg_train_losses["total"]
#             print(f'  📊 Loss Ratio (Val/Train): {loss_ratio:.3f}')
#
#         # 记录到TensorBoard
#         if LOG_ENABLED and writer:
#             for loss_type in train_losses.keys():
#                 if avg_train_losses[loss_type] < float('inf') and avg_val_losses[loss_type] < float('inf'):
#                     writer.add_scalars(f'joint_e2e_loss/{loss_type}', {
#                         'train': avg_train_losses[loss_type],
#                         'val': avg_val_losses[loss_type]
#                     }, current_epoch)
#             writer.add_scalar('joint_e2e_learning_rate', current_lr, current_epoch)
#             writer.add_scalar('joint_e2e_loss_ratio', loss_ratio, current_epoch)
#             writer.add_scalar('joint_e2e_valid_batches', valid_batches, current_epoch)
#
#         # 学习率调度
#         scheduler.step(avg_val_losses['total'])
#
#         # 早停检查
#         if avg_val_losses['total'] < float('inf') and valid_val_batches > 0:
#             early_stopper(avg_val_losses['total'], [model1, model2, model3], optimizer, current_epoch, normalizer)
#             if early_stopper.early_stop:
#                 print(f"🛑 Early stopping triggered at epoch {current_epoch} for Joint E2E Training.")
#                 break
#         else:
#             print(f"⚠️ No valid validation batches in epoch {current_epoch}, skipping early stopping check")
#
#         # 内存清理
#         torch.cuda.empty_cache()
#
#         # 训练完成，加载最佳模型
#     if os.path.exists(early_stopper.path):
#         print(f"✅ Loading best joint E2E model from epoch {early_stopper.best_epoch}")
#         checkpoint = torch.load(early_stopper.path, map_location=DEVICE, weights_only=False)
#         model1.load_state_dict(checkpoint['model1_state_dict'])
#         model2.load_state_dict(checkpoint['model2_state_dict'])
#         model3.load_state_dict(checkpoint['model3_state_dict'])
#         print(f"✅ Successfully loaded best models with validation loss: {early_stopper.val_loss_min:.6f}")
#         del checkpoint
#
#         # 清理资源
#     if writer:
#         writer.close()
#     del criterion, scaler
#     torch.cuda.empty_cache()
#
#     print("======================== End-to-End Joint Training Finished =======================================")
#     return model1, model2, model3

def train_end_to_end_joint(model1, model2, model3, optimizer, scheduler, train_loader, val_loader,
                           normalizer, epochs, early_stopper, start_epoch=0):
    """端到端联合训练所有三个模型（全精度训练，只对加速度数据标准化）"""
    print("\n====================== Starting End-to-End Joint Training (Full Precision) =========================")
    print(f"🚀 Using Joint E2E Training Mode - All models trained simultaneously")
    print(f"📊 Checkpoint saving to: {early_stopper.path}")
    print(f"📏 Data normalization: ONLY Acceleration data")
    print(f"⚡ Training mode: Full Precision (FP32)")

    criterion = nn.MSELoss(reduction='mean')
    # 移除 GradScaler - 使用全精度训练
    writer = SummaryWriter(os.path.join(LOG_RUN_DIR, 'joint_e2e_logs')) if LOG_ENABLED else None

    # 调整损失权重
    loss_weights = {
        'leaf_pos': 0.00,
        'all_pos': 0.00,
        'pose_6d': 1
    }

    print(f"📈 Loss weights: {loss_weights}")
    print(f"📊 Using acceleration normalization: {normalizer.fitted}")

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1

        model1.train()
        model2.train()
        model3.train()

        train_losses = {'total': [], 'leaf_pos': [], 'all_pos': [], 'pose_6d': []}
        valid_batches = 0
        skipped_batches = 0

        epoch_pbar = tqdm(train_loader, desc=f"🔄 Joint E2E Epoch {current_epoch}/{epochs} (FP32)", leave=True)

        # === 训练循环 ===
        for batch_idx, data in enumerate(epoch_pbar):
            try:
                acc = data[0].to(DEVICE, non_blocking=True).float()
                ori_6d = data[2].to(DEVICE, non_blocking=True).float()  # 不标准化
                p_leaf_gt = data[3].to(DEVICE, non_blocking=True).float()  # 不标准化
                p_all_gt = data[4].to(DEVICE, non_blocking=True).float()  # 不标准化
                pose_6d_gt = data[6].to(DEVICE, non_blocking=True).float()  # 不标准化

                # 数据质量检查
                is_valid, reason = improved_data_check(acc, ori_6d, p_leaf_gt)
                if not is_valid:
                    skipped_batches += 1
                    if skipped_batches <= 5:  # 只显示前5个警告
                        print(f"Warning: Skipping batch {batch_idx} - {reason}")
                    continue

                # 🔥 只对加速度数据进行标准化
                try:
                    acc_norm = normalizer.normalize_acc(acc)  # 只标准化加速度
                    # ori_6d, p_leaf_gt, p_all_gt, pose_6d_gt 保持原样
                except Exception as e:
                    print(f"Error in acceleration normalization: {e}")
                    skipped_batches += 1
                    continue

                optimizer.zero_grad(set_to_none=True)

                # === 移除 autocast，使用全精度前向传播 ===
                # FDIP_1: 预测叶节点位置
                input1 = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                p_leaf_pred = model1(input1)

                # FDIP_2: 预测所有关节位置
                zeros = torch.zeros(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 3, device=DEVICE)
                p_leaf_with_root = torch.cat(
                    [zeros, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], -1)], dim=2)
                input2 = torch.cat(
                    [acc_norm, ori_6d,  # 使用标准化的加速度
                     p_leaf_with_root.view(p_leaf_with_root.shape[0], p_leaf_with_root.shape[1], 6, 3)],
                    dim=-1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                p_all_pred = model2(input2)

                # FDIP_3: 预测6D姿态
                input_base = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                pose_6d_pred = model3(input_base, p_all_pred)

                # === 计算损失（使用原始目标数据） ===
                # 叶节点位置损失 - 使用原始目标
                loss_leaf = criterion(p_leaf_pred, p_leaf_gt.view(-1, p_leaf_gt.shape[1], 15))

                # 所有关节位置损失 - 使用原始目标
                p_all_target = torch.cat([torch.zeros_like(p_all_gt[:, :, 0:1, :]), p_all_gt], dim=2).view(
                    p_all_gt.shape[0], p_all_gt.shape[1], -1)
                loss_all_pos = criterion(p_all_pred, p_all_target)

                # 6D姿态损失 - 使用原始目标
                batch_size, seq_len = pose_6d_pred.shape[:2]
                pose_pred_reshaped = pose_6d_pred.view(batch_size, seq_len, 24, 6)
                loss_pose = rotation_matrix_loss_stable(pose_pred_reshaped, pose_6d_gt)

                # 加权总损失
                total_loss = (loss_weights['leaf_pos'] * loss_leaf +
                              loss_weights['all_pos'] * loss_all_pos +
                              loss_weights['pose_6d'] * loss_pose)

                # 损失有效性检查
                if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1000.0:
                    print(f"Warning: NaN or inf in total_loss: {total_loss}")
                    skipped_batches += 1
                    continue

                # === 全精度反向传播 - 移除scaler ===
                total_loss.backward()

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for model in [model1, model2, model3] for p in model.parameters()],
                    max_norm=100.0
                )

                if torch.isnan(grad_norm):
                    print('Warning: NaN gradient norm.')
                    skipped_batches += 1
                    continue

                # === 全精度优化器更新 - 移除scaler ===
                optimizer.step()

                # 记录损失
                train_losses['total'].append(total_loss.item())
                train_losses['leaf_pos'].append(loss_leaf.item())
                train_losses['all_pos'].append(loss_all_pos.item())
                train_losses['pose_6d'].append(loss_pose.item())

                valid_batches += 1

                # 更新进度条
                if valid_batches % 5 == 0:
                    epoch_pbar.set_postfix({
                        'total': f"{total_loss.item():.4f}",
                        'leaf': f"{loss_leaf.item():.4f}",
                        'pos': f"{loss_all_pos.item():.4f}",
                        'pose': f"{loss_pose.item():.4f}",
                        'valid': f"{valid_batches}",
                        'skip': f"{skipped_batches}",
                        'mode': 'FP32'
                    })

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                skipped_batches += 1
                continue

        # 检查有效batch数量
        if valid_batches == 0:
            print(f"❌ No valid training batches in epoch {current_epoch}!")
            continue

        print(f"📊 Epoch {current_epoch} (FP32): Valid batches: {valid_batches}, Skipped: {skipped_batches}")

        # === 验证阶段 ===
        model1.eval()
        model2.eval()
        model3.eval()
        val_losses = {'total': [], 'leaf_pos': [], 'all_pos': [], 'pose_6d': []}
        valid_val_batches = 0

        with torch.no_grad():
            for data_val in val_loader:
                try:
                    acc_val = data_val[0].to(DEVICE, non_blocking=True).float()
                    ori_val = data_val[2].to(DEVICE, non_blocking=True).float()
                    p_leaf_gt_val = data_val[3].to(DEVICE, non_blocking=True).float()
                    p_all_gt_val = data_val[4].to(DEVICE, non_blocking=True).float()
                    pose_6d_gt_val = data_val[6].to(DEVICE, non_blocking=True).float()

                    # 验证数据质量检查
                    is_valid, _ = improved_data_check(acc_val, ori_val, p_leaf_gt_val)
                    if not is_valid:
                        print("val is_valid")
                        continue

                    # 验证数据标准化 - 只对加速度
                    acc_val_norm = normalizer.normalize_acc(acc_val)

                    # === 全精度验证前向传播 ===
                    input1_val = torch.cat((acc_val_norm, ori_val), -1).view(acc_val_norm.shape[0], acc_val_norm.shape[1], -1)
                    p_leaf_pred_val = model1(input1_val)

                    zeros_val = torch.zeros(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], 3, device=DEVICE)
                    p_leaf_with_root_val = torch.cat(
                        [zeros_val, p_leaf_pred_val.view(p_leaf_pred_val.shape[0], p_leaf_pred_val.shape[1], -1)], dim=2)
                    input2_val = torch.cat(
                        [acc_val_norm, ori_val,
                         p_leaf_with_root_val.view(p_leaf_with_root_val.shape[0], p_leaf_with_root_val.shape[1], 6, 3)],
                        dim=-1).view(acc_val_norm.shape[0], acc_val_norm.shape[1], -1)
                    p_all_pred_val = model2(input2_val)

                    input_base_val = torch.cat((acc_val_norm, ori_val), -1).view(acc_val_norm.shape[0],
                                                                                 acc_val_norm.shape[1], -1)
                    pose_6d_pred_val = model3(input_base_val, p_all_pred_val)

                    # 验证损失计算 - 使用原始目标
                    loss_leaf_val = criterion(p_leaf_pred_val, p_leaf_gt_val.view(-1, p_leaf_gt_val.shape[1], 15))

                    p_all_target_val = torch.cat([torch.zeros_like(p_all_gt_val[:, :, 0:1, :]), p_all_gt_val], dim=2).view(
                        p_all_gt_val.shape[0], p_all_gt_val.shape[1], -1)
                    loss_all_pos_val = criterion(p_all_pred_val, p_all_target_val)

                    batch_size_val, seq_len_val = pose_6d_pred_val.shape[:2]
                    pose_pred_reshaped_val = pose_6d_pred_val.view(batch_size_val, seq_len_val, 24, 6)
                    loss_pose_val = rotation_matrix_loss_stable(pose_pred_reshaped_val, pose_6d_gt_val)

                    total_loss_val = (loss_weights['leaf_pos'] * loss_leaf_val +
                                      loss_weights['all_pos'] * loss_all_pos_val +
                                      loss_weights['pose_6d'] * loss_pose_val)

                    if not torch.isnan(total_loss_val) and not torch.isinf(
                            total_loss_val) and total_loss_val.item() < 1000.0:
                        val_losses['total'].append(total_loss_val.item())
                        val_losses['leaf_pos'].append(loss_leaf_val.item())
                        val_losses['all_pos'].append(loss_all_pos_val.item())
                        val_losses['pose_6d'].append(loss_pose_val.item())
                        valid_val_batches += 1

                except Exception as e:
                    continue

        print(f"Valid validation batches: {valid_val_batches}/{len(val_loader)}")

        # 计算平均损失
        avg_train_losses = {k: np.mean(v) if v else float('inf') for k, v in train_losses.items()}
        avg_val_losses = {k: np.mean(v) if v else float('inf') for k, v in val_losses.items()}
        current_lr = optimizer.param_groups[0]['lr']

        # 打印训练结果
        print(f'\n🔄 Joint E2E Epoch {current_epoch}/{epochs} (FP32) | LR: {current_lr:.6f}')
        print(f'  📊 Valid batches: Train={valid_batches}, Val={valid_val_batches}')
        print(
            f'  📈 Train - Total: {avg_train_losses["total"]:.6f}, Leaf: {avg_train_losses["leaf_pos"]:.6f}, Pos: {avg_train_losses["all_pos"]:.6f}, Pose: {avg_train_losses["pose_6d"]:.6f}')
        print(
            f'  📉 Val   - Total: {avg_val_losses["total"]:.6f}, Leaf: {avg_val_losses["leaf_pos"]:.6f}, Pos: {avg_val_losses["all_pos"]:.6f}, Pose: {avg_val_losses["pose_6d"]:.6f}')

        # 计算损失比率
        loss_ratio = 0.0
        if avg_train_losses["total"] > 0 and avg_val_losses["total"] < float('inf'):
            loss_ratio = avg_val_losses["total"] / avg_train_losses["total"]
            print(f'  📊 Loss Ratio (Val/Train): {loss_ratio:.3f}')

        # 记录到TensorBoard
        if LOG_ENABLED and writer:
            for loss_type in train_losses.keys():
                if avg_train_losses[loss_type] < float('inf') and avg_val_losses[loss_type] < float('inf'):
                    writer.add_scalars(f'joint_e2e_loss/{loss_type}', {
                        'train': avg_train_losses[loss_type],
                        'val': avg_val_losses[loss_type]
                    }, current_epoch)
            writer.add_scalar('joint_e2e_learning_rate', current_lr, current_epoch)
            writer.add_scalar('joint_e2e_loss_ratio', loss_ratio, current_epoch)
            writer.add_scalar('joint_e2e_valid_batches', valid_batches, current_epoch)

        # 学习率调度
        scheduler.step(avg_val_losses['total'])

        # 早停检查
        if avg_val_losses['total'] < float('inf') and valid_val_batches > 0:
            early_stopper(avg_val_losses['total'], [model1, model2, model3], optimizer, current_epoch, normalizer)
            if early_stopper.early_stop:
                print(f"🛑 Early stopping triggered at epoch {current_epoch} for Joint E2E Training (FP32).")
                break
        else:
            print(f"⚠️ No valid validation batches in epoch {current_epoch}, skipping early stopping check")

        # 内存清理
        torch.cuda.empty_cache()

    # 训练完成，加载最佳模型
    if os.path.exists(early_stopper.path):
        print(f"✅ Loading best joint E2E model (FP32) from epoch {early_stopper.best_epoch}")
        checkpoint = torch.load(early_stopper.path, map_location=DEVICE, weights_only=False)
        model1.load_state_dict(checkpoint['model1_state_dict'])
        model2.load_state_dict(checkpoint['model2_state_dict'])
        model3.load_state_dict(checkpoint['model3_state_dict'])
        print(f"✅ Successfully loaded best models with validation loss: {early_stopper.val_loss_min:.6f}")
        del checkpoint

    # 清理资源
    if writer:
        writer.close()
    del criterion
    torch.cuda.empty_cache()

    print("======================== End-to-End Joint Training (Full Precision) Finished =======================================")
    return model1, model2, model3





# ===== 评估函数 =====
def evaluate_pipeline(model1, model2, model3, data_loader, normalizer):
    """评估流水线（使用加速度标准化）"""
    print("\n============================ Evaluating Complete Pipeline ======================================")
    print(f"📊 Evaluation for {TRAINING_MODE} training mode with acceleration normalization")

    clear_memory()

    # 评估目录
    eval_results_dir = os.path.join("GGIP", f"evaluate_{TRAINING_MODE}_pipeline_{TIMESTAMP}")
    eval_plots_dir = os.path.join(eval_results_dir, "plots")
    eval_data_dir = os.path.join(eval_results_dir, "data")

    # 创建目录
    os.makedirs(eval_results_dir, exist_ok=True)
    os.makedirs(eval_plots_dir, exist_ok=True)
    os.makedirs(eval_data_dir, exist_ok=True)

    try:
        # 检查SMPL模型文件
        smpl_path = r"F:\FDIP\basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
        if not os.path.exists(smpl_path):
            print(f"❌ SMPL model file not found: {smpl_path}")
            print("🔧 Please download SMPL model and place it at the correct path")

            # 运行简化评估
            print("🔄 Running simplified evaluation without SMPL...")
            simplified_evaluation(model1, model2, model3, data_loader, normalizer, eval_data_dir)
            return

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

        print(f"Running {TRAINING_MODE} model evaluation with acceleration normalization...")
        valid_eval_batches = 0

        with torch.no_grad():
            for data_val in tqdm(data_loader, desc=f"Evaluating {TRAINING_MODE.upper()} Pipeline"):
                try:
                    acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                               (data_val[0], data_val[2], data_val[6])]

                    # 数据有效性检查
                    is_valid, _ = improved_data_check(acc, ori_6d)
                    if not is_valid:
                        print("evaluate is_valid")
                        continue

                    # 🔥 只对加速度进行标准化
                    acc_norm = normalizer.normalize_acc(acc)

                    # 级联推理
                    input1 = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                    p_leaf_logits = model1(input1)

                    zeros1 = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
                    p_leaf_pred = torch.cat(
                        [zeros1, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], -1)], dim=2)

                    input2 = torch.cat(
                        [acc_norm, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 6, 3)],
                        dim=-1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                    p_all_pos_flattened = model2(input2)

                    input_base = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                    pose_pred_flat = model3(input_base, p_all_pos_flattened)

                    batch_size, seq_len = pose_pred_flat.shape[:2]
                    pose_pred = pose_pred_flat.view(batch_size, seq_len, 24, 6)

                    errs_dict = evaluator.eval(pose_pred, pose_6d_gt)

                    for key in all_errors.keys():
                        if key in errs_dict and errs_dict[key].numel() > 0:
                            all_errors[key].append(errs_dict[key].flatten().cpu())

                    valid_eval_batches += 1

                except Exception as e:
                    print(f"Warning: Error processing batch in evaluation: {e}")
                    continue

        print(f"Processed {valid_eval_batches} valid evaluation batches")

        # 评估结果处理
        clear_memory()

        if all_errors["mesh_err"] and valid_eval_batches > 0:
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

            # 保存评估结果
            save_evaluation_results(final_errors, avg_errors, eval_data_dir)

        else:
            print("❌ No evaluation results were generated or no valid batches processed.")

    except Exception as e:
        print(f"Critical error in evaluation pipeline: {e}")
        print("🔄 Attempting simplified evaluation...")
        simplified_evaluation(model1, model2, model3, data_loader, normalizer, eval_data_dir)

    print(f"\n✅ {TRAINING_MODE.upper()} evaluation completed. Results saved in: {eval_results_dir}")


def simplified_evaluation(model1, model2, model3, data_loader, normalizer, eval_data_dir):
    """简化的评估函数，不依赖SMPL模型"""
    print("Running simplified evaluation (MSE-based metrics)...")

    model1.eval()
    model2.eval()
    model3.eval()

    mse_losses = []
    valid_batches = 0

    with torch.no_grad():
        for data_val in tqdm(data_loader, desc="Simplified Evaluation"):
            try:
                acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                           (data_val[0], data_val[2], data_val[6])]

                # 数据检查
                is_valid, _ = improved_data_check(acc, ori_6d)
                if not is_valid:
                    print("simple_evaluate is_valid")
                    continue

                # 加速度标准化
                acc_norm = normalizer.normalize_acc(acc)

                # 模型推理
                input1 = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                p_leaf_logits = model1(input1)

                zeros1 = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
                p_leaf_pred = torch.cat(
                    [zeros1, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], -1)], dim=2)

                input2 = torch.cat(
                    [acc_norm, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 6, 3)],
                    dim=-1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                p_all_pos_flattened = model2(input2)

                input_base = torch.cat((acc_norm, ori_6d), -1).view(acc_norm.shape[0], acc_norm.shape[1], -1)
                pose_pred_flat = model3(input_base, p_all_pos_flattened)

                # 计算MSE损失
                mse_loss = nn.MSELoss()(pose_pred_flat, pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], -1))
                mse_losses.append(mse_loss.item())
                valid_batches += 1

            except Exception as e:
                continue

    if mse_losses:
        avg_mse = np.mean(mse_losses)
        std_mse = np.std(mse_losses)

        print(f"\n🎯 Simplified Evaluation Results (Acceleration Normalized):")
        print(f"  - Average MSE Loss:           {avg_mse:.6f}")
        print(f"  - Standard Deviation:         {std_mse:.6f}")
        print(f"  - Valid Batches Processed:    {valid_batches}")

        # 保存简化评估结果
        results = {
            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'valid_batches': valid_batches,
            'total_samples': len(mse_losses),
            'normalization_type': 'acceleration_only'
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(eval_data_dir, f"simplified_eval_acc_norm_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  - Results saved to: {results_path}")
    else:
        print("❌ No valid evaluation results generated in simplified evaluation")


def save_evaluation_results(final_errors, avg_errors, eval_data_dir):
    """保存评估结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # 1. 保存原始误差数据
        raw_data_path = os.path.join(eval_data_dir, f"{TRAINING_MODE}_raw_errors_acc_norm_{timestamp}.pkl")
        with open(raw_data_path, 'wb') as f:
            pickle.dump(final_errors, f)
        print(f"Raw error data saved to: {raw_data_path}")

        # 2. 保存统计结果
        stats_data = {
            "timestamp": timestamp,
            "training_mode": TRAINING_MODE,
            "normalization_type": "acceleration_only",
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

        stats_path = os.path.join(eval_data_dir, f"{TRAINING_MODE}_evaluation_stats_acc_norm_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"Statistics saved to: {stats_path}")

        # 3. 保存为CSV格式
        csv_data = []
        for key, values in final_errors.items():
            for value in values.numpy():
                csv_data.append({
                    'training_mode': TRAINING_MODE,
                    'normalization_type': 'acceleration_only',
                    'metric': key,
                    'value': value,
                    'timestamp': timestamp
                })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(eval_data_dir, f"{TRAINING_MODE}_evaluation_data_acc_norm_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            print(f"CSV data saved to: {csv_path}")

    except Exception as e:
        print(f"Warning: Error saving data files: {e}")


# ===== 主训练函数 =====
def main():
    """改进的主函数，支持加速度数据标准化的端到端联合训练"""
    args = parse_args()

    global BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    MAX_EPOCHS = args.max_epochs
    PATIENCE = args.patience

    set_seed(SEED)
    print("==================== Starting Improved Training Pipeline =====================")
    print(f"Training mode: {'Joint End-to-End' if args.use_joint_training else 'Sequential Stages'}")
    print(f"Residual connections: {'Enabled' if args.use_residual else 'Disabled'}")
    print(f"Data normalization: ONLY Acceleration data")
    print(f"Reset best on resume: {'Enabled' if args.reset_best_on_resume else 'Disabled'}")

    setup_directories_and_paths(args)
    create_directories()

    # 数据加载
    try:
        train_loader, val_loader, normalizer = load_data_unified_split(
            train_percent=0.8,
            val_percent=0.2,
            seed=SEED
        )
        print("Using unified dataset with consistent split and acceleration normalization!")
        check_data_distribution(train_loader, val_loader, normalizer)

        # 保存标准化参数
        norm_path = os.path.join(CHECKPOINT_DIR, 'acceleration_normalizer.pth')
        normalizer.save(norm_path)
        print(f"Acceleration normalization parameters saved to: {norm_path}")

    except Exception as e:
        print(f"Failed to load unified datasets: {e}")
        sys.exit(1)

    total_start_time = time.time()

    if args.use_joint_training:
        print(f"\n=== Using End-to-End Joint Training Mode ===")

        # 初始化模型
        model1 = FDIP_1(input_dim=6 * 9, output_dim=5 * 3).to(DEVICE)

        if args.use_residual:
            model2 = FDIP_2_Residual(input_dim=6 * 12, output_dim=24 * 3).to(DEVICE)
            model3 = FDIP_3_Residual(input_dim=6 * 9, output_dim=24 * 6).to(DEVICE)
            print("Using residual-enhanced models")
        else:
            model2 = FDIP_2(input_dim=6 * 12, output_dim=24 * 3).to(DEVICE)
            model3 = FDIP_3(input_dim=6 * 9, output_dim=24 * 6).to(DEVICE)
            print("Using original models")

        # 检查点路径
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'joint_e2e_training', 'best_joint_e2e_model.pth')
        completion_marker = os.path.join(CHECKPOINT_DIR, 'joint_e2e_training', 'joint_e2e_completed.marker')

        # 检查是否已完成训练
        if os.path.exists(completion_marker) and not args.resume:
            print("Joint E2E training already completed. Loading best models and skipping training.")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                model1.load_state_dict(checkpoint['model1_state_dict'])
                model2.load_state_dict(checkpoint['model2_state_dict'])
                model3.load_state_dict(checkpoint['model3_state_dict'])
                print(f"Successfully loaded best joint E2E models from {checkpoint_path}")
                del checkpoint

                # 加载标准化器
                norm_checkpoint_path = checkpoint_path.replace('.pth', '_normalizer.pth')
                if os.path.exists(norm_checkpoint_path):
                    normalizer.load(norm_checkpoint_path)
            else:
                print(f"Error: Completion marker found, but checkpoint file {checkpoint_path} is missing!")
                sys.exit(1)
        else:
            # 设置联合优化器 - 改进的学习率策略
            param_groups = [
                {'params': model1.parameters(), 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
                {'params': model2.parameters(), 'lr': LEARNING_RATE * 0.7, 'weight_decay': WEIGHT_DECAY},
                {'params': model3.parameters(), 'lr': LEARNING_RATE * 0.1, 'weight_decay': WEIGHT_DECAY},  # 最小学习率
            ]
            optimizer = optim.AdamW(param_groups)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.7,
                patience=8,
                min_lr=1e-6,
                threshold=1e-4
            )

            # 创建早停器，传入重置标志
            early_stopper = MultiModelEarlyStopping(
                patience=PATIENCE,
                path=checkpoint_path,
                verbose=True,
                reset_best=args.reset_best_on_resume  # 传入重置标志
            )

            start_epoch = 0
            is_resuming = False

            # 检查是否有断点可以恢复
            if os.path.exists(checkpoint_path):
                print(f"Found joint E2E checkpoint. Resuming training from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                model1.load_state_dict(checkpoint['model1_state_dict'])
                model2.load_state_dict(checkpoint['model2_state_dict'])
                model3.load_state_dict(checkpoint['model3_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']

                # 检查是否需要重置最佳记录
                if args.reset_best_on_resume:
                    print("🔄 Reset best validation loss requested for resume training")
                    early_stopper.reset_best_values(start_epoch)
                else:
                    # 加载之前的早停状态
                    early_stopper.val_loss_min = checkpoint['val_loss_min']
                    early_stopper.best_score = checkpoint['best_score']
                    early_stopper.counter = checkpoint.get('early_stopping_counter', 0)
                    print(f"📊 Loaded previous best validation loss: {early_stopper.val_loss_min:.6f}")

                print(f"Resuming from Epoch {start_epoch + 1}")
                is_resuming = True

                # 加载标准化器
                norm_checkpoint_path = checkpoint_path.replace('.pth', '_normalizer.pth')
                if os.path.exists(norm_checkpoint_path):
                    normalizer.load(norm_checkpoint_path)

                del checkpoint
            else:
                print("No joint E2E checkpoint found. Starting training from scratch.")

            # 如果是续训且设置了重置标志，给出额外提示
            if is_resuming and args.reset_best_on_resume:
                print("⚠️  IMPORTANT: Best validation loss has been reset!")
                print("   The first validation result will be saved as the new best checkpoint.")
                print("   This is useful when loss weights or training configuration have changed.")

            # 执行联合训练
            model1, model2, model3 = train_end_to_end_joint(
                model1, model2, model3, optimizer, scheduler,
                train_loader, val_loader, normalizer,
                MAX_EPOCHS, early_stopper, start_epoch
            )

            # 训练完成标记
            with open(completion_marker, 'w') as f:
                f.write(f"Joint E2E training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Training mode: {TRAINING_MODE}\n")
                f.write(f"Residual connections: {'Enabled' if args.use_residual else 'Disabled'}\n")
                f.write(f"Normalization: Acceleration data only\n")
                f.write(f"Reset best on resume: {'Enabled' if args.reset_best_on_resume else 'Disabled'}\n")
                f.write(
                    f"Best model saved at epoch {early_stopper.best_epoch} with val_loss {early_stopper.val_loss_min:.6f}\n")
            print(f"Joint E2E training marked as completed.")

            # 清理训练对象
            cleanup_training_objects(optimizer, scheduler, early_stopper)

    else:
        print("\nUsing Sequential Stage Training Mode")
        print("Sequential training not implemented in this version. Please use joint training.")
        return

    print(f"\nAll {TRAINING_MODE} training complete!")
    total_end_time = time.time()
    print(f"Total training time: {(total_end_time - total_start_time) / 3600:.2f} hours")

    clear_memory()
    evaluate_pipeline(model1, model2, model3, val_loader, normalizer)
    print(f"\nImproved {TRAINING_MODE} training and evaluation finished successfully!")

if __name__ == '__main__':
    main()



