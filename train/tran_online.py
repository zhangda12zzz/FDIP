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
from data.dataset_posReg import ImuDataset
from model.net_zd import FDIP_1, FDIP_2, FDIP_3
from evaluator import PoseEvaluator

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置使用的GPU
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5  # 新增：权重衰减正则化
BATCH_SIZE = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOG_ENABLED = True
TRAIN_PERCENT = 0.9
BATCH_SIZE_VAL = 32
SEED = 42

# --- Paths ---
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
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class EarlyStopping:
    """
    如果验证损失在给定的耐心期后没有改善，则提前停止训练。
    """

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

    def __call__(self, val_loss, model, optimizer, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """当验证损失减少时保存模型。"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        # 注意：这里我们只保存模型，因为优化器状态对于最佳模型来说不是必需的
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
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
        custom_dataset = ImuDataset(TRAIN_DATA_FOLDERS)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    total_size = len(custom_dataset)
    train_size = int(total_size * train_percent)
    train_indices = range(train_size)
    val_indices = range(train_size, total_size)

    train_dataset = Subset(custom_dataset, train_indices)
    val_dataset = Subset(custom_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=4 if torch.cuda.is_available() else 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False,
        pin_memory=True, num_workers=4 if torch.cuda.is_available() else 0
    )
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    return train_loader, val_loader


# =====================================================================================
# MODIFIED TRAINING FUNCTIONS WITH LR SCHEDULER & WEIGHT DECAY
# =====================================================================================

def train_fdip_1(model, train_loader, val_loader, epochs, early_stopper):
    """训练 FDIP_1 模型，包含早停和学习率调度。"""
    print("\n=============================== Starting FDIP_1 Training =============================")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip1')) if LOG_ENABLED else None

    for epoch in range(epochs):
        current_epoch = epoch + 1
        model.train()
        train_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"FDIP_1 Epoch {current_epoch}/{epochs}", leave=True)
        for data in epoch_pbar:
            acc = data[0].to(DEVICE, non_blocking=True).float()
            ori_6d = data[2].to(DEVICE, non_blocking=True).float()
            p_leaf = data[3].to(DEVICE, non_blocking=True).float()

            x = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            target = p_leaf.view(-1, p_leaf.shape[1], 15)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = torch.sqrt(criterion(logits, target))

            if torch.isnan(loss) or torch.isinf(loss): continue

            scaler.scale(loss).backward()
            # 梯度裁剪 (可选，但推荐)
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'FDIP_1 Epoch {current_epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}')

        if LOG_ENABLED and writer:
            writer.add_scalars('loss/fdip1', {'train': avg_train_loss, 'val': avg_val_loss}, current_epoch)
            writer.add_scalar('learning_rate/fdip1', current_lr, current_epoch)

        # 学习率调度器步进
        scheduler.step()

        # 检查早停
        early_stopper(avg_val_loss, model, optimizer, current_epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered for FDIP_1.")
            break

    print(
        f"FDIP_1 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    model.load_state_dict(torch.load(early_stopper.path))

    if writer: writer.close()
    print("======================== FDIP_1 Training Finished ==========================================")
    return model


def train_fdip_2(model1, model2, train_loader, val_loader, epochs, early_stopper):
    """训练 FDIP_2 模型"""
    print("\n====================== Starting FDIP_2 Training (Online Inference) =========================")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model2.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip2')) if LOG_ENABLED else None

    for epoch in range(epochs):
        current_epoch = epoch + 1
        model1.eval()
        model2.train()
        train_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"FDIP_2 Epoch {current_epoch}/{epochs}", leave=True)
        for data in epoch_pbar:
            acc = data[0].to(DEVICE, non_blocking=True).float()
            ori_6d = data[2].to(DEVICE, non_blocking=True).float()
            p_all = data[4].to(DEVICE, non_blocking=True).float()

            with torch.no_grad():
                input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                p_leaf_logits = model1(input1)
                zeros = torch.zeros(p_leaf_logits.shape[:-1] + (3,), device=DEVICE)
                p_leaf_pred = torch.cat([zeros, p_leaf_logits], dim=-1).view(*p_leaf_logits.shape[:-1], 6, 3)

            x2 = torch.cat((acc, ori_6d, p_leaf_pred), -1).view(acc.shape[0], acc.shape[1], -1)
            target = torch.cat([torch.zeros_like(p_all), p_all], dim=-1).view(p_all.shape[0], p_all.shape[1], -1)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model2(x2)
                loss = torch.sqrt(criterion(logits, target))

            if torch.isnan(loss) or torch.isinf(loss): continue

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
                zeros_val = torch.zeros(p_leaf_logits_val.shape[:-1] + (3,), device=DEVICE)
                p_leaf_pred_val = torch.cat([zeros_val, p_leaf_logits_val], dim=-1).view(*p_leaf_logits_val.shape[:-1],
                                                                                         6, 3)
                x2_val = torch.cat((acc_val, ori_val, p_leaf_pred_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                target_val = torch.cat([torch.zeros_like(p_all_val), p_all_val], dim=-1).view(p_all_val.shape[0],
                                                                                              p_all_val.shape[1], -1)
                logits_val = model2(x2_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val))
                val_losses.append(loss_val.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'FDIP_2 Epoch {current_epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}')

        if LOG_ENABLED and writer:
            writer.add_scalars('loss/fdip2', {'train': avg_train_loss, 'val': avg_val_loss}, current_epoch)
            writer.add_scalar('learning_rate/fdip2', current_lr, current_epoch)

        scheduler.step()
        early_stopper(avg_val_loss, model2, optimizer, current_epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered for FDIP_2.")
            break

    print(
        f"FDIP_2 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    model2.load_state_dict(torch.load(early_stopper.path))

    if writer: writer.close()
    print("=========================== FDIP_2 Training Finished ==================================")
    return model2


def train_fdip_3(model1, model2, model3, train_loader, val_loader, epochs, early_stopper):
    """训练 FDIP_3 模型"""
    print("\n======================== Starting FDIP_3 Training (Online Inference)====================")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model3.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip3')) if LOG_ENABLED else None

    for epoch in range(epochs):
        current_epoch = epoch + 1
        model1.eval();
        model2.eval();
        model3.train()
        train_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"FDIP_3 Epoch {current_epoch}/{epochs}", leave=True)
        for data in epoch_pbar:
            acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in (data[0], data[2], data[6])]

            with torch.no_grad():
                input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                p_leaf_logits = model1(input1)
                zeros = torch.zeros(p_leaf_logits.shape[:-1] + (3,), device=DEVICE)
                p_leaf_pred = torch.cat([zeros, p_leaf_logits], dim=-1).view(*p_leaf_logits.shape[:-1], 6, 3)
                input2 = torch.cat((acc, ori_6d, p_leaf_pred), -1).view(acc.shape[0], acc.shape[1], -1)
                p_all_pos_flattened = model2(input2)

            input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            target = pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], 144)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model3(input_base, p_all_pos_flattened)
                loss = torch.sqrt(criterion(logits, target))

            if torch.isnan(loss) or torch.isinf(loss): continue

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
                zeros_val1 = torch.zeros(p_leaf_logits_val.shape[:-1] + (3,), device=DEVICE)
                p_leaf_pred_val = torch.cat([zeros_val1, p_leaf_logits_val], dim=-1).view(*p_leaf_logits_val.shape[:-1],
                                                                                          6, 3)
                input2_val = torch.cat((acc_val, ori_val, p_leaf_pred_val), -1).view(acc_val.shape[0], acc_val.shape[1],
                                                                                     -1)
                p_all_pos_flattened_val = model2(input2_val)
                input_base_val = torch.cat((acc_val, ori_val), -1).view(acc_val.shape[0], acc_val.shape[1], -1)
                target_val = pose_6d_gt_val.view(pose_6d_gt_val.shape[0], pose_6d_gt_val.shape[1], 144)
                logits_val = model3(input_base_val, p_all_pos_flattened_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val))
                val_losses.append(loss_val.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'FDIP_3 Epoch {current_epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}')

        if LOG_ENABLED and writer:
            writer.add_scalars('loss/fdip3', {'train': avg_train_loss, 'val': avg_val_loss}, current_epoch)
            writer.add_scalar('learning_rate/fdip3', current_lr, current_epoch)

        scheduler.step()
        early_stopper(avg_val_loss, model3, optimizer, current_epoch)
        if early_stopper.early_stop:
            print("Early stopping triggered for FDIP_3.")
            break

    print(
        f"FDIP_3 training finished. Loading best model from epoch {early_stopper.best_epoch} saved at {early_stopper.path}.")
    model3.load_state_dict(torch.load(early_stopper.path))

    if writer: writer.close()
    print("================================ FDIP_3 Training Finished =======================================")
    return model3


def evaluate_pipeline(model1, model2, model3, data_loader):
    """评估完整的模型流水线。"""
    print("\n============================ Evaluating Complete Pipeline ======================================")
    # 假设 PoseEvaluator 已正确定义
    evaluator = PoseEvaluator()
    model1.eval();
    model2.eval();
    model3.eval()
    all_errs_list = []
    with torch.no_grad():
        for data_val in tqdm(data_loader, desc="Evaluating Pipeline"):
            acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                       (data_val[0], data_val[2], data_val[6])]
            input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            p_leaf_logits = model1(input1)
            zeros1 = torch.zeros(p_leaf_logits.shape[:-1] + (3,), device=DEVICE)
            p_leaf_pred = torch.cat([zeros1, p_leaf_logits], dim=-1).view(*p_leaf_logits.shape[:-1], 6, 3)
            input2 = torch.cat((acc, ori_6d, p_leaf_pred), -1).view(acc.shape[0], acc.shape[1], -1)
            p_all_pos_flattened = model2(input2)
            input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
            pose_pred = model3(input_base, p_all_pos_flattened)
            pose_gt = pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], 144)
            errs = evaluator.eval(pose_pred, pose_gt)
            all_errs_list.append(errs.cpu())

    if all_errs_list:
        avg_errs = torch.cat(all_errs_list, dim=1).mean(dim=1)
        print("Complete Pipeline Evaluation Results:")
        print(f"SIP Error (deg): {avg_errs[0, 0].item():.4f}")
        print(f"Angular Error (deg): {avg_errs[1, 0].item():.4f}")
        print(f"Positional Error (cm): {avg_errs[2, 0].item():.4f}")
        print(f"Mesh Error (cm): {avg_errs[3, 0].item():.4f}")
        print(f"Jitter Error (100m/s^3): {avg_errs[4, 0].item():.4f}")
    else:
        print("No evaluation results generated.")


def main():
    """主函数，运行完整的训练和评估流程。"""
    set_seed(SEED)
    print("==================== Starting Full Training Pipeline =====================")
    total_start_time = time.time()

    create_directories()
    train_loader, val_loader = load_data(train_percent=TRAIN_PERCENT)

    model1 = FDIP_1(6 * 9, 5 * 3).to(DEVICE)
    model2 = FDIP_2(6 * 12, 24 * 3).to(DEVICE)
    model3 = FDIP_3(24 * 12, 24 * 6).to(DEVICE)

    patience = 15
    max_epochs = 200  # 设置一个较高的上限，由早停来决定最佳轮数

    early_stopper1 = EarlyStopping(patience=patience,
                                   path=os.path.join(CHECKPOINT_DIR, 'ggip1', 'best_model_fdip1.pth'))
    early_stopper2 = EarlyStopping(patience=patience,
                                   path=os.path.join(CHECKPOINT_DIR, 'ggip2', 'best_model_fdip2.pth'))
    early_stopper3 = EarlyStopping(patience=patience,
                                   path=os.path.join(CHECKPOINT_DIR, 'ggip3', 'best_model_fdip3.pth'))

    # --- 训练阶段 ---
    model1 = train_fdip_1(model1, train_loader, val_loader, epochs=max_epochs, early_stopper=early_stopper1)
    model2 = train_fdip_2(model1, model2, train_loader, val_loader, epochs=max_epochs, early_stopper=early_stopper2)
    model3 = train_fdip_3(model1, model2, model3, train_loader, val_loader, epochs=max_epochs,
                          early_stopper=early_stopper3)

    print("\nAll training stages complete!")
    total_end_time = time.time()
    print(f"Total training time: {(total_end_time - total_start_time) / 3600:.2f} hours")

    # --- 评估阶段 ---
    evaluate_pipeline(model1, model2, model3, val_loader)

    print("\nTraining and evaluation finished successfully!")


if __name__ == '__main__':
    main()
