import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

# import config as conf
from data.dataset_posReg import ImuDataset
from model.net_zd import FDIP_1, FDIP_2, FDIP_3
print(f"Using device")
from evaluator import PoseEvaluator

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set GPU device

# Hyperparameters
LEARNING_RATE = 1e-4
CHECKPOINT_INTERVAL = 50
BATCH_SIZE = 64
BATCH_SIZE_VAL = 16
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
LOG_ENABLED = True
TRAIN_PERCENT = 0.9

# Paths (centralized for easier management)
TRAIN_DATA_FOLDERS = [
    os.path.join("D:\\", "Dataset", "AMASS", "HumanEva", "pt"),
    os.path.join("D:\\", "Dataset", "DIPIMUandOthers", "DIP_6", "Detail")
]
CHECKPOINT_DIR = os.path.join("GGIP", "checkpoints")
LOG_DIR = "log"



def create_directories():
    """Create necessary directories for checkpoints and logs"""
    dirs = [
        os.path.join(CHECKPOINT_DIR, "ggip1"),
        os.path.join(CHECKPOINT_DIR, "ggip2"),
        os.path.join(CHECKPOINT_DIR, "ggip3"),
        LOG_DIR
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def load_data(train_percent=TRAIN_PERCENT):
    """Load and split dataset into training and validation sets"""
    print("Loading dataset...")
    try:
        custom_dataset = ImuDataset(TRAIN_DATA_FOLDERS)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    total_size = len(custom_dataset)
    train_size = int(total_size * train_percent)

    # train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    train_dataset = Subset(custom_dataset, range(train_size))
    # val_dataset 包含数据集从 train_size 索引开始到末尾的样本
    val_dataset = Subset(custom_dataset, range(train_size, total_size))

    # Use pin_memory=True for GPU to optimize memory transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=False if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        pin_memory=False if torch.cuda.is_available() else False
    )

    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    print(f"Number of batches in train_loader: {len(train_loader)}\n")

    return train_loader, val_loader

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch + 1
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save model and optimizer state to checkpoint"""
    try:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def train_fdip_1(train_loader, val_loader, pretrained_epoch=0, epochs=1):
    """Train the FDIP_1 model for leaf joint position estimation"""
    print("\n===== Starting FDIP_1 Training =====")

    # Initialize model, loss function, optimizer
    model = FDIP_1(6 * 9,  5 * 3).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    # 添加FP16所需的GradScaler
    scaler = GradScaler()

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip1')) if LOG_ENABLED else None

    # Optional: Load pretrained model (commented out by default)
    # pretrain_path = os.path.join(CHECKPOINT_DIR, 'ggip1', 'epoch_190.pkl')
    # pretrained_epoch = load_checkpoint(model, optimizer, pretrain_path)

    # Training loop
    for epoch in range(epochs):
        try:
            current_epoch = epoch + pretrained_epoch + 1
            print(f'\n===== FDIP_1 Training Epoch: {current_epoch} =====')

            # Training phase
            model.train()
            train_losses = []
            epoch_step = 1

            #批分
            for batch_idx, data in enumerate(train_loader):
                try:
                    # Extract batch data
                    acc = data[0].to(DEVICE).float()
                    ori_6d = data[2].to(DEVICE).float()
                    p_leaf = data[3].to(DEVICE).float()

                    # Prepare input and target
                    x = torch.cat((acc, ori_6d), -1)
                    model_input = x.view(x.shape[0], x.shape[1], -1)
                    target = p_leaf.view(-1, p_leaf.shape[1], 15)

                    optimizer.zero_grad()
                    with autocast():
                        # Forward pass
                        logits = model(model_input)
                        # Compute loss and update
                        loss = torch.sqrt(criterion(logits, target))

                    # Check for NaN or Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN or Inf loss detected at batch {batch_idx}")
                        continue


                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Log training progress
                    train_losses.append(loss.item())
                    if LOG_ENABLED:
                        writer.add_scalar('mse_step/train', loss, epoch_step)

                    # Print progress
                    log_interval = len(train_loader) // 10
                    if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                        print(f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * BATCH_SIZE, len(train_loader.dataset))}/{len(train_loader.dataset)}]\tLoss: {loss:.6f}')

                    epoch_step += 1
                except Exception as e:
                    print(f"Error during training batch {batch_idx}: {e}")
                    continue
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("Out of memory error caught, clearing GPU cache")
                torch.cuda.empty_cache()
                # 如果想要在OOM发生时重新尝试当前epoch的处理，可以加一个重试机制
                continue  # 继续下一轮epoch
            else:
                raise e  # 如果是其他错误，重新抛出

        # Calculate average training loss
        if train_losses:
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f'Average Training Loss: {avg_train_loss:.6f}')
            if LOG_ENABLED:
                writer.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for data_val in val_loader:
                try:
                    acc_val = data_val[0].to(DEVICE).float()
                    ori_val = data_val[2].to(DEVICE).float()
                    p_leaf_val = data_val[3].to(DEVICE).float()

                    x_val = torch.cat((acc_val, ori_val), -1)
                    input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                    target_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15)

                    logits_val = model(input_val)
                    loss_val = torch.sqrt(criterion(logits_val, target_val))
                    val_losses.append(loss_val.item())
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue

            if val_losses:
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f'FDIP_1 Val: Average Validation Loss: {avg_val_loss:.6f}\n')
                if LOG_ENABLED:
                    writer.add_scalar('mse/val', avg_val_loss, current_epoch)

        # Save checkpoint
        if current_epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'ggip1', f'epoch_{current_epoch}.pkl')
            save_checkpoint(model, optimizer, current_epoch, checkpoint_path)

        # Clear GPU memory cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # Generate predictions
    print("Generating predictions for training and validation data...")
    train_predictions = predict_fdip_1(model, train_loader)
    val_predictions = predict_fdip_1(model, val_loader)

    if LOG_ENABLED and writer:
        writer.close()

    return model, train_predictions, val_predictions

def predict_fdip_1(model, data_loader):
    """Generate predictions using the trained AGGRU_1 model"""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for data in data_loader:
            try:
                acc = data[0].to(DEVICE).float()
                ori_6d = data[2].to(DEVICE).float()

                x = torch.cat((acc, ori_6d), -1)
                model_input = x.view(x.shape[0], x.shape[1], -1)
                logits = model(model_input)

                zeros = torch.zeros(logits.shape[:-1] + (3,), device=DEVICE)
                logits_extended = torch.cat([zeros, logits], dim=-1)
                logits_extended = logits_extended.view(*logits.shape[:-1], 6, 3)
                all_predictions.append(logits_extended)
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

    if all_predictions:
        all_predictions = torch.cat(all_predictions, dim=0)
        print(f"Predictions shape: {all_predictions.shape}")
    else:
        print("No predictions generated.")
        all_predictions = None

    return all_predictions

def train_fdip_2(train_loader, val_loader, train_predictions, val_predictions, pretrained_epoch=0, epochs=1):
    """Train the FDIP_2 model for all joint position estimation"""
    print("\n===== Starting FDIP_2 Training =====")

    # Initialize model, loss function, optimizer
    model = FDIP_2(6 * 12,  24 * 3).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    scaler = GradScaler()
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip2')) if LOG_ENABLED else None

    # Training loop
    for epoch in range(epochs):
        try:
            current_epoch = epoch + pretrained_epoch + 1
            print(f'\n===== FDIP_2 Training Epoch: {current_epoch} =====')

            # Training phase
            model.train()
            train_losses = []
            epoch_step = 1

            for batch_idx, data in enumerate(train_loader):
                try:
                    acc = data[0].to(DEVICE).float()
                    ori_6d = data[2].to(DEVICE).float()
                    p_all = data[4].to(DEVICE).float()

                    batch_start = batch_idx * BATCH_SIZE
                    batch_end = min((batch_idx + 1) * BATCH_SIZE, len(train_predictions))
                    p_leaf_pred = train_predictions[batch_start:batch_end]

                    x = torch.cat((acc, ori_6d, p_leaf_pred), -1)
                    model_input = x.view(x.shape[0], x.shape[1], -1)

                    zeros = torch.zeros(p_all.shape[:-1] + (3,), device=DEVICE)
                    target = torch.cat([zeros, p_all], dim=-1)
                    target = target.view(-1, target.shape[1], 72)

                    optimizer.zero_grad()
                    with autocast():
                        logits = model(model_input)
                        loss = torch.sqrt(criterion(logits, target))

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN or Inf loss detected at batch {batch_idx}")
                        continue

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_losses.append(loss.item())
                    if LOG_ENABLED:
                        writer.add_scalar('mse_step/train', loss, epoch_step)

                    log_interval = len(train_loader) // 10
                    if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                        print(f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * BATCH_SIZE, len(train_loader.dataset))}/{len(train_loader.dataset)}]\tLoss: {loss:.6f}')

                    epoch_step += 1
                except Exception as e:
                    print(f"Error during training batch {batch_idx}: {e}")
                    continue
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("Out of memory error caught, clearing GPU cache")
                torch.cuda.empty_cache()
                # 如果想要在OOM发生时重新尝试当前epoch的处理，可以加一个重试机制
                continue  # 继续下一轮epoch
            else:
                raise e  # 如果是其他错误，重新抛出


        if train_losses:
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f'FDIP_2: Average Training Loss: {avg_train_loss:.6f}')
            if LOG_ENABLED:
                writer.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                try:
                    acc_val = data_val[0].to(DEVICE).float()
                    ori_val = data_val[2].to(DEVICE).float()
                    p_all_val = data_val[4].to(DEVICE).float()

                    loader_batch_size_val = val_loader.batch_size
                    batch_start = batch_idx_val * loader_batch_size_val
                    current_actual_batch_size = acc_val.shape[0]  # 获取当前批次的真实大小
                    batch_end = batch_start + current_actual_batch_size
                    p_leaf_pred_val = val_predictions[batch_start:batch_end]

                    x_val = torch.cat((acc_val, ori_val, p_leaf_pred_val), -1)
                    input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)

                    zeros = torch.zeros(p_all_val.shape[:-1] + (3,), device=DEVICE)
                    target_val = torch.cat([zeros, p_all_val], dim=-1)
                    target_val = target_val.view(-1, target_val.shape[1], 72)

                    logits_val = model(input_val)
                    loss_val = torch.sqrt(criterion(logits_val, target_val))
                    val_losses.append(loss_val.item())
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue

            if val_losses:
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f'FDIP_2 Val: Average Validation Loss: {avg_val_loss:.6f}\n')
                if LOG_ENABLED:
                    writer.add_scalar('mse/val', avg_val_loss, current_epoch)

        # Save checkpoint
        if current_epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'ggip2', f'epoch_{current_epoch}.pkl')
            save_checkpoint(model, optimizer, current_epoch, checkpoint_path)

        # Clear GPU memory cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # Generate predictions
    print("Generating predictions for training and validation data...")
    train_predictions_2 = predict_fdip_2(model, train_loader, train_predictions)
    val_predictions_2 = predict_fdip_2(model, val_loader, val_predictions)

    if LOG_ENABLED and writer:
        writer.close()

    return model, train_predictions_2, val_predictions_2

def predict_fdip_2(model, data_loader, aggru1_predictions):
    """Generate predictions using the trained FDIP_2 model"""
    model.eval()
    all_predictions = []

    loader_batch_size = data_loader.batch_size

    with (torch.no_grad()):
        for batch_idx, data in enumerate(data_loader):
            try:
                acc = data[0].to(DEVICE).float()
                ori_6d = data[2].to(DEVICE).float()

                if loader_batch_size == BATCH_SIZE_VAL:
                    batch_start = batch_idx * BATCH_SIZE_VAL
                    current_actual_batch_size = acc.shape[0]
                    batch_end = batch_start + current_actual_batch_size
                else:
                    batch_start = batch_idx * BATCH_SIZE
                    batch_end = min((batch_idx + 1) * BATCH_SIZE, len(aggru1_predictions))

                p_leaf_pred = aggru1_predictions[batch_start:batch_end]

                # 检查切片后的批次大小是否与输入一致
                if p_leaf_pred.shape[0] != acc.shape[0]:
                    print("Error: Batch size mismatch after slicing!")
                    continue

                x = torch.cat((acc, ori_6d, p_leaf_pred), -1)
                model_input = x.view(x.shape[0], x.shape[1], -1)
                logits = model(model_input)
                logits = logits.view(logits.shape[0], logits.shape[1], 24, 3)

                # zeros = torch.zeros(logits.shape[:-1] + (3,), device=DEVICE)
                # logits_extended = torch.cat([logits, zeros], dim=-1)
                # logits_extended = logits_extended.view(*logits.shape[:-1], 24, 3)
                all_predictions.append(logits)
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

    if all_predictions:
        all_predictions = torch.cat(all_predictions, dim=0)
        print(f"AGGRU_2 predictions shape: {all_predictions.shape}")
    else:
        print("No predictions generated.")
        all_predictions = None

    return all_predictions

def train_fdip_3(train_loader, val_loader, train_predictions_2, val_predictions_2, pretrained_epoch=0, epochs=1):
    """Train the FDIP_3 model for joint rotations in 6D representation"""
    print("\n===== Starting FDIP_3 Training =====")

    # Initialize models and evaluator
    model = FDIP_3(24 * 12,  24 * 6).to(DEVICE)
    #evaluator = PoseEvaluator()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    scaler = GradScaler()
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(LOG_DIR, 'ggip3')) if LOG_ENABLED else None

    # Training loop
    for epoch in range(epochs):
        try:
            current_epoch = epoch + pretrained_epoch + 1
            print(f'\n===== FDIP_3 Training Epoch: {current_epoch} =====')

            # Training phase
            model.train()
            train_losses = []
            epoch_step = 1

            for batch_idx, data in enumerate(train_loader):
                try:
                    acc = data[0].to(DEVICE).float()
                    ori_6d = data[2].to(DEVICE).float()
                    pose_6d = data[6].to(DEVICE).float()

                    batch_start = batch_idx * BATCH_SIZE
                    batch_end = min((batch_idx + 1) * BATCH_SIZE, len(train_predictions_2))
                    p_all_pos = train_predictions_2[batch_start:batch_end]

                    x = torch.cat((acc, ori_6d), -1)
                    input_base = x.view(x.shape[0], x.shape[1], -1)

                    p_all_flattened = p_all_pos.view(x.shape[0], x.shape[1], -1)

                    # model_input = torch.cat((input_base, p_all_flattened), -1)
                    target = pose_6d.view(pose_6d.shape[0], pose_6d.shape[1], 144)

                    optimizer.zero_grad()
                    with autocast():
                        logits = model(input_base, p_all_flattened)

                        loss = torch.sqrt(criterion(logits, target))

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN or Inf loss detected at batch {batch_idx}")
                        continue

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_losses.append(loss.item())
                    if LOG_ENABLED:
                        writer.add_scalar('mse_step/train', loss, epoch_step)

                    log_interval = len(train_loader) // 10
                    if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                        print(f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * BATCH_SIZE, len(train_loader.dataset))}/{len(train_loader.dataset)}]\tLoss: {loss:.6f}')

                    epoch_step += 1
                except Exception as e:
                    print(f"Error during training batch {batch_idx}: {e}")
                    continue
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("Out of memory error caught, clearing GPU cache")
                torch.cuda.empty_cache()
                # 如果想要在OOM发生时重新尝试当前epoch的处理，可以加一个重试机制
                continue  # 继续下一轮epoch
            else:
                raise e  # 如果是其他错误，重新抛出

        if train_losses:
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f'FDIP_3: Average Training Loss: {avg_train_loss:.6f}')
            if LOG_ENABLED:
                writer.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                try:
                    acc_val = data_val[0].to(DEVICE).float()
                    ori_val = data_val[2].to(DEVICE).float()
                    pose_6d_val = data_val[6].to(DEVICE).float()

                    loader_batch_size_val = val_loader.batch_size
                    batch_start = batch_idx_val * loader_batch_size_val
                    current_actual_batch_size = acc_val.shape[0]  # 获取当前批次的真实大小
                    batch_end = batch_start + current_actual_batch_size

                    p_all_val = val_predictions_2[batch_start:batch_end]

                    x = torch.cat((acc_val, ori_val), -1)
                    input_base_val = x.view(x.shape[0], x.shape[1], -1)
                    p_all_flattened_val = p_all_val.view(x.shape[0], x.shape[1], -1)
                    # input_val = torch.cat((input_base_val, p_all_flattened_val), -1)
                    target_val = pose_6d_val.view(pose_6d_val.shape[0], pose_6d_val.shape[1], 144)

                    logits_val = model(input_base_val, p_all_flattened_val)
                    loss_val = torch.sqrt(criterion(logits_val, target_val))
                    val_losses.append(loss_val.item())
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue

            if val_losses:
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f'AGGRU_3 Val: Average Validation Loss: {avg_val_loss:.6f}\n')
                if LOG_ENABLED:
                    writer.add_scalar('mse/val', avg_val_loss, current_epoch)

        # **Monitor Soft Limits parameters in each epoch**
        if LOG_ENABLED and writer:
            # 记录软约束层的最小角度和最大角度
            writer.add_histogram(f'limits/min_angles_epoch_{current_epoch}', model.soft_limits.min_angles.data, current_epoch)
            writer.add_histogram(f'limits/max_angles_epoch_{current_epoch}', model.soft_limits.max_angles.data, current_epoch)

            # 打印当前角度限制
            print(f"Epoch {current_epoch} - Min angles: {model.soft_limits.min_angles.data.min().item():.4f} to {model.soft_limits.min_angles.data.max().item():.4f}")
            print(f"Epoch {current_epoch} - Max angles: {model.soft_limits.max_angles.data.min().item():.4f} to {model.soft_limits.max_angles.data.max().item():.4f}")


        # Save checkpoint
        if current_epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'ggip3', f'epoch_{current_epoch}.pkl')
            save_checkpoint(model, optimizer, current_epoch, checkpoint_path)

        # Clear GPU memory cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    if LOG_ENABLED and writer:
        writer.close()

    return model

def evaluate_pipeline(model1, model2, model3, data_loader):
    """Evaluate the complete pipeline on the validation dataset"""
    print("\n===== Evaluating Complete Pipeline =====")
    evaluator = PoseEvaluator()

    # Set all models to evaluation mode
    model1.eval()
    model2.eval()
    model3.eval()

    criterion = nn.MSELoss()
    val_losses = []
    all_sip_errs = []
    all_ang_errs = []
    all_pos_errs = []
    all_mesh_errs = []
    all_jerk_errs = []

    with torch.no_grad():
        for data_val in data_loader:
            try:
                acc = data_val[0].to(DEVICE).float()
                ori_6d = data_val[2].to(DEVICE).float()
                pose_6d_gt = data_val[6].to(DEVICE).float()

                # AGGRU_1: Leaf joint prediction
                x1 = torch.cat((acc, ori_6d), -1)
                input1 = x1.view(x1.shape[0], x1.shape[1], -1)
                leaf_pred = model1(input1)

                zeros = torch.zeros(leaf_pred.shape[:-1] + (3,), device=DEVICE)
                leaf_pred_ext = torch.cat([leaf_pred, zeros], dim=-1)
                leaf_pred_ext = leaf_pred_ext.view(*leaf_pred.shape[:-1], 6, 3)

                # AGGRU_2: All joint prediction
                x2 = torch.cat((acc, ori_6d, leaf_pred_ext), -1)
                input2 = x2.view(x2.shape[0], x2.shape[1], -1)
                all_joints_pred = model2(input2)

                zeros = torch.zeros(all_joints_pred.shape[:-1] + (3,), device=DEVICE)
                all_joints_pred_ext = torch.cat([all_joints_pred, zeros], dim=-1)
                all_joints_pred_ext = all_joints_pred_ext.view(*all_joints_pred.shape[:-1], 24, 3)

                # AGGRU_3: Joint rotation prediction
                x3 = torch.cat((acc, ori_6d), -1)
                input_base = x3.view(x3.shape[0], x3.shape[1], -1)
                joints_flattened = all_joints_pred_ext.view(x3.shape[0], x3.shape[1], -1)
                input3 = torch.cat((input_base, joints_flattened), -1)

                pose_pred = model3(input3)
                pose_gt = pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], 144)

                loss = torch.sqrt(criterion(pose_pred, pose_gt))
                val_losses.append(loss.item())

                errs = evaluator.eval(pose_pred, pose_gt)
                all_sip_errs.append(errs[0, 0].item())
                all_ang_errs.append(errs[1, 0].item())
                all_pos_errs.append(errs[2, 0].item())
                all_mesh_errs.append(errs[3, 0].item())
                all_jerk_errs.append(errs[4, 0].item())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

    if val_losses:
        avg_loss = sum(val_losses) / len(val_losses)
        avg_sip_err = sum(all_sip_errs) / len(all_sip_errs)
        avg_ang_err = sum(all_ang_errs) / len(all_ang_errs)
        avg_pos_err = sum(all_pos_errs) / len(all_pos_errs)
        avg_mesh_err = sum(all_mesh_errs) / len(all_mesh_errs)
        avg_jerk_err = sum(all_jerk_errs) / len(all_jerk_errs)

        print(f"Complete Pipeline Evaluation Results:")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"SIP Error (deg): {avg_sip_err:.4f}")
        print(f"Angular Error (deg): {avg_ang_err:.4f}")
        print(f"Positional Error (cm): {avg_pos_err:.4f}")
        print(f"Mesh Error (cm): {avg_mesh_err:.4f}")
        print(f"Jitter Error (100m/s^3): {avg_jerk_err:.4f}")
    else:
        print("No evaluation results generated.")

    return avg_loss, (avg_sip_err, avg_ang_err, avg_pos_err, avg_mesh_err, avg_jerk_err)

def main():
    """Main function to run the training pipeline"""
    print("===== Starting Training Pipeline =====")
    create_directories()
    train_loader, val_loader = load_data(train_percent=TRAIN_PERCENT)

    total_start_time = time.time()
    model1, train_predictions_1, val_predictions_1 = train_fdip_1(
        train_loader, val_loader, pretrained_epoch=0, epochs=1
    )
    model2, train_predictions_2, val_predictions_2 = train_fdip_2(
        train_loader, val_loader, train_predictions_1, val_predictions_1, pretrained_epoch=0, epochs=1
    )
    model3 = train_fdip_3(
        train_loader, val_loader, train_predictions_2, val_predictions_2, pretrained_epoch=0, epochs=3
    )


    print("\nTraining complete! Saving final checkpoints...")
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Total training time for all three models: {total_training_time:.2f} seconds")

    print("\nPerforming end-to-end evaluation of the complete pipeline...")
    evaluate_pipeline(model1, model2, model3, val_loader)

    print("Training and evaluation complete!")


if __name__ == '__main__':
    main()
