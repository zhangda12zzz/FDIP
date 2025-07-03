import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
import config as conf
from data.dataset_posReg import ImuDataset
from model.net import AGGRU_1, AGGRU_2, AGGRU_3
from evaluator import PoseEvaluator

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set GPU device

# Hyperparameters
LEARNING_RATE = 1e-4
CHECKPOINT_INTERVAL = 10
BATCH_SIZE = 64
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
LOG_ENABLED = True

# Paths
TRAIN_DATA_FOLDERS = [
    "F:\\CodeForPaper\\Dataset\\AMASS\\HumanEva\\pt",
    "F:\\CodeForPaper\\Dataset\\DIPIMUandOthers\\DIP_6\\Detail"
]
CHECKPOINT_DIR = "GGIP/checkpoints"
LOG_DIR = "log"


def create_directories():
    """Create necessary directories for checkpoints and logs"""
    dirs = [
        f"{CHECKPOINT_DIR}/ggip1",
        f"{CHECKPOINT_DIR}/ggip2",
        f"{CHECKPOINT_DIR}/ggip3",
        LOG_DIR
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def load_data(train_percent=0.9):
    """Load and split dataset into training and validation sets"""
    print("Loading dataset...")
    custom_dataset = ImuDataset(TRAIN_DATA_FOLDERS)

    train_size = int(len(custom_dataset) * train_percent)
    val_size = len(custom_dataset) - train_size

    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    print(f"Number of batches in train_loader: {len(train_loader)}\n")

    return train_loader, val_loader


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch + 1


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save model and optimizer state to checkpoint"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def train_aggru_1(train_loader, val_loader, pretrained_epoch=0, epochs=1):
    """Train the AGGRU_1 model for leaf joint position estimation"""
    print("\n===== Starting AGGRU_1 Training =====")

    # Initialize model, loss function, optimizer
    model = AGGRU_1(6 * 9, 256, 5 * 3).to(DEVICE)  # 6 IMUs, 9 features each, 5 leaf joints, 3D positions
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    # TensorBoard writer
    writer = SummaryWriter(f'{LOG_DIR}/ggip1') if LOG_ENABLED else None

    # Optional: Load pretrained model
    # pretrain_path = f'{CHECKPOINT_DIR}/ggip1/epoch_190.pkl'
    # pretrained_epoch = load_checkpoint(model, optimizer, pretrain_path)

    # Training loop
    for epoch in range(epochs):
        current_epoch = epoch + pretrained_epoch + 1
        print(f'\n===== AGGRU_1 Training Epoch: {current_epoch} =====')

        # Training phase
        model.train()
        train_losses = []
        epoch_step = 1

        for batch_idx, data in enumerate(train_loader):
            # Extract batch data
            acc = data[0].to(DEVICE).float()  # Accelerometer data [batch_size, max_seq, 18]
            ori_6d = data[2].to(DEVICE).float()  # Orientation data [batch_size, max_seq, 54]
            p_leaf = data[3].to(DEVICE).float()  # Leaf joint positions [batch_size, max_seq, 5, 3]

            # Prepare input and target
            x = torch.cat((acc, ori_6d), -1)
            model_input = x.view(x.shape[0], x.shape[1], -1)  # [batch_size, max_seq, 72]
            target = p_leaf.view(-1, p_leaf.shape[1], 15)  # [batch_size, max_seq, 15]

            # Forward pass
            logits = model(model_input)

            # Compute loss and update
            optimizer.zero_grad()
            loss = torch.sqrt(criterion(logits, target).to(DEVICE))
            loss.backward()
            optimizer.step()

            # Log training progress
            train_losses.append(loss.item())
            if LOG_ENABLED:
                writer.add_scalar('mse_step/train', loss, epoch_step)

            # Print progress
            log_interval = len(train_loader) // 10  # Print every 10%
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * BATCH_SIZE, len(train_loader.dataset))}/{len(train_loader.dataset)}]\tLoss: {loss:.6f}')

            epoch_step += 1

        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Average Training Loss: {avg_train_loss:.6f}')
        if LOG_ENABLED:
            writer.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for data_val in val_loader:
                # Extract batch data
                acc_val = data_val[0].to(DEVICE).float()
                ori_val = data_val[2].to(DEVICE).float()
                p_leaf_val = data_val[3].to(DEVICE).float()

                # Prepare input and target
                x_val = torch.cat((acc_val, ori_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15)

                # Forward pass
                logits_val = model(input_val)

                # Compute validation loss
                loss_val = torch.sqrt(criterion(logits_val, target_val).to(DEVICE))
                val_losses.append(loss_val.item())

            # Calculate average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Val: Average Validation Loss: {avg_val_loss:.6f}\n')
            if LOG_ENABLED:
                writer.add_scalar('mse/val', avg_val_loss, current_epoch)

        # Save checkpoint
        if current_epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model,
                optimizer,
                current_epoch,
                f"{CHECKPOINT_DIR}/ggip1/epoch_{current_epoch}.pkl"
            )

    # Generate predictions for entire dataset
    print("Generating predictions for training and validation data...")
    train_predictions = predict_aggru_1(model, train_loader)
    val_predictions = predict_aggru_1(model, val_loader)

    if LOG_ENABLED and writer:
        writer.close()

    return model, train_predictions, val_predictions


def predict_aggru_1(model, data_loader):
    """Generate predictions using the trained AGGRU_1 model"""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for data in data_loader:
            acc = data[0].to(DEVICE).float()
            ori_6d = data[2].to(DEVICE).float()

            # Prepare input
            x = torch.cat((acc, ori_6d), -1)
            model_input = x.view(x.shape[0], x.shape[1], -1)

            # Forward pass
            logits = model(model_input)

            # Add padding zeros to match the expected format for AGGRU_2
            zeros = torch.zeros(logits.shape[:-1] + (3,), device=DEVICE)
            logits_extended = torch.cat([logits, zeros], dim=-1)  # Add 3 zeros in the last dimension
            logits_extended = logits_extended.view(*logits.shape[:-1], 6, 3)

            all_predictions.append(logits_extended)

    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"Predictions shape: {all_predictions.shape}")

    return all_predictions


def train_aggru_2(train_loader, val_loader, train_predictions, val_predictions, pretrained_epoch=0, epochs=1):
    """Train the AGGRU_2 model for all joint position estimation"""
    print("\n===== Starting AGGRU_2 Training =====")

    # Initialize model, loss function, optimizer
    model = AGGRU_2(6 * 12, 256, 23 * 3).to(DEVICE)  # 6 IMUs, 12 features each (acc+ori+leaf), 23 joints
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    # TensorBoard writer
    writer = SummaryWriter(f'{LOG_DIR}/ggip2') if LOG_ENABLED else None

    # Optional: Load pretrained model
    # pretrain_path = f'{CHECKPOINT_DIR}/ggip2/epoch_60.pkl'
    # pretrained_epoch = load_checkpoint(model, optimizer, pretrain_path)

    # Training loop
    for epoch in range(epochs):
        current_epoch = epoch + pretrained_epoch + 1
        print(f'\n===== AGGRU_2 Training Epoch: {current_epoch} =====')

        # Training phase
        model.train()
        train_losses = []
        epoch_step = 1

        for batch_idx, data in enumerate(train_loader):
            # Extract batch data
            acc = data[0].to(DEVICE).float()
            ori_6d = data[2].to(DEVICE).float()
            p_all = data[4].to(DEVICE).float()  # All joint positions [batch_size, max_seq, 23, 3]

            # Get AGGRU_1 predictions for this batch
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, len(train_predictions))
            p_leaf_pred = train_predictions[batch_start:batch_end]

            # Prepare input and target
            x = torch.cat((acc, ori_6d, p_leaf_pred), -1)
            model_input = x.view(x.shape[0], x.shape[1], -1)
            target = p_all.view(-1, p_all.shape[1], 69)  # [batch_size, max_seq, 69]

            # Forward pass
            logits = model(model_input)

            # Compute loss and update
            optimizer.zero_grad()
            loss = torch.sqrt(criterion(logits, target).to(DEVICE))
            loss.backward()
            optimizer.step()

            # Log training progress
            train_losses.append(loss.item())
            if LOG_ENABLED:
                writer.add_scalar('mse_step/train', loss, epoch_step)

            # Print progress
            log_interval = len(train_loader) // 10  # Print every 10%
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * BATCH_SIZE, len(train_loader.dataset))}/{len(train_loader.dataset)}]\tLoss: {loss:.6f}')

            epoch_step += 1

        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'AGGRU_2: Average Training Loss: {avg_train_loss:.6f}')
        if LOG_ENABLED:
            writer.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                # Extract batch data
                acc_val = data_val[0].to(DEVICE).float()
                ori_val = data_val[2].to(DEVICE).float()
                p_all_val = data_val[4].to(DEVICE).float()

                # Get AGGRU_1 predictions for this validation batch
                p_leaf_pred_val = val_predictions[batch_idx_val:batch_idx_val + 1]

                # Prepare input and target
                x_val = torch.cat((acc_val, ori_val, p_leaf_pred_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_all_val.view(-1, p_all_val.shape[1], 69)

                # Forward pass
                logits_val = model(input_val)

                # Compute validation loss
                loss_val = torch.sqrt(criterion(logits_val, target_val).to(DEVICE))
                val_losses.append(loss_val.item())

            # Calculate average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Val: Average Validation Loss: {avg_val_loss:.6f}\n')
            if LOG_ENABLED:
                writer.add_scalar('mse/val', avg_val_loss, current_epoch)

        # Save checkpoint
        if current_epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model,
                optimizer,
                current_epoch,
                f"{CHECKPOINT_DIR}/ggip2/epoch_{current_epoch}.pkl"
            )

    # Generate predictions for entire dataset
    print("Generating predictions for training and validation data...")
    train_predictions_2 = predict_aggru_2(model, train_loader, train_predictions)
    val_predictions_2 = predict_aggru_2(model, val_loader, val_predictions)

    if LOG_ENABLED and writer:
        writer.close()

    return model, train_predictions_2, val_predictions_2


def predict_aggru_2(model, data_loader, aggru1_predictions):
    """Generate predictions using the trained AGGRU_2 model"""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # Extract batch data
            acc = data[0].to(DEVICE).float()
            ori_6d = data[2].to(DEVICE).float()

            # Get AGGRU_1 predictions
            if data_loader.batch_size == 1:  # Validation loader
                batch_start = batch_idx
                batch_end = batch_idx + 1
            else:  # Training loader
                batch_start = batch_idx * BATCH_SIZE
                batch_end = min((batch_idx + 1) * BATCH_SIZE, len(aggru1_predictions))

            p_leaf_pred = aggru1_predictions[batch_start:batch_end]

            # Prepare input
            x = torch.cat((acc, ori_6d, p_leaf_pred), -1)
            model_input = x.view(x.shape[0], x.shape[1], -1)

            # Forward pass
            logits = model(model_input)

            # Add padding zeros for root joint
            zeros = torch.zeros(logits.shape[:-1] + (3,), device=DEVICE)
            logits_extended = torch.cat([logits, zeros], dim=-1)  # Add 3 zeros for root joint
            logits_extended = logits_extended.view(*logits.shape[:-1], 24, 3)

            all_predictions.append(logits_extended)

    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"AGGRU_2 predictions shape: {all_predictions.shape}")

    return all_predictions


def train_aggru_3(train_loader, val_loader, train_predictions_2, val_predictions_2,
                  pretrained_epoch=0, epochs=1):
    """Train the AGGRU_3 model for joint rotations in 6D representation"""
    print("\n===== Starting AGGRU_3 Training =====")

    # Initialize models and evaluator
    model = AGGRU_3(6 * 9 + 24 * 3, 256, 24 * 6).to(DEVICE)  # Input: IMUs + joint positions, Output: 6D rotations
    evaluator = PoseEvaluator()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    # TensorBoard writer
    writer = SummaryWriter(f'{LOG_DIR}/ggip3') if LOG_ENABLED else None

    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        current_epoch = epoch + pretrained_epoch + 1
        print(f'\n===== AGGRU_3 Training Epoch: {current_epoch} =====')

        # Training phase
        model.train()
        train_losses = []
        epoch_step = 1

        for batch_idx, data in enumerate(train_loader):
            # Extract batch data
            acc = data[0].to(DEVICE).float()  # [batch_size, max_seq, 6, 3]
            ori_6d = data[2].to(DEVICE).float()  # [batch_size, max_seq, 6, 6]
            pose_6d = data[6].to(DEVICE).float()  # [batch_size, max_seq, 24, 6]

            # Get AGGRU_2 predictions for this batch
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, len(train_predictions_2))
            p_all_pos = train_predictions_2[batch_start:batch_end]

            # Prepare input and target
            x = torch.cat((acc, ori_6d), -1)
            input_base = x.view(x.shape[0], x.shape[1], -1)
            p_all_flattened = p_all_pos.view(x.shape[0], x.shape[1], -1)
            model_input = torch.cat((input_base, p_all_flattened), -1)
            target = pose_6d.view(pose_6d.shape[0], pose_6d.shape[1], 144)  # [batch_size, max_seq, 144]

            # Forward pass
            logits = model(model_input)

            # Compute loss and evaluation metrics
            optimizer.zero_grad()
            loss = torch.sqrt(criterion(logits, target).to(DEVICE))

            # Calculate additional evaluation metrics
            offline_errs = [evaluator.eval(logits, target)]
            offline_err = torch.stack(offline_errs).mean(dim=0)

            sip_err = offline_err[0, 0].item()
            ang_err = offline_err[1, 0].item()
            jerk_err = offline_err[4, 0].item()

            # Update model
            loss.backward()
            optimizer.step()

            # Log training progress
            train_losses.append(loss.item())
            if LOG_ENABLED:
                writer.add_scalar('mse_step/train', loss, epoch_step)
                writer.add_scalar('sip_err/train', sip_err, epoch_step)
                writer.add_scalar('ang_err/train', ang_err, epoch_step)
                writer.add_scalar('jerk_err/train', jerk_err, epoch_step)

            # Print progress
            log_interval = len(train_loader) // 10  # Print every 10%
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * BATCH_SIZE, len(train_loader.dataset))}/{len(train_loader.dataset)}]\tLoss: {loss:.6f}')
                print(f'sip_err: {sip_err:.4f}, ang_err: {ang_err:.4f}, jerk_err: {jerk_err:.4f}')

            epoch_step += 1

        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'AGGRU_3: Average Training Loss: {avg_train_loss:.6f}')
        if LOG_ENABLED:
            writer.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                # Extract batch data
                acc_val = data_val[0].to(DEVICE).float()
                ori_val = data_val[2].to(DEVICE).float()
                pose_6d_val = data_val[6].to(DEVICE).float()

                # Get AGGRU_2 predictions for this validation batch
                p_all_val = val_predictions_2[batch_idx_val:batch_idx_val + 1]

                # Prepare input and target
                x = torch.cat((acc_val, ori_val), -1)
                input_base_val = x.view(x.shape[0], x.shape[1], -1)
                p_all_flattened_val = p_all_val.view(x.shape[0], x.shape[1], -1)
                input_val = torch.cat((input_base_val, p_all_flattened_val), -1)
                target_val = pose_6d_val.view(pose_6d_val.shape[0], pose_6d_val.shape[1], 144)

                # Forward pass
                logits_val = model(input_val)

                # Compute validation loss
                loss_val = torch.sqrt(criterion(logits_val, target_val).to(DEVICE))
                val_losses.append(loss_val.item())

            # Calculate average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Val: Average Validation Loss: {avg_val_loss:.6f}\n')
            if LOG_ENABLED:
                writer.add_scalar('mse/val', avg_val_loss, current_epoch)

        # Save checkpoint
        if current_epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model,
                optimizer,
                current_epoch,
                f"{CHECKPOINT_DIR}/ggip3/epoch_{current_epoch}.pkl"
            )

    end_time = time.time()
    print(f'Total training time: {end_time - start_time:.2f} seconds')

    if LOG_ENABLED and writer:
        writer.close()

    return model


def main():
    """Main function to run the training pipeline"""
    # Create necessary directories
    create_directories()

    # Load data
    train_loader, val_loader = load_data(train_percent=0.9)

    # Train AGGRU_1 (Leaf joint position estimation)
    _, train_predictions_1, val_predictions_1 = train_aggru_1(
        train_loader,
        val_loader,
        pretrained_epoch=0,
        epochs=1
    )

    # Train AGGRU_2 (All joint position estimation)
    _, train_predictions_2, val_predictions_2 = train_aggru_2(
        train_loader,
        val_loader,
        train_predictions_1,
        val_predictions_1,
        pretrained_epoch=0,
        epochs=1
    )

    # Train AGGRU_3 (Joint rotation estimation)
    _ = train_aggru_3(
        train_loader,
        val_loader,
        train_predictions_2,
        val_predictions_2,
        pretrained_epoch=0,
        epochs=1
    )

    print("Training complete!")


if __name__ == '__main__':
    main()
