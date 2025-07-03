import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import gc  # Add garbage collection

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Configuration
learning_rate = 1e-4
checkpoint_interval = 10
epochs_aggru1 = 1  # Change to desired number (e.g., 200)
epochs_aggru2 = 1  # Change to desired number (e.g., 300)
batch_size = 64
train_percent = 0.9
log_on = True
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Import after environment setup
import config as conf
from data.dataset_posReg import ImuDataset
from model.net import AGGRU_1, AGGRU_2, AGGRU_3


# Helper function to save checkpoints
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


# Helper function to predict with AGGRU_1
def predict_with_aggru1(model, data_loader):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for data in data_loader:
            acc = data[0].to(device).float()
            ori_6d = data[2].to(device).float()
            x = torch.cat((acc, ori_6d), -1)
            input = x.view(x.shape[0], x.shape[1], -1)

            logits = model(input)
            zeros = torch.zeros(logits.shape[:-1] + (3,), device=logits.device)
            logits_extended = torch.cat([logits, zeros], dim=-1)
            logits_extended = logits_extended.view(*logits.shape[:-1], 6, 3)
            all_predictions.append(logits_extended)

    return torch.cat(all_predictions, dim=0)


if __name__ == '__main__':
    # Data loading
    train_data_folder = ["F:\CodeForPaper\Dataset\AMASS\HumanEva\pt",
                         "F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_6\Detail"]
    custom_dataset = ImuDataset(train_data_folder)

    train_size = int(len(custom_dataset) * train_percent)
    val_size = len(custom_dataset) - train_size

    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    train_batches_count = len(train_loader)
    print(f"Number of batches in train_loader: {train_batches_count}\n")

    # Create directories for checkpoints
    save_dir_aggru1 = "GGIP/checkpoints/ggip1"
    save_dir_aggru2 = "GGIP/checkpoints/ggip2"
    os.makedirs(save_dir_aggru1, exist_ok=True)
    os.makedirs(save_dir_aggru2, exist_ok=True)

    # Initialize TensorBoard writers
    writer_aggru1 = SummaryWriter('log/ggip1')
    writer_aggru2 = SummaryWriter('log/ggip2')

    # === AGGRU_1 Training ===
    print("\n=== Starting AGGRU_1 Training ===")

    # Model initialization
    rnn1 = AGGRU_1(6 * 9, 256, 5 * 3).to(device)
    criterion = nn.MSELoss()
    optimizer_aggru1 = optim.Adam(rnn1.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Pretrain model loading (uncomment if needed)
    pretrain_epoch = 0
    # pretrain_path = 'GGIP/checkpoints/ggip1/epoch_190.pkl'
    # if os.path.exists(pretrain_path):
    #     pretrain_data = torch.load(pretrain_path)
    #     rnn1.load_state_dict(pretrain_data['model_state_dict'])
    #     pretrain_epoch = pretrain_data['epoch'] + 1
    #     optimizer_aggru1.load_state_dict(pretrain_data['optimizer_state_dict'])
    #     print(f"Loaded pretrained model from epoch {pretrain_epoch-1}")

    for epoch in range(epochs_aggru1):
        current_epoch = epoch + pretrain_epoch + 1
        print(f'\n===== AGGRU_1 Training Epoch: {current_epoch} =====')

        # Reset loss lists for each epoch
        train_loss_list = []
        val_loss_list = []

        # Training
        rnn1.train()
        epoch_step = 1
        for batch_idx, data in enumerate(train_loader):
            acc = data[0].to(device).float()
            ori_6d = data[2].to(device).float()
            p_leaf = data[3].to(device).float()

            x = torch.cat((acc, ori_6d), -1)
            input = x.view(x.shape[0], x.shape[1], -1)
            target = p_leaf.view(-1, p_leaf.shape[1], 15)

            logits = rnn1(input)
            optimizer_aggru1.zero_grad()

            loss = torch.sqrt(criterion(logits, target).to(device))
            if log_on:
                writer_aggru1.add_scalar('mse_step/train', loss, epoch_step)
            train_loss_list.append(loss.item())  # Store the scalar value

            loss.backward()
            optimizer_aggru1.step()
            epoch_step += 1

            log_interval = len(train_loader) // 10
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * batch_size, len(train_dataset))}/{len(train_dataset)}]\tLoss: {loss.item():.6f}')

        # Calculate average training loss
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        print(f'Average Training Loss: {avg_train_loss:.6f}')

        if log_on:
            writer_aggru1.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation
        rnn1.eval()
        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()
                ori_val = data_val[2].to(device).float()
                p_leaf_val = data_val[3].to(device).float()

                x_val = torch.cat((acc_val, ori_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_leaf_val.view(-1, p_leaf_val.shape[1], 15)

                logits_val = rnn1(input_val)
                loss_val = torch.sqrt(criterion(logits_val, target_val).to(device))
                val_loss_list.append(loss_val.item())

        # Calculate average validation loss
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        print(f'Average Validation Loss: {avg_val_loss:.6f}\n')

        if log_on:
            writer_aggru1.add_scalar('mse/val', avg_val_loss, current_epoch)

        # Save checkpoint at intervals
        if current_epoch % checkpoint_interval == 0:
            save_checkpoint(rnn1, optimizer_aggru1, current_epoch,
                            f"{save_dir_aggru1}/epoch_{current_epoch}.pkl")

    # Generate predictions for both datasets for use in AGGRU_2
    print("Generating AGGRU_1 predictions for training and validation datasets...")
    train_predictions = predict_with_aggru1(rnn1, train_loader)
    val_predictions = predict_with_aggru1(rnn1, val_loader)

    print(f"Training predictions shape: {train_predictions.shape}")
    print(f"Validation predictions shape: {val_predictions.shape}")

    # Free some memory after AGGRU_1 training
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === AGGRU_2 Training ===
    print("\n=== Starting AGGRU_2 Training ===")

    # Model initialization
    rnn2 = AGGRU_2(6 * 12, 256, 23 * 3).to(device)
    optimizer_aggru2 = optim.Adam(rnn2.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Pretrain model loading (uncomment if needed)
    pretrain_epoch = 0
    # pretrain_path = 'GGIP/checkpoints/ggip2/epoch_60.pkl'
    # if os.path.exists(pretrain_path):
    #     pretrain_data = torch.load(pretrain_path)
    #     rnn2.load_state_dict(pretrain_data['model_state_dict'])
    #     pretrain_epoch = pretrain_data['epoch'] + 1
    #     optimizer_aggru2.load_state_dict(pretrain_data['optimizer_state_dict'])
    #     print(f"Loaded pretrained model from epoch {pretrain_epoch-1}")

    for epoch in range(epochs_aggru2):
        current_epoch = epoch + pretrain_epoch + 1
        print(f'\n===== AGGRU_2 Training Epoch: {current_epoch} =====')

        # Reset loss lists for each epoch
        train_loss_list = []
        val_loss_list = []

        # Training
        rnn2.train()
        epoch_step = 1
        for batch_idx, data in enumerate(train_loader):
            acc = data[0].to(device).float()
            ori_6d = data[2].to(device).float()
            p_all = data[4].to(device).float()

            # Get AGGRU_1 predictions for this batch
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(train_predictions))
            p_leaf_modify = train_predictions[batch_start:batch_end]

            x = torch.cat((acc, ori_6d, p_leaf_modify), -1)
            input = x.view(x.shape[0], x.shape[1], -1)
            target = p_all.view(-1, p_all.shape[1], 69)

            logits = rnn2(input)
            optimizer_aggru2.zero_grad()

            loss = criterion(logits, target).to(device)
            if log_on:
                writer_aggru2.add_scalar('mse_step/train', loss, epoch_step)
            train_loss_list.append(loss.item())

            loss.backward()
            optimizer_aggru2.step()
            epoch_step += 1

            log_interval = len(train_loader) // 10
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f'Train Epoch: {current_epoch} [{min((batch_idx + 1) * batch_size, len(train_dataset))}/{len(train_dataset)}]\tLoss: {loss.item():.6f}')

        # Calculate average training loss
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        print(f'AGGRU_2: Average Training Loss: {avg_train_loss:.6f}')

        if log_on:
            writer_aggru2.add_scalar('mse/train', avg_train_loss, current_epoch)

        # Validation
        rnn2.eval()
        with torch.no_grad():
            for batch_idx_val, data_val in enumerate(val_loader):
                acc_val = data_val[0].to(device).float()
                ori_val = data_val[2].to(device).float()
                p_all_val = data_val[4].to(device).float()

                # Get AGGRU_1 predictions for this validation sample
                p_leaf_modify_val = val_predictions[batch_idx_val:batch_idx_val + 1]

                x_val = torch.cat((acc_val, ori_val, p_leaf_modify_val), -1)
                input_val = x_val.view(x_val.shape[0], x_val.shape[1], -1)
                target_val = p_all_val.view(-1, p_all_val.shape[1], 69)

                logits_val = rnn2(input_val)
                loss_val = criterion(logits_val, target_val).to(device)
                val_loss_list.append(loss_val.item())

        # Calculate average validation loss
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        print(f'AGGRU_2: Average Validation Loss: {avg_val_loss:.6f}\n')

        if log_on:
            writer_aggru2.add_scalar('mse/val', avg_val_loss, current_epoch)

        # Save checkpoint at intervals
        if current_epoch % checkpoint_interval == 0:
            save_checkpoint(rnn2, optimizer_aggru2, current_epoch,
                            f"{save_dir_aggru2}/epoch_{current_epoch}.pkl")

    # Close TensorBoard writers
    writer_aggru1.close()
    writer_aggru2.close()

    print("Training completed successfully!")
