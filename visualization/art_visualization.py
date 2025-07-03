import os
import sys
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset_posReg import ImuDataset
# Import custom modules
from model.net import AGGRU_1, AGGRU_2, AGGRU_3

# Import articulate package for visualization
import articulate as art
#import articulate.utils.paths as paths

current_working_directory = os.getcwd()
print(f"当前工作目录是: {current_working_directory}")

# Device configuration
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
CHECKPOINT_DIR = os.path.join(os.path.dirname(current_working_directory),"train", "GGIP", "checkpoints")
SMPL_MODEL_PATH = "../SMPL/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"  # Replace with your SMPL model path


def rotation_6d_to_matrix(d6):
    """
    将 6D旋转表示转换为 3x3的正交旋转矩阵
    Args:
        d6: 6D rotation representation, shape [batch_size, ..., 6]
    Returns:
        Rotation matrix, shape [batch_size, ..., 3, 3]
    """
    # Create batch dimension if not present
    if d6.dim() == 1:
        d6 = d6.unsqueeze(0)

    # Original vector and shape
    original_shape = d6.shape[:-1]
    d6 = d6.view(-1, 6)

    # Extract the first two columns
    x_raw = d6[:, 0:3]
    y_raw = d6[:, 3:6]

    # Normalize columns
    x = x_raw / torch.norm(x_raw, dim=1, keepdim=True)
    z = torch.cross(x, y_raw)
    z = z / torch.norm(z, dim=1, keepdim=True)
    y = torch.cross(z, x)

    # Stack and reshape
    matrix = torch.stack([x, y, z], dim=2)
    return matrix.view(*original_shape, 3, 3)


def load_trained_models(model1_path=None, model2_path=None, model3_path=None):
    """
    加载 AGGRU 模型的 checkpoint, 并返回三个模型

    Args:
        model1_path: Path to AGGRU_1 checkpoint
        model2_path: Path to AGGRU_2 checkpoint
        model3_path: Path to AGGRU_3 checkpoint

    Returns:
        model1, model2, model3: Loaded models
    """
    # Default checkpoint paths if not specified
    if model1_path is None:
        model1_path = os.path.join(CHECKPOINT_DIR, 'ggip1', 'epoch_final.pkl')
    if model2_path is None:
        model2_path = os.path.join(CHECKPOINT_DIR, 'ggip2', 'epoch_final.pkl')
    if model3_path is None:
        model3_path = os.path.join(CHECKPOINT_DIR, 'ggip3', 'epoch_final.pkl')

    # Initialize models
    model1 = AGGRU_1(6 * 9, 256, 5 * 3).to(DEVICE)
    model2 = AGGRU_2(6 * 12, 256, 23 * 3).to(DEVICE)
    model3 = AGGRU_3(6 * 9 + 24 * 3, 256, 24 * 6).to(DEVICE)

    # Load checkpoints
    print("Loading model checkpoints...")
    model1_checkpoint = torch.load(model1_path, map_location=DEVICE)
    model2_checkpoint = torch.load(model2_path, map_location=DEVICE)
    model3_checkpoint = torch.load(model3_path, map_location=DEVICE)

    # Load state dictionaries
    model1.load_state_dict(model1_checkpoint['model_state_dict'])
    model2.load_state_dict(model2_checkpoint['model_state_dict'])
    model3.load_state_dict(model3_checkpoint['model_state_dict'])

    # Set models to evaluation mode
    model1.eval()
    model2.eval()
    model3.eval()

    print("Models loaded successfully!")
    return model1, model2, model3

def load_imu_data(imu_data_folder, batch_size=64, return_ground_truth=False):
    print(f"Loading IMU data from {imu_data_folder}...")

    try:
        custom_dataset = ImuDataset(imu_data_folder)
        data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

        all_acc = []
        all_ori = []
        all_gt_poses = []

        # 迭代所有批次
        for batch in data_loader:
            acc_batch = batch[0].to(DEVICE).float()
            ori_batch = batch[2].to(DEVICE).float()

            # 直接从索引5获取真实姿态数据
            if return_ground_truth:
                gt_pose_batch = batch[6].to(DEVICE).float()
                all_gt_poses.append(gt_pose_batch)

            all_acc.append(acc_batch)
            all_ori.append(ori_batch)

        # 合并所有批次数据
        acc_data = torch.cat(all_acc, dim=0)
        ori_data = torch.cat(all_ori, dim=0)

        print(f"All IMU data loaded. Acceleration shape: {acc_data.shape}, Orientation shape: {ori_data.shape}")
        if return_ground_truth and all_gt_poses:
            gt_poses = torch.cat(all_gt_poses, dim=0)
            print(f"Ground truth pose shape: {gt_poses.shape}")
            return acc_data, ori_data, gt_poses
        else:
            return acc_data, ori_data

    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def predict_pose(model1, model2, model3, acc_data, ori_data):
    """
    使用AGGRU模型从IMU数据预测6D姿态。

    Args:
        model1, model2, model3: Trained models
        acc_data: Acceleration data [batch_size, sequence_length, 9]
        ori_data: Orientation data (6D representation) [batch_size, sequence_length, 9]

    Returns:
        pose_6d: Predicted 6D pose [batch_size, sequence_length, 144]
    """
    print("Predicting pose using the trained models...")
    with torch.no_grad():
        # AGGRU_1: Predict leaf joint positions
        x1 = torch.cat((acc_data, ori_data), -1)
        input1 = x1.view(x1.shape[0], x1.shape[1], -1)
        leaf_pred = model1(input1)

        # Extend to full dimensions
        zeros = torch.zeros(leaf_pred.shape[:-1] + (3,), device=DEVICE)
        leaf_pred_ext = torch.cat([leaf_pred, zeros], dim=-1)
        leaf_pred_ext = leaf_pred_ext.view(*leaf_pred.shape[:-1], 6, 3)

        # AGGRU_2: Predict all joint positions
        x2 = torch.cat((acc_data, ori_data, leaf_pred_ext), -1)
        input2 = x2.view(x2.shape[0], x2.shape[1], -1)
        all_joints_pred = model2(input2)

        # Extend to full dimensions
        zeros = torch.zeros(all_joints_pred.shape[:-1] + (3,), device=DEVICE)
        all_joints_pred_ext = torch.cat([all_joints_pred, zeros], dim=-1)
        all_joints_pred_ext = all_joints_pred_ext.view(*all_joints_pred.shape[:-1], 24, 3)

        # AGGRU_3: Predict joint rotations
        x3 = torch.cat((acc_data, ori_data), -1)
        input_base = x3.view(x3.shape[0], x3.shape[1], -1)
        joints_flattened = all_joints_pred_ext.view(x3.shape[0], x3.shape[1], -1)
        input3 = torch.cat((input_base, joints_flattened), -1)

        pose_6d = model3(input3)  # [batch_size, sequence_length, 144]

    print(f"Pose prediction complete. Output shape: {pose_6d.shape}")
    return pose_6d


def convert_6d_to_smpl_params(pose_6d):
    """
    将6D旋转表示转换为 SMPL 可接受的旋转矩阵和轴-角格式。

    Args:
        pose_6d: 6D rotation representation [batch_size, sequence_length, 144]

    Returns:
        pose_matrices: Rotation matrices [batch_size, sequence_length, 24, 3, 3]
        pose_axis_angle: SMPL-compatible pose parameters in axis-angle format [batch_size, sequence_length, 72]
    """
    print("Converting 6D rotation representation to SMPL parameters...")
    batch_size, seq_len = pose_6d.shape[0], pose_6d.shape[1]

    # Reshape 6D representation to [batch, seq_len, 24, 6]
    pose_6d_reshaped = pose_6d.view(batch_size, seq_len, 24, 6)

    # Convert to rotation matrices [batch, seq_len, 24, 3, 3]
    pose_matrices = rotation_6d_to_matrix(pose_6d_reshaped)

    # Convert rotation matrices to axis-angle representation for SMPL
    def matrix_to_axis_angle(matrices):
        """Convert rotation matrices to axis-angle representation."""
        # Extract batch dimensions
        batch_dims = matrices.shape[:-2]
        matrices = matrices.reshape(-1, 3, 3)

        # Compute trace and extract angle
        trace = matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))

        # Extract axis
        axis_x = matrices[:, 2, 1] - matrices[:, 1, 2]
        axis_y = matrices[:, 0, 2] - matrices[:, 2, 0]
        axis_z = matrices[:, 1, 0] - matrices[:, 0, 1]
        axis = torch.stack([axis_x, axis_y, axis_z], dim=1)

        # Normalize axis where angle != 0
        non_zero_mask = angle > 1e-6
        axis[non_zero_mask] = axis[non_zero_mask] / torch.norm(axis[non_zero_mask], dim=1, keepdim=True)

        # Compute axis-angle representation
        axis_angle = axis * angle.unsqueeze(1)

        # Handle special case where angle = 0
        axis_angle[~non_zero_mask] = 0.0

        # Reshape to original batch dimensions
        axis_angle = axis_angle.reshape(*batch_dims, 3)
        return axis_angle

    # Apply the conversion
    pose_axis_angle = matrix_to_axis_angle(pose_matrices)
    pose_axis_angle = pose_axis_angle.reshape(batch_size, seq_len, 24 * 3)

    print(f"Conversion complete. Matrices shape: {pose_matrices.shape}, Axis-angle shape: {pose_axis_angle.shape}")
    return pose_matrices, pose_axis_angle


def visualize_articulate_animation(pose_matrices_list, trans_list=None):
    """
    利用 articulate 库渲染 SMPL 模型动画

    Args:
        pose_matrices_list: List of rotation matrices [sequence_length, 24, 3, 3]
        trans_list: List of translation parameters [sequence_length, 3] (optional)
    """
    print("Visualizing animation using articulate package...")

    # 如果未提供平移数据，创建默认值
    if trans_list is None:
        trans_list = [torch.zeros(pose.shape[0], 3) for pose in pose_matrices_list]

    # 初始化SMPL模型
    model = art.ParametricModel(SMPL_MODEL_PATH)

    # 查看动作
    print("Opening articulate viewer. Close the window to continue.")
    model.view_motion(pose_matrices_list, trans_list)

    print("Animation visualization complete!")
def visualize_imu_to_smpl(imu_data_file, model1_path=None, model2_path=None, model3_path=None, use_ground_truth=False):
    """
    Complete workflow from IMU data to SMPL animation using articulate.

    Args:
        imu_data_file: Path to IMU data file
        model1_path: Path to AGGRU_1 checkpoint
        model2_path: Path to AGGRU_2 checkpoint
        model3_path: Path to AGGRU_3 checkpoint
    """
    # 1. Load models
    model1, model2, model3 = None, None, None
    if not use_ground_truth:
        model1, model2, model3 = load_trained_models(model1_path, model2_path, model3_path)

    # 2. Load IMU data
    if use_ground_truth:
        acc_data, ori_data, gt_poses = load_imu_data(imu_data_file, return_ground_truth=True)
    else:
        acc_data, ori_data = load_imu_data(imu_data_file)

    # 3. Predict pose
    if use_ground_truth:
        print("Using ground truth poses instead of predictions")
        # 假设gt_poses已经是正确的24个关节的144维表示(24关节*6D)
        pose_matrices, _ = convert_6d_to_smpl_params(gt_poses)
    else:
        # 使用预测的姿态
        pose_6d = predict_pose(model1, model2, model3, acc_data, ori_data)
        pose_matrices, _ = convert_6d_to_smpl_params(pose_6d)

    # 5. 转移到CPU
    pose_matrices = pose_matrices.detach().cpu()

    # 6. 创建平移数据
    trans = torch.zeros(pose_matrices.shape[0], pose_matrices.shape[1], 3)  # 注意形状应匹配帧数

    # 7. 将所有序列连接成一个连续序列
    # 选择前n个序列以避免过长
    num_sequences = min(50, pose_matrices.shape[0])
    continuous_poses = torch.cat([pose_matrices[i] for i in range(num_sequences)], dim=0)
    continuous_trans = torch.cat([trans[i] for i in range(num_sequences)], dim=0)

    print(f"Combined {num_sequences} sequences into a continuous animation with {continuous_poses.shape[0]} frames")

    # 8. 一次性可视化连续序列
    visualize_articulate_animation([continuous_poses], [continuous_trans])

    print("Visualization process complete!")

if __name__ == "__main__":
    # Replace with your actual paths
    IMU_DATA_FILE = [r"F:\CodeForPaper\Dataset\SingleOne\Pt"]
    MODEL1_PATH = os.path.join(CHECKPOINT_DIR, 'ggip1', 'epoch_100.pkl')
    MODEL2_PATH = os.path.join(CHECKPOINT_DIR, 'ggip2', 'epoch_100.pkl')
    MODEL3_PATH = os.path.join(CHECKPOINT_DIR, 'ggip3', 'epoch_100.pkl')

    USE_GROUND_TRUTH = True  # Set to True to use ground truth poses instead of predictions

    # Run the complete visualization pipeline
    visualize_imu_to_smpl(
        imu_data_file=IMU_DATA_FILE,
        model1_path=MODEL1_PATH,
        model2_path=MODEL2_PATH,
        model3_path=MODEL3_PATH,
        use_ground_truth=USE_GROUND_TRUTH
    )
