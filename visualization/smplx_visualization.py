import os
import sys
import torch
import numpy as np
#import smplx
import open3d as o3d
from matplotlib import animation
import matplotlib.pyplot as plt
# Import custom modules - make sure these are in your project path
from model.net import AGGRU_1, AGGRU_2, AGGRU_3


current_working_directory = os.getcwd()
print(f"当前工作目录是: {current_working_directory}")

# Device configuration
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
CHECKPOINT_DIR = os.path.join(os.path.dirname(current_working_directory), "GGIP", "checkpoints")
SMPL_MODEL_PATH = "../SMPL/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"  # Replace with your SMPL model path


def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to rotation matrix.
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
    Load the trained AGGRU models from checkpoints.

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


def load_imu_data(imu_data_file):
    """
    Load IMU data from file.

    Args:
        imu_data_file: Path to the IMU data file (could be .npz, .pt, etc.)

    Returns:
        acc_data: Acceleration data [batch_size, sequence_length, 9]
        ori_data: Orientation data (6D representation) [batch_size, sequence_length, 9]
    """
    print(f"Loading IMU data from {imu_data_file}...")

    # Check file extension to determine loading method
    if imu_data_file.endswith('.npz'):
        data = np.load(imu_data_file)
        acc_data = torch.tensor(data['acc'], device=DEVICE).float()
        ori_data = torch.tensor(data['ori_6d'], device=DEVICE).float()
    elif imu_data_file.endswith('.pt'):
        data = torch.load(imu_data_file, map_location=DEVICE)
        acc_data = data['acc'].float()
        ori_data = data['ori_6d'].float()
    else:
        raise ValueError(f"Unsupported file format: {imu_data_file}")

    # Ensure batch dimension
    if acc_data.dim() == 2:
        acc_data = acc_data.unsqueeze(0)
    if ori_data.dim() == 2:
        ori_data = ori_data.unsqueeze(0)

    print(f"IMU data loaded. Acceleration shape: {acc_data.shape}, Orientation shape: {ori_data.shape}")
    return acc_data, ori_data


def predict_pose(model1, model2, model3, acc_data, ori_data):
    """
    Use three models to sequentially predict pose.

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
    Convert 6D rotation representation to SMPL-compatible parameters.

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
    # This is a simplified version - in practice, you might need a more sophisticated conversion
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


def visualize_smpl_animation(pose_params, betas=None, fps=30, output_file='animation.mp4'):
    """
    Create animation using SMPL model and converted pose parameters.

    Args:
        pose_params: SMPL pose parameters [sequence_length, 72]
        betas: SMPL shape parameters [10,] (optional)
        fps: Animation frame rate
        output_file: Output video file name
    """
    print(f"Visualizing SMPL animation at {fps} FPS...")

    # Default body shape if not provided
    if betas is None:
        betas = torch.zeros(10, device=DEVICE)  # Default body shape

    # Initialize SMPL model
    smpl_model = smplx.create(
        model_path=SMPL_MODEL_PATH,
        model_type='smpl',
        gender='neutral',
        batch_size=1
    ).to(DEVICE)

    # Extract sequence length
    seq_len = pose_params.shape[0]

    # Setup figure for animation
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh objects for each frame
    print("Generating SMPL meshes for each frame...")
    meshes = []
    vertices_list = []

    for i in range(seq_len):
        # Get current frame's pose parameters
        current_pose = pose_params[i:i + 1]

        # Generate SMPL mesh for current frame
        output = smpl_model(
            betas=betas.unsqueeze(0),
            body_pose=current_pose[:, 3:],  # Body pose (excluding global orientation)
            global_orient=current_pose[:, :3],  # Global orientation
            return_verts=True
        )

        vertices = output.vertices[0].detach().cpu().numpy()
        faces = smpl_model.faces

        vertices_list.append(vertices)

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        meshes.append(mesh)

    print(f"Generated {len(meshes)} mesh frames")

    # Method 1: Open3D visualization
    def display_with_open3d():
        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add first mesh
        vis.add_geometry(meshes[0])

        # Update callback function
        def update_mesh(vis, i):
            vis.clear_geometries()
            vis.add_geometry(meshes[i])
            vis.update_renderer()
            return False

        # Set view control
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)

        # Run animation
        for i in range(seq_len):
            update_mesh(vis, i)
            vis.poll_events()
            vis.update_renderer()
            # Add delay for smoother animation
            # time.sleep(1/fps)

        vis.destroy_window()

    # Method 2: Matplotlib animation (uncomment to use)
    def create_matplotlib_animation():
        # Plot setup
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Plot the first frame
        verts = vertices_list[0]
        face_indices = smpl_model.faces

        # Create initial plot
        plot = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                               triangles=face_indices,
                               color='lightblue',
                               alpha=0.8,
                               edgecolor='black',
                               linewidth=0.1)

        # Update function for animation
        def update(frame):
            ax.clear()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            verts = vertices_list[frame]
            plot = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                   triangles=face_indices,
                                   color='lightblue',
                                   alpha=0.8,
                                   edgecolor='black',
                                   linewidth=0.1)
            return [plot]

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=seq_len,
                                      interval=1000 / fps, blit=False)

        # Save animation
        print(f"Saving animation to {output_file}...")
        ani.save(output_file, writer='ffmpeg', fps=fps)
        plt.close()

    # Use Open3D for display (interactive)
    display_with_open3d()

    # Use Matplotlib for saving (uncomment to use)
    # create_matplotlib_animation()

    print("Animation complete!")


def visualize_imu_to_smpl(imu_data_file, model1_path=None, model2_path=None, model3_path=None,
                          output_file='imu_motion.mp4'):
    """
    Complete workflow from IMU data to SMPL animation.

    Args:
        imu_data_file: Path to IMU data file
        model1_path: Path to AGGRU_1 checkpoint
        model2_path: Path to AGGRU_2 checkpoint
        model3_path: Path to AGGRU_3 checkpoint
        output_file: Output video file name
    """
    # 1. Load models
    model1, model2, model3 = load_trained_models(model1_path, model2_path, model3_path)

    # 2. Load IMU data
    acc_data, ori_data = load_imu_data(imu_data_file)

    # 3. Predict pose
    pose_6d = predict_pose(model1, model2, model3, acc_data, ori_data)

    # 4. Convert to SMPL parameters
    _, pose_params = convert_6d_to_smpl_params(pose_6d)

    # 5. Visualize
    # Get first batch and move to CPU
    pose_params = pose_params[0].detach().cpu()

    # Optional: Create custom body shape
    # betas = torch.zeros(10, device='cpu')  # Default shape

    visualize_smpl_animation(pose_params, output_file=output_file)

    print(f"Visualization process complete. Output saved to {output_file}")


if __name__ == "__main__":
    # Replace with your actual paths
    IMU_DATA_FILE = "path/to/your/imu_data.npz"
    MODEL1_PATH = os.path.join(CHECKPOINT_DIR, 'ggip1', 'epoch_final.pkl')
    MODEL2_PATH = os.path.join(CHECKPOINT_DIR, 'ggip2', 'epoch_final.pkl')
    MODEL3_PATH = os.path.join(CHECKPOINT_DIR, 'ggip3', 'epoch_final.pkl')
    OUTPUT_FILE = "imu_motion.mp4"

    # Run the complete visualization pipeline
    visualize_imu_to_smpl(
        imu_data_file=IMU_DATA_FILE,
        model1_path=MODEL1_PATH,
        model2_path=MODEL2_PATH,
        model3_path=MODEL3_PATH,
        output_file=OUTPUT_FILE
    )
