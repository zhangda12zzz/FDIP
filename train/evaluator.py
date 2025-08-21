import torch
import articulate as art


class PoseEvaluator:
    """Evaluator for 3D human poses using SMPL model"""

    def __init__(self,
                 smpl_model_path=r'I:\python\Ka_GAIP\data\SMPLmodel\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'):
        """
        Initialize the pose evaluator with a SMPL model

        Args:
            smpl_model_path: Path to the SMPL model file
        """
        self._eval_fn = art.FullMotionEvaluator(
            smpl_model_path,
            joint_mask=torch.tensor([1, 2, 16, 17])
        )

    def eval(self, pose_p, pose_t):
        """
        Evaluate the quality of predicted poses compared to ground truth

        Args:
            pose_p: Predicted poses in 6D rotation representation
            pose_t: Ground truth poses in 6D rotation representation

        Returns:
            Tensor containing:
            - Masked joint global angle error (degrees)
            - Joint global angle error (degrees)
            - Joint position error (cm) * 100
            - Vertex position error (cm) * 100
            - Prediction jitter error / 100
        """
        pose_p = art.math.r6d_to_rotation_matrix(pose_p.clone()).view(-1, 24, 3, 3)
        pose_t = art.math.r6d_to_rotation_matrix(pose_t.clone()).view(-1, 24, 3, 3)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors):
        """
        Print the evaluation metrics in a readable format

        Args:
            errors: Tensor containing the evaluation metrics
        """
        metric_names = [
            'SIP Error (deg)',
            'Angular Error (deg)',
            'Positional Error (cm)',
            'Mesh Error (cm)',
            'Jitter Error (100m/s^3)'
        ]

        for i, name in enumerate(metric_names):
            print(f'{name}: {errors[i, 0]:.2f} (+/- {errors[i, 1]:.2f})')


class PerFramePoseEvaluator:
    """
    一个真正计算逐帧误差的评估器，用于生成分布图。
    此版本全程保持 [B, S, ...] 维度，不进行合并，代码更清晰、安全。
    """

    def __init__(self,
                 smpl_model_path=r'F:\FDIP\basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
                 device='cuda:0', fps=60):
        self.device = torch.device(device)
        self.smpl_model = art.ParametricModel(smpl_model_path, device=self.device, use_pose_blendshape=False)
        self.fps = fps

    def eval(self, pose_p, pose_t):
        """
        计算一个批次中每一帧的各项误差。
        """
        # --- 0. 形状检查 ---
        print(pose_p.shape)
        if pose_p.dim() != 4 or pose_p.shape[2:] != (24, 6):
            raise ValueError(f"输入 pose_p 的形状应为 [B, S, 24, 6]，但得到 {pose_p.shape}")

        batch_size, seq_len = pose_p.shape[:2]

        # --- 1. 转换为旋转矩阵 ---
        pose_p_mat = art.math.r6d_to_rotation_matrix(pose_p)  # [B*S*24, 3, 3]
        pose_t_mat = art.math.r6d_to_rotation_matrix(pose_t)  # [B*S*24, 3, 3]

        # 重新整形为 [B*S, 24, 3, 3]
        pose_p_mat = pose_p_mat.view(batch_size * seq_len, 24, 3, 3)
        pose_t_mat = pose_t_mat.view(batch_size * seq_len, 24, 3, 3)

        # --- 2. 通过SMPL模型前向传播 ---
        pose_global_p, joint_p, verts_p = self.smpl_model.forward_kinematics(pose_p_mat, calc_mesh=True)
        pose_global_t, joint_t, verts_t = self.smpl_model.forward_kinematics(pose_t_mat, calc_mesh=True)

        # --- 3. 重新整形回序列维度 ---
        pose_global_p = pose_global_p.view(batch_size, seq_len, 24, 3, 3)
        pose_global_t = pose_global_t.view(batch_size, seq_len, 24, 3, 3)
        joint_p = joint_p.view(batch_size, seq_len, 24, 3)
        joint_t = joint_t.view(batch_size, seq_len, 24, 3)
        verts_p = verts_p.view(batch_size, seq_len, 6890, 3)
        verts_t = verts_t.view(batch_size, seq_len, 6890, 3)

        # --- 4. 计算逐帧误差 ---
        # 3.1 关节位置误差 (PA-MPJPE)
        joint_p_aligned = joint_p - joint_p[:, :, 0:1, :] + joint_t[:, :, 0:1, :]
        pos_err_per_frame = torch.norm(joint_p_aligned - joint_t, dim=3).mean(dim=2) * 100  # [B, S]

        # 3.2 网格顶点误差 (PA-MPVPE)
        verts_p_aligned = verts_p - joint_p[:, :, 0:1, :] + joint_t[:, :, 0:1, :]
        mesh_err_per_frame = torch.norm(verts_p_aligned - verts_t, dim=3).mean(dim=2) * 100  # [B, S]

        # 3.3 全局旋转误差 (MPJAE) - 修改这里
        # 将 [B, S, 24, 3, 3] 重新整形为 [B*S*24, 3, 3] 来计算角度
        pose_global_p_flat = pose_global_p.view(-1, 3, 3)  # [B*S*24, 3, 3]
        pose_global_t_flat = pose_global_t.view(-1, 3, 3)  # [B*S*24, 3, 3]

        # 计算角度误差
        angle_err_flat = art.math.radian_to_degree(
            art.math.angle_between(pose_global_p_flat, pose_global_t_flat))  # [B*S*24]

        # 重新整形回 [B, S, 24] 然后对关节维度求平均
        angle_err_per_frame = angle_err_flat.view(batch_size, seq_len, 24).mean(dim=2)  # [B, S]

        # 3.4 Jitter Error
        if seq_len > 2:
            vel = (joint_p[:, 1:] - joint_p[:, :-1]) * self.fps
            accel = (vel[:, 1:] - vel[:, :-1]) * self.fps
            jitter_err = torch.norm(accel, dim=3).mean(dim=2) / 100  # [B, S-2]
        else:
            jitter_err = torch.empty(batch_size, 0, device=self.device)

        return {
            "pos_err": pos_err_per_frame,
            "mesh_err": mesh_err_per_frame,
            "angle_err": angle_err_per_frame,
            "jitter_err": jitter_err,
        }
