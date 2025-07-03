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
