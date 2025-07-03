r"""
    SMPL/MANO/SMPLH parametric model. Modified from https://github.com/CalciferZh/SMPL.
"""


__all__ = ['ParametricModel']


import os
import pickle
import torch
import numpy as np
from . import math as M


class ParametricModel:
    r"""
    SMPL/MANO/SMPLH parametric model.

    ParametricModel 类封装了加载模型数据、进行前向运动学计算（从姿态和形状参数生成三维网格和关节位置）、
    以及一些辅助功能（如保存和可视化）的核心逻辑
    """
    def __init__(self, official_model_file: str, use_pose_blendshape=False, device=torch.device('cpu')):
        r"""
        Init an SMPL/MANO/SMPLH parametric model.

        :param official_model_file: Path to the official model to be loaded.
        :param use_pose_blendshape: Whether to use the pose blendshape.
        :param device: torch.device, cpu or cuda.

        _J_regressor (torch.Tensor): 关节回归器、_skinning_weights (torch.Tensor): 蒙皮权重、
        _posedirs (torch.Tensor): 姿态混合形基（Pose Blendshape Basis）、_shapedirs (torch.Tensor): 形状混合形基（Shape Blendshape Basis）
        _v_template (torch.Tensor): 模板顶点（Template Vertices）、_J (torch.Tensor): 模板关节位置（Template Joint Positions）
        faceface (numpy.ndarray): 面（Faces）、parent (list): 骨骼父子关系（Kinematic Tree Table
        """
        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self._J_regressor = torch.from_numpy(data['J_regressor'].toarray()).float().to(device)
        self._skinning_weights = torch.from_numpy(data['weights']).float().to(device)
        self._posedirs = torch.from_numpy(data['posedirs']).float().to(device)
        self._shapedirs = torch.from_numpy(np.array(data['shapedirs'])).float().to(device)
        self._v_template = torch.from_numpy(data['v_template']).float().to(device)
        self._J = torch.from_numpy(data['J']).float().to(device)
        self.face = data['f']
        self.parent = data['kintree_table'][0].tolist()
        self.parent[0] = None
        self.use_pose_blendshape = use_pose_blendshape

    def save_obj_mesh(self, vertex_position, file_name='a.obj'):
        r"""
        Export an obj mesh using the input vertex position.

        :param vertex_position: Vertex position in shape [num_vertex, 3].
        :param file_name: Output obj file name.

        三维网格顶点保存为标准的 OBJ 文件格式，方便在其他三维软件中查看
        """
        with open(file_name, 'w') as fp:
            for v in vertex_position:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.face + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    @staticmethod
    def save_unity_motion(pose: torch.Tensor = None, tran: torch.Tensor = None, output_dir='saved_motions/'):
        r"""
        Write motions into text files used by Unity3D `LoadMotion()`.

        :param pose: Pose tensor in shape [num_frames, *] that can reshape to [num_frame, num_joint, 3, 3]
                    (rotation matrices).
        :param tran: Translation tensor in shape [num_frames, 3] for root positions.
        :param output_dir: Output directory path.

        将姿态（旋转矩阵）和根节点平移数据转换为 Unity3D LoadMotion() 函数可读取的文本文件格式
        """
        os.makedirs(output_dir, exist_ok=True)

        if pose is not None:
            f = open(os.path.join(output_dir, 'pose.txt'), 'w')
            pose = M.rotation_matrix_to_axis_angle(pose).view(pose.shape[0], -1)
            f.write('\n'.join([','.join(['%.4f' % _ for _ in p]) for p in pose]))
            f.close()

        if tran is not None:
            f = open(os.path.join(output_dir, 'tran.txt'), 'w')
            f.write('\n'.join([','.join(['%.5f' % _ for _ in t]) for t in tran.view(tran.shape[0], 3)]))
            f.close()

    def get_zero_pose_joint_and_vertex(self, shape: torch.Tensor = None):
        r"""
        Get the joint and vertex positions in zero pose. Root joint is aligned at zero.

        :param shape: Tensor for model shapes that can reshape to [batch_size, 10]. Use None for the mean(zero) shape.
        :return: Joint tensor in shape [batch_size, num_joint, 3] and vertex tensor in shape [batch_size, num_vertex, 3]
                 if shape is not None. Otherwise [num_joint, 3] and [num_vertex, 3] assuming the mean(zero) shape.

        计算在“零姿态”（T-pose，即所有关节旋转为零）下的关节和顶点位置，并将根关节对齐到坐标原点
        """
        if shape is None:
            j, v = self._J - self._J[:1], self._v_template - self._J[:1]
        else:
            shape = shape.view(-1, 10).to(self._shapedirs.device)
            v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template
            j = torch.matmul(self._J_regressor, v)
            j, v = j - j[:, :1], v - j[:, :1]
        return j, v

    def bone_vector_to_joint_position(self, bone_vec: torch.Tensor):
        r"""
        Calculate joint positions in the base frame from bone vectors (position difference of child and parent joint)
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param bone_vec: Bone vector tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Joint position, in shape [batch_size, num_joint, 3].

        “骨骼向量”（子关节相对于父关节的位移向量）和“关节绝对位置”之间进行转换
        """
        return M.bone_vector_to_joint_position(bone_vec, self.parent)



    #前向运动学:居局->全部  反向运动学:全部->居局
    def joint_position_to_bone_vector(self, joint_pos: torch.Tensor):
        r"""
        Calculate bone vectors (position difference of child and parent joint) in the base frame from joint positions
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param joint_pos: Joint position tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Bone vector, in shape [batch_size, num_joint, 3].
        """
        return M.joint_position_to_bone_vector(joint_pos, self.parent)

    def forward_kinematics_R(self, R_local: torch.Tensor):
        r"""
        :math:`R_global = FK(R_local)`

        Forward kinematics that computes the global rotation of each joint from local rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint global rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return M.forward_kinematics_R(R_local, self.parent)

    def inverse_kinematics_R(self, R_global: torch.Tensor):
        r"""
        :math:`R_local = IK(R_global)`

        Inverse kinematics that computes the local rotation of each joint from global rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_global: Joint global rotation tensor in shape [batch_size, *] that can reshape to
                         [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint local rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return M.inverse_kinematics_R(R_global, self.parent)

    def forward_kinematics_T(self, T_local: torch.Tensor):
        r"""
        :math:`T_global = FK(T_local)`

        Forward kinematics that computes the global homogeneous transformation of each joint from
        local homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_local: Joint local transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint global transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return M.forward_kinematics_T(T_local, self.parent)

    def inverse_kinematics_T(self, T_global: torch.Tensor):
        r"""
        :math:`T_local = IK(T_global)`

        Inverse kinematics that computes the local homogeneous transformation of each joint from
        global homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_global: Joint global transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint local transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return M.inverse_kinematics_T(T_global, self.parent)

    def forward_kinematics(self, pose: torch.Tensor, shape: torch.Tensor = None, tran: torch.Tensor = None,
                           calc_mesh=False):
        r"""
        Forward kinematics that computes the global joint rotation, joint position, and additionally
        mesh vertex position from poses, shapes, and translations. (torch, batch)

        :param pose: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                     [batch_size, num_joint, 3, 3] (rotation matrices).
        :param shape: Tensor for model shapes that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran: Root position tensor in shape [batch_size, 3]. Use None for the zero positions.
        :param calc_mesh: Whether to calculate mesh vertex positions.
        :return: Joint global rotation in [batch_size, num_joint, 3, 3],
                 joint position in [batch_size, num_joint, 3],
                 and additionally mesh vertex position in [batch_size, num_vertex, 3] if calc_mesh is True.

        根据输入的姿态、形状和根平移参数，计算出最终的全局关节旋转、全局关节位置以及可选的网格顶点位置
        """
        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        pose = pose.view(pose.shape[0], -1, 3, 3)
        j, v = [_.expand(pose.shape[0], -1, -1) for _ in self.get_zero_pose_joint_and_vertex(shape)]
        T_local = M.transformation_matrix(pose, self.joint_position_to_bone_vector(j).to(pose.device))
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = M.decode_transformation_matrix(T_global)
        if calc_mesh is False:
            return pose_global, add_tran(joint_global)

        T_global[..., -1:] -= torch.matmul(T_global, M.append_zero(j, dim=-1).unsqueeze(-1))
        T_vertex = torch.tensordot(T_global, self._skinning_weights, dims=([1], [1])).permute(0, 3, 1, 2)
        if self.use_pose_blendshape:
            r = (pose[:, 1:] - torch.eye(3, device=pose.device)).flatten(1)
            v = v + torch.tensordot(r, self._posedirs, dims=([1], [2]))
        vertex_global = torch.matmul(T_vertex, M.append_one(v, dim=-1).unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, add_tran(joint_global), add_tran(vertex_global)

    def view_joint(self, joint_list: list, fps=60, distance_between_subjects=0.8):
        r"""
        View model joint (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param joint_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joint, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        import vctoolkit as vc
        import vctoolkit.viso3d as vo3d
        joint_list = [(j.view(-1, len(self.parent), 3) - j.view(-1, len(self.parent), 3)[:1, :1]).cpu().numpy()
                      for j in joint_list]

        v_list, f_list = [], []
        f = vc.joints_to_mesh(joint_list[0][0], self.parent)[1]
        for i in range(len(joint_list)):
            v = np.stack([vc.joints_to_mesh(frame, self.parent)[0] for frame in joint_list[i]])
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = np.concatenate(v_list, axis=1)
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.avi', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)


    #可视化辅助函数
    def view_mesh(self, vertex_list: list, fps=60, distance_between_subjects=0.8):
        r"""
        View model mesh (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param vertex_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_vertex, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        import vctoolkit.viso3d as vo3d
        v_list, f_list = [], []
        f = self.face.copy()
        for i in range(len(vertex_list)):
            v = vertex_list[i].clone().view(-1, self._v_template.shape[0], 3)
            v[:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = torch.cat(v_list, dim=1).cpu().numpy()
        faces = np.concatenate(f_list)
        if verts.shape[0] > 1:
            vo3d.render_sequence_3d(verts, faces, 720, 720, 'a.avi', fps, visible=True)
        else:
            vo3d.vis_mesh(verts[0], faces)

    def view_motion(self, pose_list: list, tran_list: list = None, fps=60, distance_between_subjects=0.8):
        r"""
        View model motion (poses and translations) (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param pose_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joint, 3, 3].
        :param tran_list: List in length [num_subject] of tensors that can all reshape to [num_frame, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        verts = []
        for i in range(len(pose_list)):
            pose = pose_list[i].view(-1, len(self.parent), 3, 3)
            tran = tran_list[i].view(-1, 3) - tran_list[i].view(-1, 3)[:1] if tran_list else None
            verts.append(self.forward_kinematics(pose, tran=tran, calc_mesh=True)[2])
        self.view_mesh(verts, fps, distance_between_subjects=distance_between_subjects)
