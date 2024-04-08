from torch.nn.utils.rnn import *
import articulate as art
from articulate.utils.torch import *
from config import *
from model.ref_pip_utils import *
# from model.dynamics import PhysicsOptimizer


class PIP(torch.nn.Module):
    name = 'PIP'
    n_hidden = 256

    def __init__(self, device=None):
        super(PIP, self).__init__()
        self.rnn1 = RNNWithInit(input_size=72,
                                output_size=joint_set.n_leaf * 3,
                                hidden_size=self.n_hidden,
                                num_rnn_layer=2,
                                dropout=0.4)
        self.rnn2 = RNN(input_size=72 + joint_set.n_leaf * 3,
                        output_size=joint_set.n_full * 3,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn3 = RNN(input_size=72 + joint_set.n_full * 3,
                        output_size=joint_set.n_reduced * 6,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn4 = RNNWithInit(input_size=72 + joint_set.n_full * 3,
                                output_size=24 * 3,
                                hidden_size=self.n_hidden,
                                num_rnn_layer=2,
                                dropout=0.4)
        self.rnn5 = RNN(input_size=72 + joint_set.n_full * 3,
                        output_size=2,
                        hidden_size=64,
                        num_rnn_layer=2,
                        dropout=0.4)

        body_model = art.ParametricModel(paths.smpl_file)
        self.inverse_kinematics_R = body_model.inverse_kinematics_R
        self.forward_kinematics = body_model.forward_kinematics
        # self.dynamics_optimizer = PhysicsOptimizer(debug=False)
        self.rnn_states = [None for _ in range(5)]
        self.device = device

        # self.load_state_dict(torch.load(paths.weights_file))
        self.eval()

    # def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
    #     glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
    #     global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
    #     global_full_pose[:, joint_set.reduced] = glb_reduced_pose
    #     pose = self.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
    #     pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
    #     pose[:, 0] = root_rotation.view(-1, 3, 3)
    #     return pose
    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        batch = glb_reduced_pose.shape[0]
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(batch, -1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(batch, glb_reduced_pose.shape[1], 24, 1, 1)
        # TODO:when debuging next time, try this.
        # global_full_pose[:, 0] = root_rotation.view(-1, 3, 3)
        global_full_pose[:, :, joint_set.reduced] = glb_reduced_pose
        # 到这里为止，global_full_pose的15个关节点是原本的6D转成了矩阵，剩下的都是单位矩阵。都是Joint global rotation
        # 全局坐标是相对根节点？
        
        pose = global_full_pose.clone().detach()
        for i in range(global_full_pose.shape[0]):
            pose[i] = self.inverse_kinematics_R(global_full_pose[i]).view(-1, 24, 3, 3) # 到这一步变成了相对父节点的相对坐标
        pose[:, :, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, :, 0:1] = root_rotation.view(batch, -1, 1, 3, 3)       # 第一个是全局根节点方向
        return pose.contiguous()


    def forward(self, x, lj_init):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 3-tuple
                  (tensor [num_frames, 72], tensor [15]).
        """
        # x, lj_init = list(zip(*x))
        leaf_joint = self.rnn1(list(zip(x, lj_init)))
        full_joint = self.rnn2([torch.cat(_, dim=-1) for _ in zip(leaf_joint, x)])
        global_6d_pose = self.rnn3([torch.cat(_, dim=-1) for _ in zip(full_joint, x)])
        # joint_velocity = self.rnn4(list(zip([torch.cat(_, dim=-1) for _ in zip(full_joint, x)], jvel_init)))
        # contact = self.rnn5([torch.cat(_, dim=-1) for _ in zip(full_joint, x)])
        
        return leaf_joint, full_joint, global_6d_pose#, joint_velocity, contact
    
    def calSMPLpose(self, imu, ljpos):
        if acc_scale:
            n,t,_ = imu.shape
            acc = imu[:,:,:18].view(n,t,6,3)
            ori = imu[:,:,18:].view(n,t,6,9)
            acc = acc
            imu = torch.cat((acc.view(n,t,-1), ori.view(n,t,-1)), dim=-1)   
        
        leaf_pos, all_pos, global_pose = self.forward(imu, ljpos) # [n,t,72], [n,15]
        return torch.stack(leaf_pos), torch.stack(all_pos), torch.stack(global_pose)
    
    @torch.no_grad()
    def calSMPLpose_eval(self, imu, init_pose): # [t,90]
        init_p = torch.zeros((1, 24, 3, 3)).to(self.device)
        for i in range(24):
            init_p[:,i] = torch.eye(3).to(self.device)
        if init_pose is not None:
            # self.dynamics_optimizer.reset_states()
            init_pose = init_pose[0:1].view(1, 15, 6)
            p = art.math.r6d_to_rotation_matrix(init_pose).view(1, 15, 3, 3)
            init_p[:,joint_set.reduced] = p
        lj_init = self.forward_kinematics(init_p)[1][0, joint_set.leaf].view(1,-1)
        return self.calSMPLpose(imu, lj_init)


    # @torch.no_grad()
    # def predict(self, glb_acc, glb_rot, init_pose):
        r"""
        Predict the results for evaluation.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """
        self.dynamics_optimizer.reset_states()
        init_pose = init_pose.view(1, 24, 3, 3)
        init_pose[0, 0] = torch.eye(3)
        lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
        jvel_init = torch.zeros(24 * 3)
        x = (normalize_and_concat(glb_acc, glb_rot), lj_init, jvel_init)
        leaf_joint, full_joint, global_6d_pose, joint_velocity, contact = [_[0] for _ in self.forward([x])]
        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) * vel_scale
        pose_opt, tran_opt = [], []
        for p, v, c, a in zip(pose, joint_velocity, contact, glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c, a)
            pose_opt.append(p)
            tran_opt.append(t)
            
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        return pose_opt#, tran_opt