import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

import articulate as art
import config as conf
from model.graph import Graph_B, Graph_J, Graph_P, Graph_A, Unpool


def euler2mat(euler):
    r'''
        euler: [n,t,v,3] => return[n,t,v,3,3]
    '''
    n,t,_ = euler.shape
    euler = euler.view(n,t,15,3)
    cos = torch.cos(euler)  #[n,t,v,3]
    sin = torch.sin(euler)  #[n,t,v,3]
    mat = cos.new_zeros(n,t,15,9)
    
    mat[:,:,:,0] = cos[:,:,:,1] * cos[:,:,:,2]
    mat[:,:,:,1] = -cos[:,:,:,1] * sin[:,:,:,2]
    mat[:,:,:,2] = sin[:,:,:,1]
    mat[:,:,:,3] = cos[:,:,:,0] * sin[:,:,:,2] + cos[:,:,:,2] * sin[:,:,:,0] * sin[:,:,:,1]
    mat[:,:,:,4] = cos[:,:,:,0] * cos[:,:,:,2] - sin[:,:,:,0] * sin[:,:,:,1] * sin[:,:,:,2]
    mat[:,:,:,5] = -cos[:,:,:,1] * sin[:,:,:,0]
    mat[:,:,:,6] = sin[:,:,:,0] * sin[:,:,:,2] - cos[:,:,:,0] * cos[:,:,:,2] * sin[:,:,:,1]
    mat[:,:,:,7] = cos[:,:,:,2] * sin[:,:,:,0] + cos[:,:,:,0] * sin[:,:,:,1] * sin[:,:,:,2]
    mat[:,:,:,8] = cos[:,:,:,0] * cos[:,:,:,1]

    return mat.contiguous()


class s_gcn(nn.Module):
    r'''
        用于输入数据x的gcn
        输入：动态信息x:[n, d(in_channels), t, v]； 邻接矩阵A:[k, v, v(w)]
    '''
    def __init__(self, in_channels, out_channels, k_num):
        super().__init__()

        self.k_num = k_num
        self.lin = nn.Linear(in_channels, out_channels*(k_num))

    def forward(self, x, A_skl):        # x:[n, d(in_channels), t, v]; A:[k, v, v(w)]
        x = x.permute(0,2,3,1)  #[n,t,v,d]
        x = self.lin(x)
        x = x.permute(0,3,1,2)
        
        n, kc, t, v = x.size()                                             # n = 64(batchsize), kc = 128, t = 49, v = 21
        x = x.view(n, self.k_num,  kc//(self.k_num), t, v)             # [64, 4, 32, 49, 21]
        A_all = A_skl
        x = torch.einsum('nkctv, kvw->nctw', (x, A_all))    # [n,c,t,v]
        
        return x

class AGGRU_1(nn.Module):
    r'''
        GCN+GRU网络，输入图时序信息，输出预测结果
    '''
    
    def __init__(self, n_in_dec, n_hid_dec, n_out_dec, strategy='uniform', edge_weighting=True):
        super().__init__()

        self.graphB = Graph_B(strategy=strategy)
        graph_b = torch.tensor(self.graphB.A_b, dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('graph_imu', graph_b)   # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        
        k_num_imu, j_6 = self.graph_imu.size(0), self.graph_imu.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        if edge_weighting:
            self.emul_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
            self.eadd_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
        else:
            self.emul_out = 1
            self.eadd_out = nn.Parameter(torch.ones(self.A_graph_out.size()))
            
        self.imu_gcn = s_gcn(12, 12, k_num_imu)
        
        self.in_fc = torch.nn.Linear(n_in_dec, n_hid_dec)
        self.in_dropout = nn.Dropout(0.2)

        self.gru = nn.GRU(n_hid_dec, n_hid_dec, num_layers=2, bidirectional=True, batch_first=True)        
        self.out_fc = nn.Linear(2 * n_hid_dec, n_hid_dec)
        self.out_reg = nn.Linear(n_hid_dec, n_out_dec)
        

    def forward(self, x, hidden=None):                     
        n, t, d = x.size()  # [n,t,72]
        imu = x.view(n,t,6,12)
        imu_data = imu.permute(0,3,1,2) #[n,d,t,v]
        
        # 使用s-GC模块
        imu_res = imu_data + self.imu_gcn(imu_data, self.graph_imu * self.emul_out + self.eadd_out)
        imu_res = imu_res.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        # 对比消融实验：没有sGC模块
        # imu_res = imu_data.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        
        input = imu_res
        input = self.in_dropout(input)
        input = relu(self.in_fc(input))
        
        result, _ = self.gru(input, hidden) 
        result = input + self.out_fc(result)
        
        output = self.out_reg(result)
        
        return output

class AGGRU_2(nn.Module):
    r'''
        GCN+GRU网络，输入图时序信息，输出预测结果
    '''
    
    def __init__(self, n_in_dec, n_hid_dec, n_out_dec, strategy='uniform', edge_weighting=True):
        super().__init__()

        self.graphB = Graph_B(strategy=strategy)
        graph_b = torch.tensor(self.graphB.A_b, dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('graph_imu', graph_b)   # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        
        k_num_imu, j_6 = self.graph_imu.size(0), self.graph_imu.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        if edge_weighting:
            self.emul_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
            self.eadd_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
        else:
            self.emul_out = 1
            self.eadd_out = nn.Parameter(torch.ones(self.A_graph_out.size()))
            
        self.imu_gcn = s_gcn(15, 15, k_num_imu)
        
        self.in_fc = torch.nn.Linear(n_in_dec, n_hid_dec)
        self.in_dropout = nn.Dropout(0.2)

        self.gru = nn.GRU(n_hid_dec, n_hid_dec, num_layers=2, bidirectional=True, batch_first=True)        
        self.out_fc = nn.Linear(2 * n_hid_dec, n_hid_dec)
        self.out_reg = nn.Linear(n_hid_dec, n_out_dec)
        

    def forward(self, x, hidden=None):                     
        n, t, d = x.size()  # [n,t,90]
        imuAndpos = x.view(n,t,6,15)
        imu_data = imuAndpos.permute(0,3,1,2) #[n,d,t,v]
        
        # 使用s-GC模块
        imu_res = imu_data + self.imu_gcn(imu_data, self.graph_imu * self.emul_out + self.eadd_out)
        imu_res = imu_res.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        # 对比消融实验：没有sGC模块
        # imu_res = imu_data.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        
        input = imu_res
        input = self.in_dropout(input)
        input = relu(self.in_fc(input))
        
        result, _ = self.gru(input, hidden) 
        result = input + self.out_fc(result)
        
        output = self.out_reg(result)
        
        return output


class AGGRU_3(nn.Module):   # GAIP的rnn3，输出结果为矩阵15*9
    r'''
        GCN+GRU网络，输入图时序信息，输出预测结果
    '''
    
    def __init__(self, n_in_dec, n_hid_dec, n_out_dec, strategy='uniform', edge_weighting=True):
        super().__init__()

        self.graphA = Graph_A(strategy=strategy)
        self.graphJ = Graph_J(strategy=strategy)
        graph_a = torch.tensor(self.graphA.A_a, dtype=torch.float32, requires_grad=False)
        graph_j = torch.tensor(self.graphJ.A_j, dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('graph_pos', graph_a)   # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        self.register_buffer('graph_imu', graph_j)   # A_graph 本身不变，通过 emul 进行训练使得 A_graph 变得近似“可训练”
        
        k_num_pos, j_24 = self.graph_pos.size(0), self.graph_pos.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        k_num_imu, j_15 = self.graph_imu.size(0), self.graph_imu.size(1)  # k_num：卷积核的个数（构造邻接矩阵时从不同的特点构造了不止一个矩阵）
        if edge_weighting:
            self.emul_in = nn.Parameter(torch.ones(self.graph_pos.size()))   # [k_num, j_num, j_num]
            self.eadd_in = nn.Parameter(torch.ones(self.graph_pos.size()))   # [k_num, j_num, j_num]
            self.emul_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
            self.eadd_out = nn.Parameter(torch.ones(self.graph_imu.size()))   # [k_num, j_num, j_num]
        else:
            self.emul_in = 1
            self.eadd_in = nn.Parameter(torch.ones(self.A_graph_in.size()))
            self.emul_out = 1
            self.eadd_out = nn.Parameter(torch.ones(self.A_graph_out.size()))

        self.pos_gcn = s_gcn(3, 3, k_num_pos)     # 针对位移的gcn
        self.imu_gcn = s_gcn(12, 12, k_num_imu)

        self.in_fc = torch.nn.Linear(n_in_dec, n_hid_dec)
        self.in_dropout = nn.Dropout(0.2)

        self.gru = nn.GRU(n_hid_dec, n_hid_dec, num_layers=2, bidirectional=True, batch_first=True)        
        self.out_fc = nn.Linear(2 * n_hid_dec, n_hid_dec)
        self.out_reg = nn.Linear(n_hid_dec, n_out_dec)


    def forward(self, x, hidden=None):
        n, t, d = x.size()
        pos = x[:,:,16*12:].view(n,t,24,3)  #[n,t,24*3]
        pos = pos.permute(0,3,1,2)                      # [n,3,t,24]
        imu = x[:,:,:16*12]  #[n,t,3*15+9*15] 需要填充ori和acc
        acc = imu[:,:,:16*3].view(n,t,16,3)
        ori = imu[:,:,16*3:].view(n,t,16,9)
        imu_data = torch.concat((acc,ori), dim=-1)
        imu_data = imu_data.permute(0,3,1,2)            # [n,12,t,15]
        
        
        # 使用s-GC模块
        pos_res = pos + self.pos_gcn(pos, self.graph_pos * self.emul_in + self.eadd_in)
        imu_res = imu_data + self.imu_gcn(imu_data, self.graph_imu * self.emul_out + self.eadd_out)
        pos_res = pos_res.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        imu_res = imu_res.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        # 对比消融实验：没有sGC模块
        # pos_res = pos.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        # imu_res = imu_data.permute(0, 2, 1, 3).contiguous().view(n,t,-1)
        
        input = torch.concat((imu_res, pos_res), dim=-1)
        input = self.in_dropout(input)
        input = relu(self.in_fc(input))
        
        result, _ = self.gru(input, hidden) 
        result = input + self.out_fc(result)
        
        output = self.out_reg(result)
        
        return output


class GGIP(nn.Module):
    def __init__(self, n_hid_dec=256, strategy='uniform', edge_weighting=True):
        super().__init__()
        self.name = 'GGIP'
        
        self.gip1 = AGGRU_1(6*12, n_hid_dec, 5*3)
        self.gip2 = AGGRU_2(6*15, n_hid_dec, 23*3)
        self.gip3 = AGGRU_3(24*3+16*12, n_hid_dec, 15*6, strategy=strategy)    # uniform / spatial
        # self.gip3 = AGGRU_3(24*3+16*12, n_hid_dec, 15*6)
        # self.gip3 = AGGRU_3(24*3+16*12, n_hid_dec, 15*9)
        
        self.smpl_model_func = art.ParametricModel(conf.paths.smpl_file)
        self.global_to_local_pose = self.smpl_model_func.inverse_kinematics_R
        self.loadPretrain()
        self.eval()
        
    def forward(self, x, saperateTrain=True):
        r'''
            要求输入imu的顺序：根、左右脚、头、左右手。SMPL joint order: [0,7,8,12,20,21]
        '''
        n,t,_ = x.shape  # x:[n,t,acc(18)+ori(54)]
        acc = x[:,:,:18].view(n,t,6,3)
        ori = x[:,:,18:].view(n,t,6,9)  # order: 根、左右脚、头、左右手
        
        input1 = torch.cat((acc, ori), -1).view(n,t,-1) #[n,t,6*12]
        output1 = self.gip1(input1)                     #[n,t,15]
        
        p_leaf = output1.view(n,t,5,3)
        p_leaf = torch.cat((p_leaf.new_zeros(n, t, 1, 3), p_leaf), -2)  #[n,t,6,3]
        
        input2 = torch.cat((acc, ori, p_leaf), -1).view(n,t,-1)     #[n,t,6*15]
        if saperateTrain:
            input2_ = input2.detach()
        else:
            input2_ = input2
        output2 = self.gip2(input2_)
        
        p_all = output2.view(n,t,23,3)
        p_all = torch.cat((p_all.new_zeros(n, t, 1, 3), p_all), -2).view(n, t, 72)
        full_acc = acc.new_zeros(n, t, 16, 3)
        full_ori = ori.new_zeros(n, t, 16, 9)
        imu_pos = [0,4,5,11,14,15]  # [14,15,4,5,11,0]左手右手，左腿右腿，头，根节点
        full_acc[:,:,imu_pos] = acc
        full_ori[:,:,imu_pos] = ori
        full_acc = full_acc.view(n,t,-1)
        full_ori = full_ori.view(n,t,-1)
        
        # input3 = torch.concat((p_all, full_acc, full_ori), dim=-1)  # 默认6d模型ggip3的输入顺序
        input3 = torch.concat((full_acc, full_ori, p_all), dim=-1)  #[n,t,24*3+16*12]
        if saperateTrain:
            input3_ = input3.detach()
        else:
            input3_ = input3
        output3 = self.gip3(input3_)                                 #[n,t,90]
        
        return output1, output2, output3
    
    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        batch = glb_reduced_pose.shape[0]
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(batch, -1, conf.joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(batch, glb_reduced_pose.shape[1], 24, 1, 1)
        global_full_pose[:, :, conf.joint_set.reduced] = glb_reduced_pose
        
        pose = global_full_pose.clone().detach()
        for i in range(global_full_pose.shape[0]):
            pose[i] = self.global_to_local_pose(global_full_pose[i]).view(-1, 24, 3, 3) # 到这一步变成了相对父节点的相对坐标
        pose[:, :, conf.joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, :, 0:1] = root_rotation.view(batch, -1, 1, 3, 3)       # 第一个是全局根节点方向
        return pose.contiguous()
  
    def loadPretrain(self, seperate=False):
        if seperate:
            path1 = 'model/weight/seperateTri/Rl_192epoch.pkl'
            path2 = 'model/weight/seperateTri/Ra_80epoch.pkl'
            path3 = 'model/weight/seperateTri/Rp_280epoch.pkl'
            self.gip1.load_state_dict(torch.load(path1)['model_state_dict'])
            self.gip2.load_state_dict(torch.load(path2)['model_state_dict'])
            self.gip3.load_state_dict(torch.load(path3)['model_state_dict'])
        else:
            pathWight = 'model/weight/ggip_all_6d_optloss_spatial.pt'
            self.load_state_dict(torch.load(pathWight)) 
            
    def forwardRaw(self, imu):
        r'''
            要求输入imu的顺序：[n,t,72]
            acc和ori分开输入，acc在前(:18)，ori在后(18:)
            顺序为：关节点顺序为右手左手、右脚左脚、头、根，acc（18）+ori（54），已经经过标准化。(加速度没有除以缩放因子)
        '''
        n,t,_ = imu.shape
        acc = imu[:,:,:18].view(n,t,6,3)
        ori = imu[:,:,18:].view(n,t,6,9)
        
        order = [5,2,3,4,0,1]
        acc = acc[:,:,order]
        ori = ori[:,:,order]
        input = torch.cat((acc.view(n,t,-1), ori.view(n,t,-1)), dim=-1)
        leaf_pos, all_pos, r6dpose = self.forward(input)
        return leaf_pos, all_pos, r6dpose
    
    def ggip3ForwardRaw(self, imu, joint_all):
        r'''
            标准输入：
                acc和ori分开输入，acc在前(:18)，ori在后(18:)
                顺序为：关节点顺序为右手左手、右脚左脚、头、根，acc（18）+ori（54），已经经过标准化。(加速度没有除以缩放因子)
                joint_all就是
        '''
        n,t,_ = imu.shape
        acc = imu[:,:,:18].view(n,t,6,3)
        ori = imu[:,:,18:].view(n,t,6,9)
        
        order = [5,2,3,4,0,1]
        acc = acc[:,:,order]
        ori = ori[:,:,order]
    
        full_acc = acc.new_zeros(n, t, 16, 3)
        full_ori = ori.new_zeros(n, t, 16, 9)
        imu_pos = [0,4,5,11,14,15]  # [14,15,4,5,11,0]左手右手，左腿右腿，头，根节点
        full_acc[:,:,imu_pos] = acc
        full_ori[:,:,imu_pos] = ori
        full_acc = full_acc.view(n,t,-1)
        full_ori = full_ori.view(n,t,-1)
        
        p_all_modify = joint_all.view(n,t,23*3)
        noise = 0.025 * torch.randn(p_all_modify.shape).to(p_all_modify.device).float()   # 为了鲁棒添加的高斯噪声，标准差为0.4
        p_all_noise =  p_all_modify + noise
        p_all = p_all_noise.view(p_all_noise.shape[0], p_all_noise.shape[1], 23, 3)
        p_all = torch.cat((p_all.new_zeros(n, t, 1, 3), p_all), -2).view(n, t, 72)
    
        input = torch.concat((full_acc, full_ori, p_all), dim=-1)
        
        
        pose_6d = self.gip3.forward(input)
        # # pose_mat = self.gip3.forward(input)   # mat(9d) version
        # pose_euler = self.gip3.forward(input)   # euler(3d) version
        # pose_mat = euler2mat(pose_euler)
        
        return pose_6d #[]
    
    def calSMPLpose(self, imu):
        r'''
            要求输入imu的顺序：根、左右脚、头、左右手。SMPL joint order: [0,7,8,12,20,21]
            要求acc和ori分开输入，acc在前(:18)，ori在后(18:)
        '''
        _,_,global_pose = self.forward(imu) # [n,t,15*6=90]
        return global_pose
        
    def calFullJointPos(self, imu):
        r'''
            要求输入imu的顺序：根、左右脚、头、左右手。SMPL joint order: [0,7,8,12,20,21]
            要求acc和ori分开输入，acc在前(:18)，ori在后(18:)
        '''
        _,full_joint_position,_ = self.forward(imu) # [n,t,23*3]
        return full_joint_position
    
    @torch.no_grad()
    def predictPose_single(self, acc, ori, preprocess=False):
        r'''
            acc: [t,6,3]
            ori: [t,6,3,3]
            顺序为：根、左右脚、头、左右手（有预处理） /  右手左手、右脚左脚、头、根（无预处理）
        '''
        if not preprocess:
            order = [2,3,4,0,1,5]
            acc_cal = acc[:,order]
            ori_cal = ori[:,order]
            
            acc_tmp = torch.cat((acc_cal[:, 5:], acc_cal[:, :5] - acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) #/ conf.acc_scale
            ori_tmp = torch.cat((ori_cal[:, 5:], ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5])), dim=1)
        else:
            acc_tmp = acc
            ori_tmp = ori
        
        t = acc_tmp.shape[0]
        root = ori_tmp[:,0].view(t,3,3)
        
        acc_tmp = acc_tmp.view(t,-1)  #[t,18]
        ori_tmp = ori_tmp.view(t,-1)  #[t,54]
        imu = torch.cat((acc_tmp, ori_tmp), dim=-1).unsqueeze(0) #[1,t,72]
        
        leaf_pos, all_pos, glo_pose = self.forward(imu)
        # pose = self._reduced_glb_euler_to_full_local_mat(root, glo_pose)     # euler版本
        # pose = self._reduced_glb_axis_to_full_local_mat(root, glo_pose)     # axis版本
        pose = self._reduced_glb_6d_to_full_local_mat(root, glo_pose)     # r6d版本
        # pose = self._reduced_glb_mat_to_full_local_mat(root, glo_pose)    # 矩阵版本
        return pose.squeeze(0)
