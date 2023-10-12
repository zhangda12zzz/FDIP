import sys
import torch
import numpy as np

import data.bvhProcess.BVH_mod as BVH
from articulate.math.Quaternions import Quaternions
from option_parser import get_std_bvh
from data.bvhProcess.skeleton import build_edge_topology
from data.bvhProcess.bvh_writer import write_bvh
from data.bvhProcess.Kinematics import ForwardKinematics

"""
1.
Specify the joints that you want to use in training and test. Other joints will be discarded.
Please start with root joint, then left leg chain, right leg chain, head chain, left shoulder chain and right shoulder chain.
See the examples below.
指定要在训练和测试中使用的关节。 其他关节将被丢弃。
请从根关节开始，然后是左腿链、右腿链、头链、左肩链和右肩链。
请参阅下面的示例。
"""
corps_name_1 = ['Hips', 
                'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 
                'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 
                'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 
                'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
# corps_name_1 = ['Pelvis', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_3 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_boss = ['Hips', 
                   'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 
                   'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 
                   'Spine', 'Spine1', 'Spine2', 
                   'Neck', 'Neck1', 'Head', 
                   'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 
                   'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
# mixamo导出的PN3Robot就是corps_name_boss模型
corps_name_boss2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'Left_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Right_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_cmu = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_monkey = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_three_arms = ['Three_Arms_Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_three_arms_split = ['Three_Arms_split_Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHand_split', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHand_split']
corps_name_Prisoner = ['HipsPrisoner', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm']
corps_name_mixamo2_m = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine1_split', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftShoulder_split', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightShoulder_split', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_smpl = ['m_avg_Pelvis', 
                 'm_avg_L_Hip', 'm_avg_L_Knee', 'm_avg_L_Ankle', 'm_avg_L_Foot', 
                 'm_avg_R_Hip', 'm_avg_R_Knee', 'm_avg_R_Ankle', 'm_avg_R_Foot', 
                 'm_avg_Spine1', 'm_avg_Spine2', 'm_avg_Spine3',
                 'm_avg_Neck', 'm_avg_Head',
                 'm_avg_L_Collar', 'm_avg_L_Shoulder', 'm_avg_L_Elbow', 'm_avg_L_Wrist',
                 'm_avg_R_Collar', 'm_avg_R_Shoulder', 'm_avg_R_Elbow', 'm_avg_R_Wrist']    # 采用smpl简化的22个关节
corps_name_pn3 = ['Hips', 
               'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
               'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
               'Spine', 'Spine1', 'Spine2',
               'Neck', 'Head',
               'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandMiddle1',
               'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandMiddle1']   # 22
# axis导出的PN3Robot才是pn3模型
# corps_name_pn3 = ['Hips', 
#                'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End',
#                'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End',
#                'Spine', 'Spine1', 'Spine2',
#                'Neck', 'Neck1', 'Head', 'Head_End',
#                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 
#                'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']   # 26


"""
2.
Specify five end effectors' name.
Please follow the same order as in 1.
指定五个末端执行器的名称。
请遵循与 1 中相同的顺序。
"""
ee_name_1 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_2 = ['LeftToe_End', 'RightToe_End', 'HeadTop_End', 'LeftHand', 'RightHand']
ee_name_3 = ['LeftFoot', 'RightFoot', 'Head', 'LeftHand', 'RightHand']
ee_name_cmu = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_monkey = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_three_arms_split = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand_split', 'RightHand_split']
ee_name_Prisoner = ['LeftToe_End', 'RightToe_End', 'HeadTop_End', 'LeftHand', 'RightForeArm']
# ee_name_example = ['LeftToe', 'RightToe', 'Head', 'LeftHand', 'RightHand']
ee_name_smpl = ['m_avg_L_Foot', 'm_avg_R_Foot', 'm_avg_Head', 'm_avg_L_Wrist', 'm_avg_R_Wrist']
ee_name_pn3 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHandMiddle1', 'RightHandMiddle1']
# ee_name_pn3 = ['LeftToe_End', 'RightToe_End', 'Head_End', 'LeftHand', 'RightHand']



corps_names = [corps_name_1, corps_name_2, corps_name_3, corps_name_cmu, corps_name_monkey, corps_name_boss,
               corps_name_boss, corps_name_three_arms, corps_name_three_arms_split, corps_name_Prisoner, corps_name_mixamo2_m, corps_name_smpl, corps_name_pn3]
ee_names = [ee_name_1, ee_name_2, ee_name_3, ee_name_cmu, ee_name_monkey, ee_name_1, ee_name_1, ee_name_1, ee_name_three_arms_split, ee_name_Prisoner, ee_name_2, ee_name_smpl, ee_name_pn3]
"""
3.
Add previously added corps_name and ee_name at the end of the two above lists.
在上面两个列表的末尾加上你前面自己写的 corps_name 和 ee_name。
"""
# corps_names.append(corps_name_example)
# ee_names.append(ee_name_example)


class BVH_file:
    r'''
    bvh文件读取类，并确认当前bvh的人体骨骼是哪一种预先定义的人体骨骼类型
    可以据此进行拓扑计算、设置根节点等和骨架相关的操作
    '''
    def __init__(self, file_path=None, args=None, dataset=None, new_root=None):
        if file_path is None:
            file_path = get_std_bvh(dataset=dataset)
        self.anim, self._names, self.frametime = BVH.load(file_path)    # 动画（包括关节方向和位置、父节点），关节名称，每帧时刻
        if new_root is not None:
            self.set_new_root(new_root)
        self.skeleton_type = -1         # 骨骼类型
        self.edges = []                 # 【父关节 - 子关节 - 偏移量】的List
        self.edge_mat = []              # 未赋值
        self.edge_num = 0               # 未赋值
        self._topology = None           # 骨骼拓扑，通过self.topology调用，序号i对应的值为关节i父节点的序号
        self.ee_length = []             # 末端关节的长度（会归一化为身高）

        for i, name in enumerate(self._names):
            if ':' in name:
                name = name[name.find(':') + 1:]
                self._names[i] = name

        full_fill = [1] * len(corps_names)
        for i, ref_names in enumerate(corps_names):
            for ref_name in ref_names:
                if ref_name not in self._names:
                    full_fill[i] = 0
                    break

        if full_fill[3]:
            self.skeleton_type = 3
        else:
            for i, _ in enumerate(full_fill):
                if full_fill[i]:
                    self.skeleton_type = i
                    break

        if self.skeleton_type == 2 and full_fill[4]:
            self.skeleton_type = 4

        if 'Neck1' in self._names:
            self.skeleton_type = 5
        if 'Left_End' in self._names:
            self.skeleton_type = 6
        if 'Three_Arms_Hips' in self._names:
            self.skeleton_type = 7
        if 'Three_Arms_Hips_split' in self._names:
            self.skeleton_type = 8

        if 'LHipJoint' in self._names:
            self.skeleton_type = 3

        if 'HipsPrisoner' in self._names:
            self.skeleton_type = 9

        if 'Spine1_split' in self._names:
            self.skeleton_type = 10

        """
        4. 
        Here, you need to assign self.skeleton_type the corresponding index of your own dataset in corps_names or ee_names list.
        You can use self._names, which contains the joints name in original bvh file, to write your own if statement.
        在这里，您需要为 self.skeleton_type 分配自己的数据集在 corps_names 或 ee_names 列表中的相应索引。
        你可以使用包含原始 bvh 文件中的关节名称的 self._names 来编写您自己的 if 语句。
        """
        # if ...:
        #     self.skeleton_type = 11
        if 'm_avg_Pelvis' in self._names:
            self.skeleton_type = 11
        if 'LeftHandMiddle1' in self._names:
            self.skeleton_type = 12

        if self.skeleton_type == -1:
            print(self._names)
            raise Exception('Unknown skeleton')


        if self.skeleton_type == 11:    #SMPL
            self.delete_root_4dataset()     # 做imu和pose的数据集用的
            # self.delete_root()  # mixamo need to do this, but my data don't need
            # self.modify_smpl_root() # 24->22
            # print("smpl")
        if self.skeleton_type == 12:    #PN3Robot mixamo
            # self.remove_neck1()
            pass
        if self.skeleton_type == 0:     #PN3Robot from axis
            pass

        self.details = [i for i, name in enumerate(self._names) if name not in corps_names[self.skeleton_type]]
        self.joint_num = self.anim.shape[1]
        self.corps = []                         # 要使用的、动画中存在的关节名称
        self.simplified_name = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        for name in corps_names[self.skeleton_type]:
            for j in range(self.anim.shape[1]):
                if name == self._names[j]:
                    self.corps.append(j)
                    break

        if len(self.corps) != len(corps_names[self.skeleton_type]):
            for i in self.corps: print(self._names[i], end=' ')
            print(self.corps, self.skeleton_type, len(self.corps), sep='\n')
            raise Exception('Problem in file', file_path)

        self.ee_id = []                                 # 末端关节序号
        for i in ee_names[self.skeleton_type]:      # 对于skeleton_type类型骨骼的末端关节而言
            self.ee_id.append(corps_names[self.skeleton_type].index(i))

        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1
        for i in range(self.anim.shape[1]):
            if i in self.details:
                self.simplify_map[i] = -1

        self.edges = build_edge_topology(self.topology, self.offset)    # 【父关节 - 子关节 - 偏移量】的List
        # self.characterHeight = self.get_height()
        # print(self.characterHeight) 

    def scale(self, alpha):
        self.anim.offsets *= alpha
        global_position = self.anim.positions[:, 0, :]
        global_position[1:, :] *= alpha
        global_position[1:, :] += (1 - alpha) * global_position[0, :]

    def rotate(self, theta, axis):
        q = Quaternions(np.hstack((np.cos(theta/2), np.sin(theta/2) * axis)))
        position = self.anim.positions[:, 0, :].copy()
        rotation = self.anim.rotations[:, 0, :]
        position[1:, ...] -= position[0:-1, ...]
        q_position = Quaternions(np.hstack((np.zeros((position.shape[0], 1)), position)))
        q_rotation = Quaternions.from_euler(np.radians(rotation))
        q_rotation = q * q_rotation
        q_position = q * q_position * (-q)
        self.anim.rotations[:, 0, :] = np.degrees(q_rotation.euler())
        position = q_position.imaginaries
        for i in range(1, position.shape[0]):
            position[i] += position[i-1]
        self.anim.positions[:, 0, :] = position

    @property
    def topology(self):
        r'''
        构建骨骼拓扑，调用self.topology时返回的其实是计算过的self._topology
        '''
        if self._topology is None:
            self._topology = self.anim.parents[self.corps].copy()
            for i in range(self._topology.shape[0]):
                if i >= 1: self._topology[i] = self.simplify_map[self._topology[i]]
            self._topology = tuple(self._topology)
        return self._topology

    def get_ee_id(self):
        return self.ee_id

    def to_numpy(self, quater=False, edge=True):
        r'''
        
        '''
        rotations = self.anim.rotations[:, self.corps, :]   # 取所有关节的旋转[frame, num_joint, 3]
        if quater:      # 如果是四元数表示
            rotations = Quaternions.from_euler(np.radians(rotations)).qs    # 欧拉角->四元数,[frame, num_joint, 4]
            positions = self.anim.positions[:, 0, :]    # 只取根节点位置
        else:
            positions = self.anim.positions[:, 0, :]
        if edge:        # 如果按照边来表示旋转
            index = []
            for e in self.edges:    
                index.append(e[0])  # 取每条边的起始点（父节点）序号
            rotations = rotations[:, index, :]  # 将父节点序号的方向作为边的方向（边相对于关节点、维度-1）

        rotations = rotations.reshape(rotations.shape[0], -1)

        return np.concatenate((rotations, positions), axis=1)

    def to_tensor(self, quater=False, edge=True):
        r'''
        将to_numpy输出的【rotations, position】的[frame_num, (joint_num-1)*4+3]维度的数组
        转换为[(joint_num-1)*4+3, frame_num]维度的tensor
        '''
        res = self.to_numpy(quater, edge)
        res = torch.tensor(res, dtype=torch.float)
        res = res.permute(1, 0)     
        res = res.reshape((-1, res.shape[-1]))
        return res

    def get_position(self):
        positions = self.anim.positions
        positions = positions[:, self.corps, :]
        return positions

    @property
    def offset(self):
        r'''
        返回骨骼self.corps的self.anim.offsets，含义通bvh文件，是和父节点的位移差
        '''
        return self.anim.offsets[self.corps]

    @property
    def names(self):
        return self.simplified_name

    def get_height(self):
        offset = self.offset
        topo = self.topology

        # 脚 -> 根节点的距离
        res = 0
        p = self.ee_id[0]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        # 头 -> 根节点的距离
        p = self.ee_id[2]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        # 求和为身高
        return res

    def write(self, file_path):
        motion = self.to_numpy(quater=False, edge=False)
        rotations = motion[..., :-3].reshape(motion.shape[0], -1, 3)
        positions = motion[..., -3:]
        write_bvh(self.topology, self.offset, rotations, positions, self.names, 1.0/30, 'xyz', file_path)

    def get_ee_length(self):
        if len(self.ee_length): return self.ee_length
        degree = [0] * len(self.topology)
        for i in self.topology:     # 对于所有关节的父节点而言
            if i < 0: continue      # 如果是根关节的父节点（-1），跳过
            degree[i] += 1          # 否则对应父节点数据+1，degree对应的应该是“入度”（由子节点指向父节点的话）

        for j in self.ee_id:        # 对于末端效应关节
            length = 0
            while degree[j] <= 1:   # 如果关节入度<=1（即不是其他关节的父节点、或只有1个子节点）
                t = self.offset[j]              # 记录offset
                length += np.dot(t, t) ** 0.5   # 求offset的模长（L2损失开方）
                j = self.topology[j]            # 将j转移为其父节点

            self.ee_length.append(length)   # 到这里self.ee_length包含的是 四肢+头部的长度+胸腹距离（入度>1的点就是胸腔&臀部根节点）

        height = self.get_height()          # 计算人体身高
        ee_group = [[0, 1], [2], [3, 4]]    # 划分为 双脚、头、双手
        for group in ee_group:
            maxv = 0
            for j in group:
                maxv = max(maxv, self.ee_length[j]) # 组内最长距离
            for j in group:
                self.ee_length[j] *= height / maxv  # TODO:我不理解?算出来的结果都是height？

        return self.ee_length

    def set_new_root(self, new_root):
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[new_root], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        self.anim.offsets[0] = -self.anim.offsets[new_root]
        self.anim.offsets[new_root] = np.zeros((3, ))
        self.anim.positions[:, new_root, :] = new_pos
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, new_root, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_rot0 = (-rot1)
        new_rot0 = np.degrees(new_rot0.euler())
        new_rot1 = np.degrees(new_rot1.euler())
        self.anim.rotations[:, 0, :] = new_rot0
        self.anim.rotations[:, new_root, :] = new_rot1

        new_seq = []
        vis = [0] * self.anim.rotations.shape[1]
        new_idx = [-1] * len(vis)
        new_parent = [0] * len(vis)

        def relabel(x):
            nonlocal new_seq, vis, new_idx, new_parent
            new_idx[x] = len(new_seq)
            new_seq.append(x)
            vis[x] = 1
            for y in range(len(vis)):
                if not vis[y] and (self.anim.parents[x] == y or self.anim.parents[y] == x):
                    relabel(y)
                    new_parent[new_idx[y]] = new_idx[x]

        relabel(new_root)
        self.anim.rotations = self.anim.rotations[:, new_seq, :]
        self.anim.offsets = self.anim.offsets[new_seq]
        names = self._names.copy()
        for i, j in enumerate(new_seq):
            self._names[i] = names[j]
        self.anim.parents = np.array(new_parent, dtype=np.int)


    def delete_root(self):
        r'''
            针对mixamo导出的SMPL模型，25个关节 - > 22个关节，删除根节点的同时，删掉两个手部关节。
        '''

        # cal new root pos
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[1], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_root_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        new_root_off = np.zeros((3, ))
        # cal new root rot
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 1, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_root_rot = np.degrees(new_rot1.euler())

        rot_index = [0,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,20,21, 22,23]
        rot_index_noroot = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23]


        rot = self.anim.rotations   # delete node 1 
        self.anim.rotations = rot[:,rot_index]
        self.anim.rotations[:,0] = new_root_rot

        pos = self.anim.positions
        # new_root_pos = pos[:,0]+pos[:,1]
        self.anim.positions = pos[:,rot_index]
        self.anim.positions[:,0] = new_root_pos

        off = self.anim.offsets
        # new_root_off = off[0]+off[1]
        self.anim.offsets = off[rot_index]
        self.anim.offsets[0] = new_root_off

        ori = self.anim.orients
        self.anim.orients = ori[rot_index]

        parents = self.anim.parents
        parents = parents - 1
        parents[21:] -= 1
        self.anim.parents = parents[rot_index_noroot]

        self._names = list(np.array(self._names)[rot_index_noroot])

    def modify_smpl_root(self):
        r'''
            
        '''

        # cal new root pos
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[1], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_root_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        new_root_off = np.zeros((3, ))
        # cal new root rot
        # rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        # rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 1, :]), order='xyz')
        # new_rot1 = rot0 * rot1
        # new_root_rot = np.degrees(new_rot1.euler())

        rot_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]
        rot_index_noroot = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23]


        rot = self.anim.rotations   # delete node 1 
        self.anim.rotations = rot[:,rot_index]
        # self.anim.rotations[:,0] = new_root_rot

        pos = self.anim.positions
        # new_root_pos = pos[:,0]+pos[:,1]
        self.anim.positions = pos[:,rot_index]
        self.anim.positions[:,0] = new_root_pos

        off = self.anim.offsets
        # new_root_off = off[0]+off[1]
        self.anim.offsets = off[rot_index]
        self.anim.offsets[0] = new_root_off

        ori = self.anim.orients
        self.anim.orients = ori[rot_index]

        parents = self.anim.parents
        # parents = parents - 1
        parents[20:] -= 1
        self.anim.parents = parents[rot_index]
        # self.anim.parents = parents[rot_index_noroot]

        self._names = list(np.array(self._names)[rot_index])

    def delete_root_4dataset(self):
        r'''
            25个关节点到24个关节的预处理，只删除根节点。
        '''
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[1], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_root_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        new_root_off = np.zeros((3, ))
        # cal new root rot
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 1, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_root_rot = np.degrees(new_rot1.euler())

        rot_index = [0,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,19,20, 21,22,23,24]
        rot_index_noroot = [1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,19,20, 21,22,23,24]


        rot = self.anim.rotations   # delete node 1 
        self.anim.rotations = rot[:,rot_index]
        self.anim.rotations[:,0] = new_root_rot

        pos = self.anim.positions
        # new_root_pos = pos[:,0]+pos[:,1]
        self.anim.positions = pos[:,rot_index]
        self.anim.positions[:,0] = new_root_pos

        off = self.anim.offsets
        # new_root_off = off[0]+off[1]
        self.anim.offsets = off[rot_index]
        self.anim.offsets[0] = new_root_off

        ori = self.anim.orients
        self.anim.orients = ori[rot_index]

        parents = self.anim.parents
        parents = parents - 1
        parents[21:] -= 1
        self.anim.parents = parents[rot_index_noroot]

        self._names = list(np.array(self._names)[rot_index_noroot])

    def remove_neck1(self):
        offsets = self.anim.offsets
        positions = self.anim.positions
        rotations = self.anim.rotations
        parents = self.anim.parents
        orients = self.anim.orients
        
        # =====需求1:
        # 这几个offset和导出数据不符：根节点、左右大腿、左右肩膀
        # 所以根节点的positions也要改
        
        # 修改 根、左右大腿、左右肩膀、头的offset
        offsets[0, :] += [0, 104.348106 - 97.120201, 0] # 根
        offsets[1, :] += [0, -7.228100, 0]              # 右大腿
        offsets[4, :] += [0, +0.0002, -0.0003]           # 右脚
        offsets[6, :] += [0, -7.228100, 0]              # 左大腿
        offsets[9, :] += [0, +0.0002, -0.0003]           # 左脚
        offsets[16, :] += [0, 4.25, 0]                  # 头
        offsets[18, :] += [-6.564614 + 2.9, 0, 0]       # 右肩
        offsets[19, :] += [-12.435387 + 16.1, 0, 0]     # 右上臂
        offsets[46, :] += [6.564614 - 2.9, 0, 0]        # 左肩
        offsets[47, :] += [12.435387 - 16.1, 0, 0]      # 左上臂
        
        positions[:,0] += [0, 104.348106 - 97.120201, 0]
        
        # =====需求2:
        # rotations只需要把下标15的关节【neck1】合并到16节点【head】里就可以
        # 注意连锁变化：关节数-1，head的offset也要改
        
        # 另注：因为其他关节的positions并没有用上，所以positions只改根节点的就可以了，其他的不管啦！
        # 但是offset记得要改
        rot_neck1 = Quaternions.from_euler(np.radians(rotations[:,15]), order='xyz')
        rot_head = Quaternions.from_euler(np.radians(rotations[:,16]), order='xyz')
        rot_head_new = rot_neck1 * rot_head
        rot_head_new = np.degrees(rot_head_new.euler())
        rotations[:,16] = rot_head_new
        
        parents[16] = 14
        
        index = [x for x in range(74) if x!=15]
        
        self.anim.offsets = offsets[index]
        self.anim.positions = positions[:, index]
        self.anim.rotations = rotations[:, index]
        self.anim.parents = parents[index]
        self.anim.orients = orients[index]
        
        # names = self._names
        self._names.pop(15)
        
    
        
    
        
    
        
