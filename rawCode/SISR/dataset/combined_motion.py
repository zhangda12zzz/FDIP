import os
import numpy as np
import torch
from torch.utils.data import Dataset
import copy

from dataset.motion_dataset import MotionData
from dataset.bvh_parser import BVH_file
from option_parser import get_std_bvh
from dataset import get_test_set


class MixedData0(Dataset):
    """
    Mixed data for many skeletons but one topologies
    """
    def __init__(self, args, motions, skeleton_idx):
        super(MixedData0, self).__init__()

        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.args.data_augment == 0 or torch.rand(1) < 0.5:
            return [self.motions[item], self.skeleton_idx[item]]            # 
        else:
            return [self.motions_reverse[item], self.skeleton_idx[item]]


class MixedData(Dataset):
    """
    data_gruop_num * 2 * samples
    """
    def __init__(self, args, datasets_groups):
        device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.final_data = []
        self.length = 0
        self.offsets = []
        self.joint_topologies = []
        self.ee_ids = []
        self.means = []
        self.vars = []
        dataset_num = 0
        seed = 19260817
        total_length = 10000000
        all_datas = []
        for datasets in datasets_groups:    # 对于每一种拓扑骨骼而言
            offsets_group = []
            means_group = []
            vars_group = []
            dataset_num += len(datasets)    # 每种拓扑下包括了几种模型
            tmp = []
            for i, dataset in enumerate(datasets):  # i:序号, dataset:模型名，比如'Smpl'
                new_args = copy.copy(args)
                new_args.data_augment = 0
                new_args.dataset = dataset

                tmp.append(MotionData(new_args))    # 给模型名为dataset的模型构建数据类、存储了归一化后的[n,C,t_w]的动态信息

                mean = np.load('./dataset/Mixamo/mean_var/{}_mean.npy'.format(dataset))    # [C,1]，C和tmp中数据的C相等
                var = np.load('./dataset/Mixamo/mean_var/{}_var.npy'.format(dataset))      # [C,1]
                mean = torch.tensor(mean)
                mean = mean.reshape((1,) + mean.shape)  #[1,C,1]
                var = torch.tensor(var)
                var = var.reshape((1,) + var.shape)     #[1,C,1]

                means_group.append(mean)
                vars_group.append(var)      # 保存对应的均值、方差信息
    
                file = BVH_file(get_std_bvh(dataset=dataset))       # std_bvh路径的file中存储了全部静态信息（预处理后的）
                if i == 0:                                      # 对于这组模型（同类拓扑）的第一个模型，需要记录一下拓扑信息
                    self.joint_topologies.append(file.topology) # 骨骼拓扑【就是每个节点的父节点编号】，[v]
                    self.ee_ids.append(file.get_ee_id())        # 末端节点编号，[5]
                new_offset = file.offset                                    # 静态关节offset，[v,3]
                new_offset = torch.tensor(new_offset, dtype=torch.float)    # tensor[v,3]
                new_offset = new_offset.reshape((1,) + new_offset.shape)    # [1,v,3]
                offsets_group.append(new_offset)

                total_length = min(total_length, len(tmp[-1]))  # 用一组同拓扑数据中的最小数据量作为统一的total_length
            all_datas.append(tmp)                                   # all_datas包括了tmp包括了MitionData，最终all_datas有两个元素，每个元素内是同拓扑的、不同模型的MotionData
            offsets_group = torch.cat(offsets_group, dim=0)         # [n_c,v,3]
            offsets_group = offsets_group.to(device)                
            means_group = torch.cat(means_group, dim=0).to(device)  # [n_c,C,1]
            vars_group = torch.cat(vars_group, dim=0).to(device)    # [n_c,C,1]，n_c指的是同拓扑下、模型的个数
            self.offsets.append(offsets_group)  # [2, [n_c,v,3]]
            self.means.append(means_group)      # [2, [n_c,C,1]]
            self.vars.append(vars_group)        # [2, [n_c,C,1]]，外面是list，里面是tensor

        for datasets in all_datas:  # all_datas有2个元素，每个datasets包括了数个MotionData
            pt = 0
            motions = []
            skeleton_idx = []
            for dataset in datasets:    # 对于每个模型的MotionData而言
                motions.append(dataset[:])              # dataset[:]就是[n,C,t_w]的动态数据，这里的n是窗口裁剪后的动态数据量
                skeleton_idx += [pt] * len(dataset)     # skeleton_idx中添加了n个[0]元素，代表这n个数据属于第0个模型
                pt += 1
            motions = torch.cat(motions, dim=0)         # tensor[n*n_c, C, t_w]
            if self.length != 0 and self.length != len(skeleton_idx):   # 确定总长度，默认为 n*n_c，有其他设置则向下对齐
                self.length = min(self.length, len(skeleton_idx))
            else:
                self.length = len(skeleton_idx)
            self.final_data.append(MixedData0(args, motions, skeleton_idx)) # 将同拓扑的不同模型（也就是多个MotionData）统一为数据集类MotionData0

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        return data * var + means

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:    # 每次取数据、都会每种拓扑各取一个
            res.append(data[item])
        return res  # 包括了[2, motion, character], 其中、motion:tensor[1,C,t_w], character:一个int


class TestData(Dataset):
    def __init__(self, args, characters, std_path=None):
        self.characters = characters
        self.file_list = get_test_set()
        self.mean = []
        self.joint_topologies = []
        self.var = []
        self.offsets = []
        self.ee_ids = []
        self.args = args
        self.device = torch.device(args.cuda_device)
        self.std_path = std_path

        for i, character_group in enumerate(characters):
            mean_group = []
            var_group = []
            offsets_group = []
            for j, character in enumerate(character_group):
                if self.std_path:
                    file = BVH_file(self.std_path[i][j])
                else:
                    file = BVH_file(get_std_bvh(dataset=character))
                # file = BVH_file(get_std_bvh(dataset=character))
                if j == 0:
                    self.joint_topologies.append(file.topology)
                    self.ee_ids.append(file.get_ee_id())
                new_offset = file.offset
                new_offset = torch.tensor(new_offset, dtype=torch.float)
                new_offset = new_offset.reshape((1,) + new_offset.shape)
                offsets_group.append(new_offset)
                mean = np.load('./dataset/Mixamo/mean_var/{}_mean.npy'.format(character))
                var = np.load('./dataset/Mixamo/mean_var/{}_var.npy'.format(character))
                mean = torch.tensor(mean)
                mean = mean.reshape((1, ) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1, ) + var.shape)
                mean_group.append(mean)
                var_group.append(var)

            mean_group = torch.cat(mean_group, dim=0).to(self.device)
            var_group = torch.cat(var_group, dim=0).to(self.device)
            offsets_group = torch.cat(offsets_group, dim=0).to(self.device)
            self.mean.append(mean_group)
            self.var.append(var_group)
            self.offsets.append(offsets_group)

    def __getitem__(self, item):
        res = []
        bad_flag = 0
        for i, character_group in enumerate(self.characters):
            res_group = []
            ref_shape = None
            for j in range(len(character_group)):
                new_motion = self.get_item(i, j, item)
                if new_motion is not None:
                    new_motion = new_motion.reshape((1, ) + new_motion.shape)
                    new_motion = (new_motion - self.mean[i][j]) / self.var[i][j]
                    ref_shape = new_motion
                res_group.append(new_motion)

            if ref_shape is None:
                print('Bad at {}'.format(item))
                return None
            for j in range(len(character_group)):
                if res_group[j] is None:
                    bad_flag = 1
                    res_group[j] = torch.zeros_like(ref_shape)
            if bad_flag:
                print('Bad at {}'.format(item))

            res_group = torch.cat(res_group, dim=0)
            res.append([res_group, list(range(len(character_group)))])
        return res

    def __len__(self):
        return len(self.file_list)

    def get_item(self, gid, pid, id):
        character = self.characters[gid][pid]
        path = './dataset/Mixamo/{}/'.format(character)
        if isinstance(id, int):
            file = path + self.file_list[id]
        elif isinstance(id, str):
            file = id
        else:
            raise Exception('Wrong input file type')
        if not os.path.exists(file):
            raise Exception('Cannot find file')
        file = BVH_file(file)
        motion = file.to_tensor(quater=self.args.rotation == 'quaternion')
        motion = motion[:, ::2]
        length = motion.shape[-1]
        length = length // 4 * 4
        return motion[..., :length].to(self.device)

    def denorm(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return data * var + means

    def normalize(self, gid, pid, data):
        means = self.mean[gid][pid, ...]
        var = self.var[gid][pid, ...]
        return (data - means) / var
