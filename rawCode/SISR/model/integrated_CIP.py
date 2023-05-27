import os
import torch

from model.enc_and_dec_cnn import AE, AE_CIP, StaticEncoder
# from model.enc_and_dec_rnn import AE, StaticEncoder
# from model.enc_and_dec_crnn import AE, StaticEncoder
from model.vanilla_gan import Discriminator
from model.skeleton import build_edge_topology
from model.Kinematics import ForwardKinematics
from dataset.bvh_parser import BVH_file
from option_parser import get_std_bvh

class testCIP:
    def __init__(self, args):
        self.args = args

class IntegratedModelCIP:
    # origin_offsets should have shape num_skeleton * J * 3
    def __init__(self, args, joint_topology, origin_offsets: torch.Tensor, device, characters, std_path=None):
        r'''
        构建融合模型（生成器+判别器）
        输入数据：self，args，骨骼拓扑（单独一种），origin_offsets（总之初始化时是None），device，同拓扑的一组模型名称
        '''
        self.args = args
        self.joint_topology = joint_topology    # [v]
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3))) #[v-1, 3]，每个包括了【父节点、自节点、offset】
        self.fk = ForwardKinematics(args, self.edges)   # 包括了骨骼对应关系、父节点等正向运动学相关数据的计算类

        self.height = [] # for normalize ee_loss
        self.real_height = []
        self.std_path = std_path
        
        char_idx = 0
        for char in characters:
            if self.std_path:
                std_bvh_path = self.std_path[char_idx]
            else:
                std_bvh_path = get_std_bvh(dataset=char)
            if args.use_sep_ee:
                h = BVH_file(std_bvh_path).get_ee_length()
            else:
                h = BVH_file(std_bvh_path).get_height()
            if args.ee_loss_fact == 'learn':
                h = torch.tensor(h, dtype=torch.float)
            else:
                h = torch.tensor(h, dtype=torch.float, requires_grad=False)
            self.real_height.append(BVH_file(std_bvh_path).get_height())
            self.height.append(h.unsqueeze(0))
            char_idx += 1
            
        self.real_height = torch.tensor(self.real_height, device=device)    #[n_c]
        self.height = torch.cat(self.height, dim=0)                         #[n_c]
        self.height = self.height.to(device)
        if not args.use_sep_ee: self.height.unsqueeze_(-1)                  #[n_c, 1]
        if args.ee_loss_fact == 'learn': self.height_para = [self.height]
        else: self.height_para = []

        if not args.simple_operator:                                        # 包括了3重网络
            self.auto_encoder = AE_CIP(args, topology=self.edges).to(device)        # 自动编码器：编码器+解码器
            self.discriminator = Discriminator(args, self.edges).to(device)     # 判别器
            self.static_encoder = StaticEncoder(args, self.edges).to(device)    # 静态编码器
        else:
            raise Exception('Conventional operator not yet implemented')

    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        r'''生成网络参数：自动判别器+静态编码器（+身高）参数 '''
        return list(self.auto_encoder.parameters()) + list(self.static_encoder.parameters()) + self.height_para

    def D_parameters(self):
        r''' 判别网络：判别器参数 '''
        return list(self.discriminator.parameters())

    def save(self, path, epoch):
        from option_parser import try_mkdir

        path = os.path.join(path, str(epoch))
        try_mkdir(path)

        torch.save(self.height, os.path.join(path, 'height.pt'))
        torch.save(self.auto_encoder.state_dict(), os.path.join(path, 'auto_encoder.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pt'))
        torch.save(self.static_encoder.state_dict(), os.path.join(path, 'static_encoder.pt'))

        print('Save at {} succeed!'.format(path))

    def load(self, path, epoch=None):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')

        if epoch is None:
            all = [int(q) for q in os.listdir(path) if os.path.isdir(path + q)]
            if len(all) == 0:
                raise Exception('Empty loading path')
            epoch = sorted(all)[-1]

        # debug
        # epoch = '10000_smpl2aj_success'

        path = os.path.join(path, str(epoch))
        print('loading from epoch {}......'.format(epoch))

        self.auto_encoder.load_state_dict(torch.load(os.path.join(path, 'auto_encoder.pt'),
                                                     map_location=self.args.cuda_device))
        self.static_encoder.load_state_dict(torch.load(os.path.join(path, 'static_encoder.pt'),
                                                       map_location=self.args.cuda_device))
        print('load succeed!')
