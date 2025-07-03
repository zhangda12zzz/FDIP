import os
import torch


"""
IntegratedModelGIP 类实现了一个包含生成器（pose_encoder）和判别器（discriminator）的集成模型。
生成器处理骨骼姿势编码，判别器用于判断生成的动作是否真实。
通过 parameters 方法，合并生成器和判别器的所有可训练参数。
save 和 load 方法用于保存和加载模型的权重，以便在训练过程中进行检查点保存和恢复。
"""
from model.ref_Transpose import TransPoseNet
from model.ref_pip import PIP
from model.motion_discriminator import MotionDiscriminator
from model.net import GGIP

# class testCIP:
#     def __init__(self, args):
#         self.args = args

class IntegratedModelGIP:
    # origin_offsets should have shape num_skeleton * J * 3
    def __init__(self, args, num_past_frame=20, num_future_frame=5):
        r'''
        构建融合模型（生成器+判别器）
        输入数据：self，args，骨骼拓扑（单独一种），origin_offsets（总之初始化时是None），device，同拓扑的一组模型名称
        '''
        self.args = args
        device = args.device

        if not args.simple_operator:                                        # 包括了3重网络
            # self.auto_encoder = TransPoseNet(num_past_frame, num_future_frame).to(device) # Transpose
            # self.auto_encoder = PIP(device=device).to(device) # PIP
            self.pose_encoder = GGIP(strategy='spatial').to(device)
            self.discriminator = MotionDiscriminator(rnn_size=256, input_size=90, num_layers=2, output_size=1).to(device)     # 判别器，要随着6d/9d的更改而更改
        else:
            raise Exception('Conventional operator not yet implemented')

    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        r'''生成网络参数：自动判别器+静态编码器（+身高）参数 '''
        return list(self.pose_encoder.parameters())# + list(self.static_encoder.parameters()) + self.height_para
        # return list(self.auto_encoder.parameters()) + list(self.pose_encoder.parameters())

    def D_parameters(self):
        r''' 判别网络：判别器参数 '''
        return list(self.discriminator.parameters())

    def save(self, path, epoch):
        from option_parser import try_mkdir

        path = os.path.join(path, str(epoch))
        try_mkdir(path)

        # torch.save(self.height, os.path.join(path, 'height.pt'))
        # torch.save(self.auto_encoder.state_dict(), os.path.join(path, 'auto_encoder.pt'))
        torch.save(self.pose_encoder.state_dict(), os.path.join(path, 'pose_encoder.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pt'))
        # torch.save(self.static_encoder.state_dict(), os.path.join(path, 'static_encoder.pt'))

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

        # self.auto_encoder.load_state_dict(torch.load(os.path.join(path, 'auto_encoder.pt'),
        #                                              map_location=self.args.cuda_device))
        self.pose_encoder.load_state_dict(torch.load(os.path.join(path, 'pose_encoder.pt'),
                                                     map_location=self.args.cuda_device))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pt'),
                                                     map_location=self.args.cuda_device))
        # self.static_encoder.load_state_dict(torch.load(os.path.join(path, 'static_encoder.pt'),
        #                                                map_location=self.args.cuda_device))
        print('load succeed!')
