import torch
import torch.nn as nn
from model.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear


class Encoder(nn.Module):
    def __init__(self, args, topology):     # topology是骨骼拓扑、数量比关节少1【v-1】
        super(Encoder, self).__init__()
        self.topologies = [topology]
        if args.rotation == 'euler_angle': self.channel_base = [3]
        elif args.rotation == 'quaternion': self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(topology) + 1]     # 包括了“全局位置”这一拟定的edge，所以 edge数量 = v-1+1 = v，和关节数相等
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        bias = True
        if args.skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(args.num_layers):    # 确定每层的channel_base（含义是：每个骨骼的特征数）
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(args.num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]       # 骨骼特征数 * edge数量(v)
            out_channels = self.channel_base[i+1] * self.edge_num[i]    # 骨骼特征数' * edge数量(v)
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            for _ in range(args.extra_conv):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == args.num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))     # 2层，每层【SkeletonConv -> SkeletonPool -> LeakyReLU】

            self.topologies.append(pool.new_edges)                  # 添加pool后的拓扑，作为下一层网络的参数
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)      # 根据拓扑变化，变化edge_num，作为下一层网络的参数
            if i == args.num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):  #【Encoder】
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            if self.args.skeleton_info == 'concat' and offset is not None:  
                self.convs[i].set_offset(offset[i]) # 设置卷积层、把动态motion和静态offset拼接
            input = layer(input)    # 训练时，input:[n,C(4v),t_w(64)] => [n,C'(8v'),t_w/2] => [n,C''(16v''),t_w/4]
        return input


class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        if args.skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(args.num_layers):
            seq = []
            in_channels = enc.channel_list[args.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[args.num_layers - i - 1], args.skeleton_dist)

            if i != 0 and i != args.num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[args.num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))  # 这一步进行的时间维度的扩充
            seq.append(self.unpools[-1])
            for _ in range(args.extra_conv):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[args.num_layers - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq)) #2层，每层【Upsample->SkeletonUnpool->SkeletonConv->LeakyReLU】

    def forward(self, input, offset=None):  # Decoder
        for i, layer in enumerate(self.layers):
            if self.args.skeleton_info == 'concat':
                self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)    # 训练时，input:[n,C''(16v''),t_w/4(16)] => [n,C'(8v'),t_w/2(32)] => [n,C(4v), t_w]
        # throw the padded rwo for global position
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = input[:, :-1, :]

        return input


class AE(nn.Module):
    def __init__(self, args, topology):
        r'''利用拓扑构建的编码器+解码器，topology是骨骼拓扑(v-1)'''
        super(AE, self).__init__()
        self.enc = Encoder(args, topology)
        self.dec = Decoder(args, self.enc)

    def forward(self, input, offset=None):  # input:[n,C(4v-4+3),t_w(64)]
        latent = self.enc(input, offset)    # latent:[n,C''(16v'',112),t_w/4(16)]
        result = self.dec(latent, offset)
        return latent, result


# eoncoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, args, edges):
        super(StaticEncoder, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        activation = nn.LeakyReLU(negative_slope=0.2)
        channels = 3    # 可以类比AE里的channel_base

        for i in range(args.num_layers):
            neighbor_list = find_neighbor(edges, args.skeleton_dist)    # 因为edge加上了root，所以尺寸和关节相同了
            seq = []
            # 骨骼线性层    [n,v,3] => [n, 6v, 1]
            seq.append(SkeletonLinear(neighbor_list, in_channels=channels * len(neighbor_list),
                                      out_channels=channels * 2 * len(neighbor_list), extra_dim1=True))
            if i < args.num_layers - 1:
                # 骨骼池化层    【channels * 2 * len(neighbor_list) => channels * 2 * len(self.pooling_list)】
                # 相当于是 [n,6v,1]=>[n,6v',1]
                pool = SkeletonPool(edges, channels_per_edge=channels*2, pooling_mode='mean')
                seq.append(pool)
                edges = pool.new_edges
            seq.append(activation)
            channels *= 2
            self.layers.append(nn.Sequential(*seq))
            # 最终self.layers应该是 (args.num_layers-1)个【线性层-池化层-激活函数】的网络 & 1个【线性层-激活函数】的网络

    # input should have shape B * E * 3
    def forward(self, input: torch.Tensor):
        output = [input]    # [n,v,3] + [n,6v'] + [n,12v']
        for i, layer in enumerate(self.layers):
            input = layer(input)
            output.append(input.squeeze(-1))    # 在这里压缩了最后一维的1
        return output



class AE_CIP(nn.Module):
    def __init__(self, args, topology):
        r'''利用拓扑构建的编码器+解码器，topology是骨骼拓扑(v-1)'''
        super(AE_CIP, self).__init__()
        self.enc = Encoder(args, topology)
        self.enc_cip = Encoder_CIP(args, self.enc)
        self.dec = Decoder(args, self.enc)

    def forward(self, input, offset=None):  # input:[n,C(42),t_w(64)], offset:[n,v(22),3] + [n,6v'(72)] + [n,12v'(144)]
        latent = self.enc_cip(input, offset)    # latent:[n,C''(16v'',192),t_w/4(16)]
        result = self.dec(latent, offset)
        return latent, result
    
class Encoder_CIP(nn.Module):
    def __init__(self, args, enc: Encoder):     # topology是骨骼拓扑、数量比关节少1【v-1】
        super(Encoder_CIP, self).__init__()
        self.layers = nn.ModuleList()
        # self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        
        # if args.skeleton_info == 'concat': add_offset = True
        # else: add_offset = False
        add_offset = False
        
        in_channels = enc.channel_list[args.num_layers] // 12 * 7 # // enc.channel_list[0].size() * enc.channel_list[1].size()
        out_channels = in_channels
        neighbor_list = find_neighbor(enc.topologies[args.num_layers - 1], args.skeleton_dist)
        
        bidirectional = True
        self.rnn = nn.GRU(42, in_channels // 4, 2, bidirectional=bidirectional, batch_first = True)
        self.rnn_en = nn.Linear(in_channels // 4 * (2 if bidirectional else 1), in_channels // 4)
        
        self.conv1 = nn.Conv1d(in_channels // 4, out_channels // 2, 15, stride=2, padding=7)
        self.conv2 = nn.Conv1d(in_channels // 2, out_channels, 15, stride=2, padding=7)
        
        # for i in range(args.num_layers):
        #     seq = []
        #     if i != 0 and i != args.num_layers - 1:
        #         bias = False
        #     else:
        #         bias = True
                
        #     seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
        #                             joint_num= 7, kernel_size=kernel_size, stride=2,
        #                             padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
        #                             in_offset_channel=3 * enc.channel_base[args.num_layers - 1] // enc.channel_base[0]))
        #     self.convs.append(seq[-1])
        #     if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

        #     self.layers.append(nn.Sequential(*seq))
            

    def forward(self, input, offset=None):  #【Encoder】
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        # if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
        #     input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)   #[n,88,64]

        input = input.permute(0,2,1)      #[n,C(42),t_w(64)]=>[n,64,42]
        input = self.rnn(input)[0]
        input = self.rnn_en(input)        #[n,64,C']
        input = input.permute(0,2,1)
        
        input = self.conv1(input)
        input = self.conv2(input)
        # for i, layer in enumerate(self.layers):
        #     # if self.args.skeleton_info == 'concat' and offset is not None:  
        #     #     self.convs[i].set_offset(offset[i]) # 设置卷积层、把动态motion和静态offset拼接
        #     input = layer(input)    # 训练时，input:[n,C(4v),t_w(64)] => [n,C'(8v'),t_w/2] => [n,C''(16v''),t_w/4]
        return input

