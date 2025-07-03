import argparse


def get_parser():
    """
    创建并返回一个参数解析器，用于解析命令行参数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./train', help='保存所有训练输出的目录。')
    parser.add_argument('--cuda_device', type=str, default='cuda:0', help='使用的CUDA设备，例如："cuda:0"。')
    parser.add_argument('--num_layers', type=int, default=2, help='模型中的层数。')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='优化器的学习率。')
    parser.add_argument('--alpha', type=float, default=0, help='稀疏正则化的惩罚因子。')
    parser.add_argument('--batch_size', type=int, default=64, help='训练时的批量大小。')
    parser.add_argument('--upsampling', type=str, default='linear', help="上采样方法：'stride2', 'nearest', 或 'linear'。")
    parser.add_argument('--downsampling', type=str, default='stride2', help='下采样方法："stride2" 或 "max_pooling"。')
    parser.add_argument('--batch_normalization', type=int, default=0, help='是否启用批归一化：1 启用，0 不启用。')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='激活函数："ReLU", "LeakyReLU", 或 "tanh"。')
    parser.add_argument('--rotation', type=str, default='quaternion', help='旋转表示方式："euler_angle" 或 "quaternion"。')
    parser.add_argument('--data_augment', type=int, default=1, help='是否启用数据增强：1 启用，0 不启用。')
    parser.add_argument('--epoch_num', type=int, default=306, help='训练的总轮数。')
    parser.add_argument('--window_size', type=int, default=128, help='每个窗口的时间轴长度。')
    parser.add_argument('--kernel_size', type=int, default=15, help='卷积的核大小（必须为奇数）。')
    parser.add_argument('--base_channel_num', type=int, default=-1, help='模型中基础通道数。')
    parser.add_argument('--normalization', type=int, default=1, help='是否启用归一化：1 启用，0 不启用。')
    parser.add_argument('--verbose', type=int, default=1, help='是否启用详细日志：1 启用，0 不启用。')
    parser.add_argument('--skeleton_dist', type=int, default=2, help='骨骼数据的距离度量。')
    parser.add_argument('--skeleton_pool', type=str, default='mean', help='骨骼数据的池化方法："mean" 或其他。')
    parser.add_argument('--extra_conv', type=int, default=0, help='是否启用额外的卷积层：1 启用，0 不启用。')
    parser.add_argument('--padding_mode', type=str, default='reflection', help='卷积的填充模式："reflection" 或其他。')
    parser.add_argument('--dataset', type=str, default='Mixamo', help='使用的数据集，例如："Mixamo"。')
    parser.add_argument('--fk_world', type=int, default=0, help='是否在世界坐标系中启用前向运动学：1 启用，0 不启用。')
    parser.add_argument('--patch_gan', type=int, default=1, help='是否启用PatchGAN：1 启用，0 不启用。')
    parser.add_argument('--debug', type=int, default=0, help='是否启用调试模式：1 启用，0 不启用。')
    parser.add_argument('--skeleton_info', type=str, default='concat', help='处理骨骼信息的方法："concat" 或其他。')
    parser.add_argument('--ee_loss_fact', type=str, default='height', help='末端执行器损失的因子："height" 或其他。')
    parser.add_argument('--pos_repr', type=str, default='3d', help='位置表示方式："3d" 或其他。')
    parser.add_argument('--D_global_velo', type=int, default=0, help='是否在判别器中启用全局速度：1 启用，0 不启用。')
    parser.add_argument('--gan_mode', type=str, default='finetune', help='GAN模式："lsgan", "none", 或 "finetune"。')
    parser.add_argument('--pool_size', type=int, default=50, help='GAN训练池的大小。')
    parser.add_argument('--is_train', type=int, default=1, help='是否启用训练模式：1 启用，0 不启用。')
    parser.add_argument('--model', type=str, default='mul_top_mul_ske', help='使用的模型架构。')
    parser.add_argument('--epoch_begin', type=int, default=0, help='训练的起始轮数。')
    parser.add_argument('--lambda_rec', type=float, default=5, help='重建损失的权重。')
    parser.add_argument('--lambda_cycle', type=float, default=5, help='循环一致性损失的权重。')
    parser.add_argument('--lambda_ee', type=float, default=100, help='末端执行器损失的权重。')
    parser.add_argument('--lambda_global_pose', type=float, default=2.5, help='全局姿态损失的权重。')
    parser.add_argument('--lambda_position', type=float, default=1, help='位置损失的权重。')
    parser.add_argument('--ee_velo', type=int, default=1, help='是否启用末端执行器速度：1 启用，0 不启用。')
    parser.add_argument('--ee_from_root', type=int, default=1, help='是否从根节点启用末端执行器：1 启用，0 不启用。')
    parser.add_argument('--scheduler', type=str, default='none', help='学习率调度器："none" 或其他。')
    parser.add_argument('--rec_loss_mode', type=str, default='extra_global_pos', help='重建损失模式："extra_global_pos" 或其他。')
    parser.add_argument('--adaptive_ee', type=int, default=0, help='是否启用自适应末端执行器损失：1 启用，0 不启用。')
    parser.add_argument('--simple_operator', type=int, default=0, help='是否启用简单操作符：1 启用，0 不启用。')
    parser.add_argument('--use_sep_ee', type=int, default=0, help='是否启用单独的末端执行器损失：1 启用，0 不启用。')
    parser.add_argument('--eval_seq', type=int, default=0, help='是否启用序列评估：1 启用，0 不启用。')
    return parser


def get_args():
    """
    解析并返回命令行参数。
    """
    parser = get_parser()
    return parser.parse_args()


def get_std_bvh(args=None, dataset=None):
    """
    根据数据集名称返回标准BVH文件路径。
    """
    if args is None and dataset is None: raise Exception('Unexpected parameter')
    if dataset is None: dataset = args.dataset
    std_bvh = './dataset/Mixamo/std_bvhs/{}.bvh'.format(dataset)
    return std_bvh


def try_mkdir(path):
    """
    尝试创建目录，如果目录不存在则创建。
    """
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
