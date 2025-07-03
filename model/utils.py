# Classes in this file are mainly borrowed from Jun-Yan Zhu's cycleGAN repository

from torch import nn
import torch
import random
from torch.optim import lr_scheduler


class Cos_loss(nn.Module):
    '''
    输入: 预测的旋转矩阵 x 和真实的旋转矩阵 y，它们的形状为 [n, t, 15*9]。
    输出: 一个标量损失值，表示旋转矩阵之间的平均角度差异。
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        r'''
            x,y: [n,t,15*9]
        '''
        n,t,_ = x.shape
        x = x.view(n,t,15,3,3)
        y = y.view(n,t,15,3,3)
        index = [0,1,11,12]     # 在输出的SMPL15节点中、左右大腿、左右肩膀对应的关节点下标
        all = n*t*4
        
        pre = x[:,:,index].view(all,3,3)
        gt = y[:,:,index].view(all,3,3)
        loss = pre.new_zeros(all,1)
        for i in range(all):
            tmp = torch.trace(pre[i].transpose(0,1).matmul(gt[i]))-1    # TODO:tmp大于2
            if tmp > 2:     # 只是一个临时处理
                tmp = torch.Tensor([2]).to(tmp.device)
            loss[i] = abs(torch.acos((tmp)/2))
        return torch.mean(loss)


class GAN_loss(nn.Module):
    '''
     gan_mode: 字符串，指定使用的 GAN 损失模式。可以是以下几种之一：

    'lsgan': 采用最小均方误差（MSE）损失，适用于 Least Squares GAN（LSGAN）。
    'vanilla': 采用二元交叉熵（BCE）损失，适用于经典的 GAN。6
    'none': 没有损失函数，通常在一些调试或实验的场景下使用。
    real_label 和 fake_label: 这两个值分别表示真实标签和虚假标签的数值，默认分别为 1.0 和 0.0。可以根据需要自定义这些标签值。

    prediction: 这是生成器或判别器的预测值，通常是一个标量或者一个张量，表示生成样本或判别结果。

    target_is_real: 布尔值，指示当前标签是“真实”的（True）还是“假的”（False）。这个值帮助决定标签是 real_label 还是 fake_label。

    2. 输出 (Output)：
    输出: 一个标量的损失值，表示生成器或判别器的损失。具体计算方式取决于 gan_mode 和 target_is_real 的值：
    如果 gan_mode 是 'lsgan'，输出的是均方误差损失。
    如果 gan_mode 是 'vanilla'，输出的是二元交叉熵损失。
    如果 gan_mode 是 'none'，输出 None，表示没有计算损失。
    3. 功能 (Functionality)：
    该类的功能是根据不同的 GAN 模式计算损失函数，并返回与生成器或判别器输出相关的损失值。

    初始化 GAN_loss 对象时：

    根据 gan_mode，初始化适当的损失函数（MSE 或 BCE）。
    real_label 和 fake_label 用来为“真实”或“虚假”的样本分配相应的标签值。
    get_target_tensor 方法：

    这个方法根据 target_is_real 来返回“真实”或“虚假”的目标标签（real_label 或 fake_label）。并且会根据预测的形状将目标标签扩展到相同的形状。
    例如，如果 prediction 是形状为 [n, 1] 的张量，而 target_is_real 为 True，则目标标签会是一个形状为 [n, 1] 且值为 real_label 的张量。
    __call__ 方法：

    这个方法将预测值 (prediction) 和目标标签（由 get_target_tensor 获取）作为输入，计算并返回损失值。
    损失值是通过使用初始化时的损失函数 (self.loss) 来计算的。
    '''
    def __init__(self, gan_mode, real_lable=1.0, fake_lable=0.0):
        super(GAN_loss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_lable))
        self.register_buffer('fake_label', torch.tensor(fake_lable))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'none':
            self.loss = None
        else:
            raise Exception('Unknown GAN mode')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class Criterion_EE:
    '''
    Criterion_EE 是一个自定义的损失函数类，用于计算某些特定任务中的误差它包含了一个基础损失函数
    base_criterion，并且能够根据一些附加条件（如 ee_velo 参数）计算额外的损失。


    '''
    def __init__(self, args, base_criterion, norm_eps=0.008):
        self.args = args
        self.base_criterion = base_criterion
        self.norm_eps = norm_eps

    def __call__(self, pred, gt):
        '''

        Parameters
        ----------
        pred
        gt

        Returns   常规损失和额外损失的加权和
        -------

        '''
        reg_ee_loss = self.base_criterion(pred, gt)
        if self.args.ee_velo:
            gt_norm = torch.norm(gt, dim=-1)
            contact_idx = gt_norm < self.norm_eps
            extra_ee_loss = self.base_criterion(pred[contact_idx], gt[contact_idx])
        else:
            extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return []

class Criterion_EE_2:
    '''
   Criterion_EE_2 引入了一个 自适应参数（ada_para），这可以使损失函数根据输入的特征调整其计算方式
    '''
    def __init__(self, args, base_criterion, norm_eps=0.008):
        print('Using adaptive EE')
        self.args = args
        self.base_criterion = base_criterion
        self.norm_eps = norm_eps
        self.ada_para = nn.Linear(15, 15).to(torch.device(args.cuda_device))   # TODO: 尝试用一个线性层来调整参数

    def __call__(self, pred, gt):
        pred = pred.reshape(pred.shape[:-2] + (-1,))
        gt = gt.reshape(gt.shape[:-2] + (-1,))
        pred = self.ada_para(pred)
        reg_ee_loss = self.base_criterion(pred, gt)
        extra_ee_loss = 0
        return reg_ee_loss + extra_ee_loss * 100

    def parameters(self):
        return list(self.ada_para.parameters())

class Eval_Criterion:
    def __init__(self, parent):
        '''

        Parameters
        ----------
        parent
        '''
        self.pa = parent
        self.base_criterion = nn.MSELoss()
        pass

    def __call__(self, pred, gt):
        for i in range(1, len(self.pa)):
            pred[..., i, :] += pred[..., self.pa[i], :]
            gt[..., i, :] += pred[..., self.pa[i], :]
        return self.base_criterion(pred, gt)


class ImagePool():
    """ImagePool 是一个用于存储先前生成图像的缓冲区类，它的主要目的是在生成对抗网络（GANs）中，
    特别是在训练鉴别器（discriminator）时，提供一个历史生成图像的缓冲池。这种缓冲池可以帮助缓解训练中的
    模式崩溃（mode collapse）问题，使得鉴别器不仅仅依赖最新生成的图像，而是使用历史图像进行训练，从而提高
    训练的稳定性。
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        输入：

pool_size: 图像池的大小。
images: 最新生成的图像集合。
输出：

返回一个包含历史图像或当前图像的图像集合。
功能：

ImagePool 通过缓存生成的图像并按随机概率选择返回历史图像或当前图像，提供历史图像供鉴别器训练，帮助提高训练的稳定性。
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


def get_scheduler(optimizer, opt):
    """optimizer: 优化器对象，通常是 torch.optim 中定义的优化器，如 torch.optim.Adam。这个优化器的学习率将通过调度器来更新。
opt: 包含训练配置的对象，必须是 BaseOptions 类的子类。该对象包含了与学习率调度相关的各种配置项，主要是：
opt.lr_policy：学习率策略，支持四种策略：linear、step、plateau、cosine。
opt.n_epochs：训练总轮数，用于 linear 策略。
opt.n_epochs_decay：学习率衰减的轮数，用于 linear 策略。
opt.epoch_count：当前训练轮数，用于 linear 策略。
opt.lr_decay_iters：每多少个迭代周期调整一次学习率，用于 step 策略。

    get_scheduler：

输入：优化器和配置对象。
输出：根据学习率策略返回相应的学习率调度器。
功能：根据配置返回不同的学习率策略（如线性衰减、阶梯衰减、余弦退火等）。
get_ee：

输入：位置数据、父节点索引、末端执行器索引等。
输出：末端执行器的位置或速度信息。
功能：用于处理关节位置和速度，特别是末端执行器的位置计算。
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_ee(pos, pa, ees, velo=False, from_root=False):
    """
    输入：
get_ee 函数用于处理某些位置的数据，通常用于处理某些类型的位姿或骨骼数据。在某些任务中，可能需要对人体模型或其他物体的关键点进行处理，尤其是末端执行器（End-Effector，简称 EE）的处理。

    输入：
pos: 一个张量，通常表示一组位置（如人体各个关键点的坐标）。这个张量的维度通常是 [batch_size, time_steps, num_joints, 3]，表示每个样本的多个时间步，多个关节的 3D 坐标。
pa: 父节点索引列表，表示每个节点（如关节）与其父节点之间的关系。通常用一个列表来表示每个关节的父关节。
ees: 末端执行器（End-Effector，EE）节点的索引，用于选择与末端执行器相关的位置（例如手部或脚部的关节）。
velo: 布尔值，表示是否需要计算速度。如果为 True，会计算位置的差分。
from_root: 布尔值，表示是否从根节点开始累积位置。默认为 False。
输出：
返回一个处理后的 pos 张量，通常是末端执行器的坐标。如果 velo=True，则返回速度信息。

    """
    pos = pos.clone()
    for i, fa in enumerate(pa):
        if i == 0: continue
        if not from_root and fa == 0: continue
        pos[:, :, i, :] += pos[:, :, fa, :]

    pos = pos[:, :, ees, :]
    if velo:
        pos = pos[:, 1:, ...] - pos[:, :-1, ...]
        pos = pos * 10
    return pos


if __name__ == '__main__':
    tmp = torch.Tensor(2)
    print(tmp)
