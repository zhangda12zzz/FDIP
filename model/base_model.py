import os
import torch
import torch.optim

from abc import ABC, abstractmethod
from model.loss_record import LossRecorder


class BaseModel(ABC):
    """这是一个模型的抽象基类 (ABC)。
    要创建子类，你需要实现以下五个函数：
        -- <__init__>:                      初始化类；首先调用 BaseModel.__init__(self, opt)。
        -- <set_input>:                     从数据集中解包数据并应用预处理。
        -- <forward>:                      生成中间结果。
        -- <optimize_parameters>:           计算损失、梯度并更新网络权重。
    """

    def __init__(self, args, log_path=None):
        self.args = args
        self.is_train = args.is_train
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.model_save_dir = os.path.join(args.save_dir, 'models')  # 将所有检查点保存到 save_dir

        if self.is_train:
            from torch.utils.tensorboard import SummaryWriter
            if log_path:
                self.log_path = os.path.join(log_path)
            else:
                self.log_path = os.path.join(args.save_dir, 'logs')
            self.writer = SummaryWriter(self.log_path)
            self.loss_recoder = LossRecorder(self.writer)

        self.epoch_cnt = 0
        self.schedulers = []
        self.optimizers = []

    @abstractmethod  # 必须在子类中实现
    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。
        参数:
            input (dict): 包括数据本身及其元数据信息。
        """
        pass

    @abstractmethod
    def compute_test_result(self):
        """
        在前向传播后，执行一些操作，如输出 bvh 文件、获取误差值等。
        """
        pass

    @abstractmethod
    def forward(self):
        """定义模型的前向传播。这个方法在训练和测试阶段都需要被调用。子类需要实现具体的前向传播操作。"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        计算损失、梯度并更新网络权重；在每个训练迭代中调用。
        """
        pass

    def get_scheduler(self, optimizer):
        if self.args.scheduler == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - self.args.n_epochs_origin) / float(self.args.n_epochs_decay + 1)
                return lr_l
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        if self.args.scheduler == 'Step_LR':
            print('Step_LR 调度器已设置')
            return torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
        if self.args.scheduler == 'Plateau':
            print('Plateau_LR 调度器已设置')
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5, verbose=True)
        if self.args.scheduler == 'MultiStep':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])

    def setup(self):
        """加载并打印网络；创建调度器
        参数:
            opt (Option class) -- 存储所有实验标志；需要是 BaseOptions 的子类
        """
        if self.is_train:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def epoch(self):
        """
        每个训练轮次调用一次，更新 epoch_cnt 计数器，并为每个调度器调用 scheduler.step()，以更新学习率。
        """
        self.loss_recoder.epoch()
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()
        self.epoch_cnt += 1

    def test(self):
        """在测试时使用的前向传播函数。
        这个函数将 <forward> 函数包装在 no_grad() 中，因此我们不会保存中间步骤以进行反向传播。
        它还会调用 <compute_visuals> 以生成额外的可视化结果。
        """
        with torch.no_grad():
            self.forward()
            self.compute_test_result()
