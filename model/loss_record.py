import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class SingleLoss:
    """
    记录单个损失项在每个训练步骤和每个 epoch 的损失值。
将损失值写入 TensorBoard。

    """
    def __init__(self, name: str, writer: SummaryWriter):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.writer = writer

    def add_scalar(self, val, step=None):
        if step is None: step = len(self.loss_step)
        self.loss_step.append(val)
        self.loss_epoch_tmp.append(val)
        self.writer.add_scalar('Train/step_' + self.name, val, step)

    def epoch(self, step=None):
        if step is None: step = len(self.loss_epoch)
        loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        self.loss_epoch_tmp = []
        self.loss_epoch.append(loss_avg)
        self.writer.add_scalar('Train/epoch_' + self.name, loss_avg, step)

    def save(self, path):
        loss_step = np.array(self.loss_step)
        loss_epoch = np.array(self.loss_epoch)
        np.save(path + self.name + '_step.npy', loss_step)
        np.save(path + self.name + '_epoch.npy', loss_epoch)


class LossRecorder:
    """

管理多个 SingleLoss 实例，方便同时记录多个损失项。
提供统一的接口来添加损失值、记录 epoch 和保存数据。
    """
    def __init__(self, writer: SummaryWriter):
        self.losses = {}
        self.writer = writer

    def add_scalar(self, name, val, step=None):
        if isinstance(val, torch.Tensor): val = val.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer)
        self.losses[name].add_scalar(val, step)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        for loss in self.losses.values():
            loss.save(path)
