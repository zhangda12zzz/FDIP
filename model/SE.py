import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        # Squeeze步骤: 全局平均池化，将空间维度（H,W）压缩为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出为1x1的特征图

        # Excitation步骤: 两个全连接层，用于计算通道加权
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)  # 降维
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)  # 恢复通道数

        # Sigmoid 激活函数，用于生成通道权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: 输入特征图，形状为 [batch_size, channels, height, width]
        """
        # Squeeze：全局平均池化，获得每个通道的全局信息（形状为 [batch_size, channels]）
        batch_size, channels, _, _ = x.size()  # 获取输入特征图的尺寸
        y = self.avg_pool(x).view(batch_size, channels)  # 压缩后，形状变为 [batch_size, channels]

        # Excitation：通过全连接层计算每个通道的加权系数
        y = F.relu(self.fc1(y))  # 第一层全连接和 ReLU 激活
        y = self.fc2(y)  # 第二层全连接
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)  # 通过 Sigmoid 激活并恢复为 [batch_size, channels, 1, 1]

        # 重标定：逐通道地对输入特征图进行加权
        return x * y.expand_as(x)  # 对每个通道进行加权，输出新的特征图


# 示例使用：假设输入特征图 x 的形状为 (batch_size, channels, height, width)
x = torch.randn(32, 64, 128, 128)  # 假设 batch_size=32，通道数=64，图像尺寸为 128x128
se_block = SEBlock(64)  # 初始化 SE 模块，通道数为 64
out = se_block(x)  # 输出加权后的特征图

print(f"Input shape: {x.shape}")  # 输出输入形状
print(f"Output shape: {out.shape}")  # 输出加权后的输出形状
