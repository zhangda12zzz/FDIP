import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STGCN_Block(nn.Module):
    """
    基础的时空图卷积网络 (ST-GCN) 模块 (PyTorch 实现 - 已修正)。

    包含一个图卷积层 (空间) 和一个时间卷积层 (时间)，并带有残差连接。
    输入张量格式: (N, C_in, T, V)
        N: 批量大小
        C_in: 输入通道数 (特征维度)
        T: 时间步长
        V: 关节点数量
    """
    def __init__(self, in_channels, output_channels, temporal_kernel_size, adjacency_matrix,
                 intermediate_channels_factor=2, # GCN 中间通道数的倍数
                 dropout_rate=0.2, use_residual=True, name="stgcn_block"):
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.use_residual = use_residual
        self.name = name

        # --- 归一化邻接矩阵 ---
        self.register_buffer('normalized_A', self._normalize_adjacency(adjacency_matrix))

        gcn_intermediate_channels = output_channels * intermediate_channels_factor

        # --- 图卷积 (空间) 相关层 ---
        self.gcn_conv = nn.Conv2d(in_channels, gcn_intermediate_channels, kernel_size=(1, 1))
        self.bn_gcn = nn.BatchNorm2d(gcn_intermediate_channels)

        # --- 时间卷积 (时间) 相关层 ---
        padding_t = (temporal_kernel_size - 1) // 2
        padding_v = 0
        # 注意：TCN 的输入通道数现在是 GCN 的输出通道数 gcn_intermediate_channels
        self.tcn_conv = nn.Conv2d(gcn_intermediate_channels, output_channels,
                                  kernel_size=(temporal_kernel_size, 1),
                                  padding=(padding_t, padding_v))
        self.bn_tcn = nn.BatchNorm2d(output_channels)

        # --- 激活函数和 Dropout ---
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # --- 残差连接相关 ---
        if use_residual:
            if in_channels != output_channels:
                self.residual_conv = nn.Conv2d(in_channels, output_channels, kernel_size=(1, 1))
            else:
                self.residual_conv = nn.Identity()
        else:
            # 如果不使用残差，则将其设为 None，并在 forward 中处理
            self.residual_conv = None


    def _normalize_adjacency(self, A):
        if not isinstance(A, torch.Tensor):
            A = torch.from_numpy(A).float()
        V = A.shape[0]
        # 确保在正确的设备上创建 eye 矩阵
        A_hat = A + torch.eye(V, dtype=torch.float32, device=A.device)
        D_hat_diag = torch.sum(A_hat, axis=1)
        valid_indices = D_hat_diag > 0
        D_hat_inv_sqrt_diag = torch.zeros_like(D_hat_diag)
        if torch.any(valid_indices): # 检查是否有有效索引，避免对空张量操作
             D_hat_inv_sqrt_diag[valid_indices] = torch.pow(D_hat_diag[valid_indices], -0.5)
        D_inv_sqrt_matrix = torch.diag(D_hat_inv_sqrt_diag)
        normalized_A = D_inv_sqrt_matrix @ A_hat @ D_inv_sqrt_matrix
        return normalized_A

    def forward(self, x):
        """
        前向传播 (已修正)。

        Args:
            x (torch.Tensor): 输入张量，形状 (N, C_in, T, V)。

        Returns:
            torch.Tensor: 输出张量，形状 (N, output_channels, T, V)。
        """
        # --- 计算残差捷径 ---
        if self.use_residual:
            shortcut = self.residual_conv(x) # residual_conv 会处理维度匹配
        else:
            shortcut = 0 # 如果不用残差，捷径为 0

        # --- 主路径 ---
        # 1. 图卷积 (GCN)
        #    a. 特征变换 WX
        x_main = self.gcn_conv(x)
        x_main = self.bn_gcn(x_main)
        x_main = self.relu(x_main)
        #    b. 邻接矩阵聚合 A' * (WX) <--- ***修正点 1：添加 einsum***
        #利用图的邻接关系 (self.normalized_A)，将每个节点 (v) 的特征更新为其邻居节点 (w) 特征的加权平均（或加权和）
        x_main = torch.einsum('vw,nctw->nctv', self.normalized_A, x_main)

        # 2. 时间卷积 (TCN)
        x_main = self.tcn_conv(x_main) # TCN 输入是 GCN 聚合后的结果
        x_main = self.bn_tcn(x_main)
        x_main = self.relu(x_main)

        # 3. Dropout
        x_main = self.dropout(x_main)

        # --- 组合：主路径 + 残差捷径 --- <--- ***修正点 2：加上 shortcut***
        output = x_main + shortcut
        output = self.relu(output) # 最后的激活函数

        return output

# --- 如何使用示例 ---
if __name__ == '__main__':
    # 假设参数
    N_BATCH = 4
    TIME_STEPS = 30
    NUM_JOINTS = 24 # V
    INPUT_CHANNELS = 128 # C_in
    OUTPUT_CHANNELS_BLOCK1 = 64
    OUTPUT_CHANNELS_BLOCK2 = 128
    TEMPORAL_KERNEL = 9

    # 创建一个随机的邻接矩阵 (你需要用你真实的骨架邻接矩阵替换)
    # 使用 numpy 创建，然后转为 tensor
    adj_matrix_np = np.random.uniform(size=(NUM_JOINTS, NUM_JOINTS))
    adj_matrix_np = (adj_matrix_np + adj_matrix_np.T) / 2 # 使其对称
    adj_matrix_np = np.where(adj_matrix_np > 0.7, 1.0, 0.0) # 二值化示例
    adj_matrix_torch = torch.from_numpy(adj_matrix_np).float()

    # 创建随机输入数据 (N, C_in, T, V)
    input_tensor = torch.randn(N_BATCH, INPUT_CHANNELS, TIME_STEPS, NUM_JOINTS)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化 ST-GCN 模块
    stgcn_block1 = STGCN_Block(in_channels=INPUT_CHANNELS,
                               output_channels=OUTPUT_CHANNELS_BLOCK1,
                               temporal_kernel_size=TEMPORAL_KERNEL,
                               adjacency_matrix=adj_matrix_torch) # 确保 adj_matrix_torch 已定义

    stgcn_block2 = STGCN_Block(in_channels=OUTPUT_CHANNELS_BLOCK1,
                               output_channels=OUTPUT_CHANNELS_BLOCK2,
                               temporal_kernel_size=TEMPORAL_KERNEL,
                               adjacency_matrix=adj_matrix_torch)

    # 构建一个简单的模型测试
    model = nn.Sequential(
        stgcn_block1,
        stgcn_block2
    ).to(device) # 确保 device 已定义

    input_tensor = torch.randn(N_BATCH, INPUT_CHANNELS, TIME_STEPS, NUM_JOINTS).to(device) # 确保 N_BATCH, INPUT_CHANNELS 等已定义

    model.train()
    output_tensor_train = model(input_tensor)

    model.eval()
    with torch.no_grad():
        output_tensor_eval = model(input_tensor)

    print("Input Shape:", input_tensor.shape)
    print("Output Shape (Train):", output_tensor_train.shape)
    print("Output Shape (Eval):", output_tensor_eval.shape)

    try:
        from torchinfo import summary
        summary(model, input_size=(N_BATCH, INPUT_CHANNELS, TIME_STEPS, NUM_JOINTS))
    except ImportError:
        print("\nInstall torchinfo for model summary: pip install torchinfo")