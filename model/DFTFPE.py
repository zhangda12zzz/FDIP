# ----------------------------
# 时空图卷积网络模块
# ----------------------------
import sys

import torch
from torch import nn
import torch.nn.functional as F
from model.graph import Graph_B, Graph_A
from torch.nn import Linear


class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConvBlock, self).__init__()
        self.A = A  # 邻接矩阵

        # 特征转换矩阵
        self.W = nn.Linear(in_channels, out_channels)

        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.Identity()  # 使用nn.Identity()替代lambda

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A = None):
        """
        输入: [batch_size, seq_len, num_nodes, in_channels]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        # print(f"x_GCN shape: {x.shape}")

        # 残差连接
        res = self.residual(x.reshape(batch_size * seq_len * num_nodes, -1))
        res = res.reshape(batch_size, seq_len, num_nodes, -1)

        # 图卷积: X' = AXW
        x_transformed = self.W(x)  # XW: [batch, seq, nodes, out_channels]

        adj_matrix = A if A is not None else self.A  # 邻接矩阵乘以权重系数

        # 通过邻接矩阵传播，einsum是相当于每个batch和seq后进行矩阵乘法
        x_conv = torch.einsum('nm,bsmd->bsnd', adj_matrix, x_transformed)

        # 批归一化和激活
        x_conv = x_conv.reshape(batch_size * seq_len, num_nodes, -1)
        x_conv = x_conv.transpose(1, 2)  # [batch*seq, out_channels, nodes]
        x_conv = self.bn(x_conv)      #默认特征在第1个维度
        x_conv = x_conv.transpose(1, 2)  # [batch*seq, nodes, out_channels]
        x_conv = x_conv.reshape(batch_size, seq_len, num_nodes, -1)

        # 残差连接和激活
        out = self.relu(x_conv + res)

        return out


# -----------------------------------------------------------------------------
# 时间卷积块
# -----------------------------------------------------------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        输入: [batch_size, channels, seq_len]
        输出: [batch_size, out_channels, seq_len]
        """
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x) + res
        return x


# -------------------------------
# 改进的时空图卷积网络
# -------------------------------
class SpatioTemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_joints=6,stage='early'):
        """
        时空图卷积网络，用于处理IMU数据的时空关系

        参数:
            in_channels (int): 每个节点的输入特征维度
            hidden_dim (int): 隐藏层维度
            num_joints (int): IMU节点数量，默认为6 (根, 左脚, 右脚, 头, 左手, 右手)
        """
        super().__init__()
        self.sensor_joints = num_joints

        if stage == "early":
            self.graph = Graph_B(strategy="uniform")
            self.num_joints = 6
            self.A = self.graph.A_b
            self.need_mapping = False
        elif stage == "mid":
            self.graph = Graph_A(strategy="uniform")
            self.num_joints = 24
            self.A = self.graph.A_a
            self.need_mapping = True
            self.map_sensor2joints = nn.Conv1d(in_channels=self.sensor_joints,  # 输入通道数实际上是关节数
                                                out_channels=self.num_joints,    # 输出关节数
                                                kernel_size=1,              # 1x1卷积
                                                stride=1)
        else:
            self.graph = Graph_A(strategy="uniform")  # 或 Graph_A
            self.num_joints = 24  # 或 24
            self.A = self.graph.A_a  # 或 A_a
            self.need_mapping = False

        # 构建IMU节点图结构
        graph_adj = torch.tensor(self.A, dtype=torch.float32, requires_grad=False)  #转变为tensor
        if len(graph_adj.shape) == 3:
            graph_adj = graph_adj[0]  # 去掉批次维度，变成[6,6]或[24,24]
        self.register_buffer('adjacency', graph_adj)
        # 在SpatioTemporalGCN初始化中添加
        # print(f"Adjacency matrix shape: {graph_adj.shape}")

        # 拓扑自适应层 - 学习调整节点间连接强度(让权重矩阵变得可学习)
        self.edge_importance = nn.Parameter(torch.ones_like(graph_adj))

        # 图卷积层
        self.gcn1 = GraphConvBlock(in_channels, hidden_dim // 2, self.adjacency)   #注册在了模型缓冲区

        self.gcn2 = GraphConvBlock(hidden_dim // 2, hidden_dim, self.adjacency)

        # 时间卷积层
        self.tcn = TemporalConvBlock(
            in_channels=hidden_dim * self.num_joints,
            out_channels=hidden_dim * self.num_joints,
            kernel_size=5
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_joints, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        时空图卷积前向传播

        参数:
            x (tensor): 输入特征张量 [batch_size, seq_len, in_channels]
                       注意：输入特征应包含所有IMU节点的信息

        返回:
            tensor: 处理后的特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, input_joints, features_per_channel = x.shape

        if self.need_mapping == True:
            x_reshape = x.reshape(batch_size * seq_len, self.sensor_joints, features_per_channel)
            x_mapped = self.map_sensor2joints(x_reshape)
            x = x_mapped.reshape(batch_size , seq_len, self.num_joints, features_per_channel)
        # print(f"x shape: {x.shape}")

        # 加入边重要性权重
        adj_with_importance = self.adjacency * self.edge_importance
        # print(f"Adjacency matrix with importance shape: {adj_with_importance.shape}")
        # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB","ST0")
        # print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        # 第一层图卷积
        x_gcn1 = self.gcn1(x, adj_with_importance)
        # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB","ST1")
        # print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

        # 第二层图卷积
        x_gcn2 = self.gcn2(x_gcn1, adj_with_importance)
        # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB","ST2")
        # print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


        # 重塑并转置以应用时间卷积
        # [batch, seq, nodes, hidden] -> [batch, seq, nodes*hidden] -> [batch, nodes*hidden, seq]
        x_reshaped = x_gcn2.reshape(batch_size, seq_len, -1).transpose(1, 2)
        # print(f"x_reshaped shape: {x_reshaped.shape}")

        # 应用时间卷积
        x_temporal = self.tcn(x_reshaped).transpose(1, 2)
        # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB","ST3")
        # print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        # 输出层
        output = self.output_layer(x_temporal)

        return output


# -----------------------------
# 非对称滑动窗口双向GRU模块
# -----------------------------
class AsymmetricBidirectionalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, stage='early',frame_rate=60):
        """
        非对称滑动窗口双向GRU，针对局部动态建模优化

        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度(前后向GRU各占一半)
            window_size (int): 滑动窗口大小，默认25帧
            current_frame_idx (int): 当前帧在窗口中的索引位置(0-indexed)，默认19(第20帧)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stage = stage
        self.frame_rate = frame_rate


        if stage == 'early':
            self.window_size = int(self.frame_rate * 0.8) #0.8秒
            self.current_frame_idx = int(self.window_size * 2 // 3)  # 32帧
            self.droupout = 0.1
            self.stride = max(1, frame_rate // 10)
        elif stage == 'mid':
            self.window_size = int(self.frame_rate * 0.4) #0.4秒
            self.current_frame_idx = int(self.window_size * 2 // 3)
            self.droupout = 0.2
            self.stride = max(1, frame_rate // 15)
        else:
            self.window_size = int(self.frame_rate * 0.2) #0.2秒
            self.current_frame_idx = int(self.window_size * 2 // 3)
            self.droupout = 0.3
            self.stride = max(1, frame_rate // 20)

        # 前向GRU - 处理当前帧及其前序帧(共current_frame_idx+1帧)
        self.forward_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.droupout
        )

        # 后向GRU - 处理当前帧后的帧(共window_size-current_frame_idx-1帧)
        self.backward_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.droupout
        )

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # 时间注意力机制用于加权融合窗口内特征[一般不需要]
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        非对称滑动窗口双向GRU前向传播 (矢量化高性能版)

        参数:
            x: 输入序列 [batch_size, seq_len, input_dim]

        返回:
            output: 处理后的序列特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, feat_dim = x.shape
        half_left = self.current_frame_idx
        half_right = self.window_size - half_left - 1

        # 1. 对输入序列进行一次性填充，适配中心化窗口
        # [B, L, D] -> [B, D, L] for pad -> [B, D, L'] -> [B, L', D]
        x_padded = F.pad(x.transpose(1, 2), (half_left, half_right), mode='replicate').transpose(1, 2)

        padded_len = x_padded.shape[1]
        # 2. 使用 unfold 高效提取所有滑动窗口
        # [B, L', D] -> [B, D, L, W]
        # windows = x_padded.transpose(1, 2).unfold(dimension=2, size=self.window_size, step=1)

        windows = x_padded.transpose(1, 2).unfold(dimension=2, size=self.window_size, step=self.stride)

        # 计算输出序列的长度
        output_seq_len = windows.shape[2]

        # [B, D, L_out, W] -> [B, L_out, W, D]
        windows = windows.permute(0, 2, 3, 1)
        # -> [(B * L_out), W, D]
        windows = windows.reshape(batch_size * output_seq_len, self.window_size, feat_dim)

        # 前向 GRU 处理
        # 输入: [(B*L), W_f, D], 输出 hidden: [num_layers, (B*L), H/2]
        _, forward_hidden = self.forward_gru(windows[:, :self.current_frame_idx + 1])
        forward_hidden = forward_hidden[-1]

        _, backward_hidden = self.backward_gru(torch.flip(windows[:, self.current_frame_idx:], dims=[1]))
        backward_hidden = backward_hidden[-1]

        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        # 将形状重塑为 [B, L_out, H]
        outputs = combined.view(batch_size, output_seq_len, self.hidden_dim)

        outputs = self.norm(outputs)
        return outputs

    # def _apply_sliding_window_part(self, x):
    #     """应用滑动窗口分割序列
    #
    #     Args:
    #         x: 输入序列 [batch_size, seq_len, input_dim]
    #
    #     Returns:
    #         windowed_x: 窗口化序列 [batch_size, seq_len-window_size+1, window_size, input_dim]
    #     """
    #     batch_size, seq_len, feat_dim = x.shape
    #
    #     # 如果序列长度小于窗口大小，填充序列
    #     if seq_len < self.window_size:
    #         padding = torch.zeros(batch_size, self.window_size - seq_len, feat_dim,
    #                               device=x.device, dtype=x.dtype)
    #         x = torch.cat([x, padding], dim=1)
    #         seq_len = self.window_size
    #
    #     # 计算窗口数量
    #     num_windows = seq_len - self.window_size + 1
    #
    #     # 创建滑动窗口视图
    #     windowed_x = x.unfold(1, self.window_size, 1)
    #     return windowed_x.reshape(batch_size, num_windows, self.window_size, feat_dim)
    #
    #
    # def _apply_centered_sliding_window(self, x):
    #     """应用滑动窗口分割序列 --- 以当前帧为中心的滑动窗口
    #
    #     Args:
    #         x: 输入序列 [batch_size, seq_len, input_dim]
    #
    #     Returns:
    #         windowed_x: 窗口化序列 [batch_size, seq_len-window_size+1, window_size, input_dim]
    #     """
    #     batch_size, seq_len, feat_dim = x.shape
    #     half_left = self.current_frame_idx  # 当前帧在窗口中的位置
    #     half_right = self.window_size - half_left - 1
    #
    #     padded_x = F.pad(x, (0, 0, half_left, half_right))  # 在时间维度左右对张量pad填充，F是填充函数
    #     windowed = []
    #
    #     for t in range(seq_len):
    #         window = padded_x[:, t:t + self.window_size]  # [batch, window_size, feat_dim]
    #         windowed.append(window)
    #
    #     windowed_x = torch.stack(windowed, dim=1)  # [batch_size, seq_len, window_size, feat_dim]
    #     return windowed_x

    # def forward(self, x):
    #     """
    #     非对称滑动窗口双向GRU前向传播
    #
    #     参数:
    #         x: 输入序列 [batch_size, seq_len, input_dim]
    #
    #     返回:
    #         output: 处理后的序列特征 [batch_size, seq_len, hidden_dim]
    #     """
    #     batch_size, seq_len, _ = x.shape
    #     half_left = self.current_frame_idx
    #     half_right = self.window_size - half_left - 1
    #     padded_x = F.pad(x, (0, 0, half_left, half_right))
    #     outputs = []
    #     # 应用滑动窗口
    #     # windowed_x = self._apply_centered_sliding_window(x)
    #     # batch_size, num_windows, _, _ = windowed_x.shape
    #
    #     # # 为每个窗口的当前帧创建输出容器-空张量
    #     # outputs = torch.zeros(batch_size, num_windows, self.hidden_dim, device=x.device)
    #
    #     # 处理每个窗口
    #     for t in range(seq_len):
    #         # window = windowed_x[:, i]  # [batch_size, window_size, input_dim]
    #         window = padded_x[:, t:t + self.window_size]
    #         # 提取前向序列(包括当前帧)---[最后维度保持不变]
    #         forward_seq = window[:, :self.current_frame_idx + 1]
    #         # 提取后向序列(包括当前帧后的帧)，需要翻转
    #         # backward_seq = window[:, self.current_frame_idx:]
    #         backward_seq = torch.flip(window[:, self.current_frame_idx:], dims=[1])
    #         # backward_seq = torch.flip(backward_seq, dims=[1])   # 后向需要时间翻转
    #
    #         print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB", "gru1之前")
    #         print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    #         # 前向GRU处理
    #         _, forward_hidden = self.forward_gru(forward_seq)
    #         print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB", "gru1之后")
    #         print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    #         # 取最后一层的隐藏状态
    #         forward_hidden = forward_hidden[-1]  # [batch_size, hidden_dim//2]
    #
    #         print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB", "gru2之前")
    #         print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    #         # 后向GRU处理
    #         _, backward_hidden = self.backward_gru(backward_seq)
    #         print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB", "gru2之后")
    #         print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    #
    #         # 取最后一层的隐藏状态
    #         backward_hidden = backward_hidden[-1]  # [batch_size, hidden_dim//2]
    #
    #         # 拼接前向和后向隐藏状态
    #         combined = torch.cat([forward_hidden, backward_hidden], dim=1)
    #
    #         # 存储当前窗口的输出---第二个维度相合并
    #         outputs.append= combined
    #
    #     outputs = torch.stack(outputs, dim=1)
    #     # 应用层归一化
    #     outputs = self.norm(outputs)
    #     return outputs



#===========================================
# 一个支持温度缩放的自定义多头注意力模块
#===========================================
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        assert batch_first, "CustomMultiheadAttention 仅支持 batch_first=True"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 必须能被 num_heads 整除"

        # 为 query、key、value 分别定义线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, temperature=1.0):
        batch_size, seq_len, _ = query.size()

        # 1. 线性投影
        q = self.q_proj(query)  # 投影 query
        k = self.k_proj(key)    # 投影 key
        v = self.v_proj(value)  # 投影 value

        # 2. 为多头注意力重塑形状
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 带温度的缩放点积注意力
        scaling_factor = float(self.head_dim) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling_factor

        # --- 在此应用温度缩放 ---
        attn_scores = attn_scores / temperature

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 4. 计算输出
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights


class DualAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # 使用自定义的注意力模块
        self.self_attn_body = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn_limb = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.cross_attn_body2limb = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_limb2body = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # 前馈网络
        self.ffn_body = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_limb = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # 归一化层
        self.norm1_body = nn.LayerNorm(embed_dim)
        self.norm2_body = nn.LayerNorm(embed_dim)
        self.norm3_body = nn.LayerNorm(embed_dim)
        self.norm1_limb = nn.LayerNorm(embed_dim)
        self.norm2_limb = nn.LayerNorm(embed_dim)
        self.norm3_limb = nn.LayerNorm(embed_dim)

        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, body_feats, limb_feats, temperature=1.0):
        # 1. 带温度的自注意力
        body_self, _ = self.self_attn_body(body_feats, body_feats, body_feats, temperature=temperature)
        body = self.norm1_body(body_feats + body_self)

        limb_self, _ = self.self_attn_limb(limb_feats, limb_feats, limb_feats, temperature=temperature)
        limb = self.norm1_limb(limb_feats + limb_self)

        # 2. 带温度的交叉注意力
        body_cross, _ = self.cross_attn_body2limb(body, limb, limb, temperature=temperature)
        limb_cross, _ = self.cross_attn_limb2body(limb, body, body, temperature=temperature)

        # 自适应融合
        body_combined = torch.cat([body, body_cross], dim=-1)
        limb_combined = torch.cat([limb, limb_cross], dim=-1)

        body_gate = self.fusion_gate(body_combined)
        limb_gate = self.fusion_gate(limb_combined)

        body = self.norm2_body(body * body_gate[..., 0:1] + body_cross * body_gate[..., 1:2])
        limb = self.norm2_body(limb * limb_gate[..., 0:1] + limb_cross * limb_gate[..., 1:2])

        # 3. 前馈网络
        body = self.norm3_body(body + self.ffn_body(body))
        limb = self.norm3_limb(limb + self.ffn_limb(limb))

        return body, limb


class DualStreamTransformerFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=3, output_dim=15, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([      #三个堆叠的双流Transformer，不是三个阶段
            DualAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.pose_regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, output_dim)
        )

    def forward(self, body_feats, limb_feats, temperature=1.0):
        # 将温度参数传递给每一层
        for layer in self.layers:
            body_feats, limb_feats = layer(body_feats, limb_feats, temperature=temperature)

        # print('body_feats shape:', body_feats.shape)
        # print('limb_feats shape:', limb_feats.shape)

        fused_features = self.fusion(torch.cat([body_feats, limb_feats], dim=-1))
        pose_params = self.pose_regressor(fused_features)
        return pose_params


class DSTFPE(nn.Module):
    def __init__(self, num_nodes=6, trunk_dim=108, limb_dim=108, hidden_dim=512, output_dim=15, num_heads=8,
                 stage='early'):
        super().__init__()
        self.stage = stage

        # --- 将阶段(stage)映射到具体的温度值 ---
        self.temperature_map = {
            'early': 2.0,  # 早期阶段：高温度，平滑注意力
            'mid': 1.0,  # 中期阶段：标准温度
            'late': 0.5  # 后期阶段：低温度，锐化注意力
        }
        self.temperature = self.temperature_map.get(self.stage, 1.0)  # 如果stage无效，默认为1.0

        # 全局路径：时空图卷积
        self.st_gcn = SpatioTemporalGCN(
            in_channels=trunk_dim // num_nodes,
            hidden_dim=hidden_dim,
            num_joints=num_nodes,
            stage=self.stage
        )
        # 细节路径：双向GRU
        self.bi_gru = AsymmetricBidirectionalGRU(
            input_dim=limb_dim,
            hidden_dim=hidden_dim,
            stage=self.stage
        )
        # 融合模块
        self.transformer_fusion = DualStreamTransformerFusion(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            output_dim=output_dim,
            dropout=0.1
        )

    def forward(self, trunk_features, limb_features):
        # 全局路径
        # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB","刚进dstfpe")
        # print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        gcn_out = self.st_gcn(trunk_features)
        # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB","gcn之后")
        # print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        # 细节路径
        gru_out = self.bi_gru(limb_features)
        # print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB","gru之后")
        # print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

        # 对 GRU 的输出进行上采样，以匹配 GCN 的序列长度
        # 1. 获取目标序列长度
        target_seq_len = gcn_out.shape[1]

        # 2. 检查是否需要上采样
        if gru_out.shape[1] != target_seq_len:
            # F.interpolate 需要 (B, C, L) 的格式，所以先交换维度
            gru_out_permuted = gru_out.permute(0, 2, 1)  # [B, L_short, D] -> [B, D, L_short]

            # 3. 执行线性插值上采样
            gru_out_upsampled = F.interpolate(
                gru_out_permuted,
                size=target_seq_len,      # 目标长度
                mode='linear',            # 使用线性插值，适用于时序数据
                align_corners=False       # 推荐设置为 False
            )

            # 4. 将维度交换回来
            gru_out = gru_out_upsampled.permute(0, 2, 1) # [B, D, L_orig] -> [B, L_orig, D]

        # 使用特定阶段的温度进行融合
        joint_positions = self.transformer_fusion(gcn_out, gru_out, temperature=self.temperature)

        return joint_positions



