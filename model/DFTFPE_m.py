# ----------------------------
# 轻量化时空图卷积网络模块
# ----------------------------

import torch
from torch import nn
import torch.nn.functional as F
from model.graph import Graph_B, Graph_A
from torch.nn import Linear


# 增强的图卷积块
class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, reduction_ratio=4):
        super(GraphConvBlock, self).__init__()
        self.A = A

        # 多层深度可分离图卷积
        self.depthwise_conv1 = nn.Linear(in_channels, in_channels, bias=False)
        self.depthwise_conv2 = nn.Linear(in_channels, in_channels, bias=False)
        self.depthwise_conv3 = nn.Linear(in_channels, in_channels, bias=False)

        self.pointwise_conv1 = nn.Linear(in_channels, in_channels // 2)
        self.pointwise_conv2 = nn.Linear(in_channels // 2, out_channels // 2)
        self.pointwise_conv3 = nn.Linear(out_channels // 2, out_channels)

        # 多层通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.Linear(out_channels, max(1, out_channels // reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, out_channels // reduction_ratio), max(1, out_channels // (reduction_ratio // 2))),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, out_channels // (reduction_ratio // 2)), out_channels),
            nn.Sigmoid()
        )

        # 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 8, 1),
            nn.Sigmoid()
        )

        # 增强残差连接
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Linear(in_channels, max(1, out_channels // 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(1, out_channels // 8), max(1, out_channels // 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(1, out_channels // 4), out_channels),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(0.1)
            )
        else:
            self.residual = nn.Identity()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels // 2)
        self.bn3 = nn.BatchNorm1d(out_channels // 2)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, A=None):
        batch_size, seq_len, num_nodes, _ = x.shape

        # 残差连接
        res = self.residual(x.reshape(batch_size * seq_len * num_nodes, -1))
        res = res.reshape(batch_size, seq_len, num_nodes, -1)

        # 多层深度可分离图卷积
        adj_matrix = A if A is not None else self.A

        # 第一层
        x_depth1 = self.depthwise_conv1(x)
        x_conv1 = torch.einsum('nm,bsmd->bsnd', adj_matrix, x_depth1)
        x_point1 = self.pointwise_conv1(x_conv1)
        x_point1 = self.bn2(x_point1.reshape(-1, x_point1.shape[-1])).reshape(batch_size, seq_len, num_nodes, -1)
        x_point1 = self.relu(x_point1)
        x_point1 = self.dropout(x_point1)

        # 第二层
        x_depth2 = self.depthwise_conv2(x_point1)
        x_conv2 = torch.einsum('nm,bsmd->bsnd', adj_matrix, x_depth2)
        x_point2 = self.pointwise_conv2(x_conv2)
        x_point2 = self.bn3(x_point2.reshape(-1, x_point2.shape[-1])).reshape(batch_size, seq_len, num_nodes, -1)
        x_point2 = self.relu(x_point2)
        x_point2 = self.dropout(x_point2)

        # 第三层
        x_depth3 = self.depthwise_conv3(x_point2)
        x_conv3 = torch.einsum('nm,bsmd->bsnd', adj_matrix, x_depth3)
        x_point3 = self.pointwise_conv3(x_conv3)

        # 通道注意力
        x_flat = x_point3.reshape(batch_size * seq_len, num_nodes, -1)
        x_flat = x_flat.transpose(1, 2)
        pooled = x_flat.mean(dim=2)
        channel_att = self.channel_attention(pooled).unsqueeze(-1)
        x_flat = x_flat * channel_att

        # 空间注意力
        x_spatial = x_flat.transpose(1, 2).reshape(batch_size * seq_len * num_nodes, -1)
        spatial_att = self.spatial_attention(x_spatial)
        x_spatial = x_spatial * spatial_att
        x_flat = x_spatial.reshape(batch_size * seq_len, num_nodes, -1).transpose(1, 2)

        x_flat = self.bn4(x_flat)
        x_flat = x_flat.transpose(1, 2)
        x_conv = x_flat.reshape(batch_size, seq_len, num_nodes, -1)

        out = self.relu(x_conv + res)
        return out


# -----------------------------------------------------------------------------
# 增强的时间卷积块
# -----------------------------------------------------------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=4):
        super(TemporalConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        groups = min(groups, in_channels)

        # 多尺度Ghost卷积
        self.primary_conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size,
                      padding=padding, groups=groups),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.primary_conv2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=5,
                      padding=2, groups=groups),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 便宜操作
        self.cheap_operation1 = nn.Sequential(
            nn.Conv1d(out_channels // 4, out_channels // 4, 3,
                      padding=1, groups=out_channels // 4),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation2 = nn.Sequential(
            nn.Conv1d(out_channels // 4, out_channels // 4, 5,
                      padding=2, groups=out_channels // 4),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 第二层多尺度卷积
        self.conv2_primary1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 4, kernel_size,
                      padding=padding, groups=min(groups, out_channels)),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.conv2_primary2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 4, kernel_size=5,
                      padding=2, groups=min(groups, out_channels)),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.conv2_cheap1 = nn.Sequential(
            nn.Conv1d(out_channels // 4, out_channels // 4, 3,
                      padding=1, groups=out_channels // 4),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.conv2_cheap2 = nn.Sequential(
            nn.Conv1d(out_channels // 4, out_channels // 4, 5,
                      padding=2, groups=out_channels // 4),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # 增强残差连接
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels // 2, kernel_size=1),
                nn.BatchNorm1d(out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)

        # 第一层多尺度Ghost卷积
        x1_primary1 = self.primary_conv1(x)
        x1_primary2 = self.primary_conv2(x)
        x1_cheap1 = self.cheap_operation1(x1_primary1)
        x1_cheap2 = self.cheap_operation2(x1_primary2)
        x1 = torch.cat([x1_primary1, x1_primary2, x1_cheap1, x1_cheap2], dim=1)

        # 第二层多尺度Ghost卷积
        x2_primary1 = self.conv2_primary1(x1)
        x2_primary2 = self.conv2_primary2(x1)
        x2_cheap1 = self.conv2_cheap1(x2_primary1)
        x2_cheap2 = self.conv2_cheap2(x2_primary2)
        x2 = torch.cat([x2_primary1, x2_primary2, x2_cheap1, x2_cheap2], dim=1)

        return x2 + res


# 第三阶段专用：增强空间图卷积模块
class EnhancedSpatialModule(nn.Module):
    def __init__(self, channels, num_joints):
        super().__init__()
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(1)

        # 动态图学习
        self.graph_learner = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, num_joints * num_joints),
            nn.Sigmoid()
        )

        # 多层图卷积
        self.graph_conv1 = nn.Linear(channels, channels // 2)
        self.graph_conv2 = nn.Linear(channels // 2, channels // 2)
        self.graph_conv3 = nn.Linear(channels // 2, channels)

        self.bn1 = nn.BatchNorm1d(channels // 2)
        self.bn2 = nn.BatchNorm1d(channels // 2)
        self.bn3 = nn.BatchNorm1d(channels)

    def forward(self, x, static_adj):
        batch_size, seq_len, num_joints, channels = x.shape

        # 学习动态邻接矩阵
        x_pooled = x.mean(dim=1).reshape(batch_size * num_joints, channels)
        dynamic_adj = self.graph_learner(x_pooled).view(batch_size, num_joints, num_joints)

        # 结合静态和动态图
        combined_adj = 0.7 * static_adj.unsqueeze(0) + 0.3 * dynamic_adj

        # 多层图卷积
        x_flat = x.reshape(batch_size * seq_len, num_joints, channels)

        # 第一层
        x1 = self.graph_conv1(x_flat)
        x1 = torch.einsum('bij,bjk->bik',
                          combined_adj.unsqueeze(1).expand(-1, seq_len, -1, -1).reshape(-1, num_joints, num_joints), x1)
        x1 = self.bn1(x1.transpose(1, 2)).transpose(1, 2)
        x1 = F.relu(x1)

        # 第二层
        x2 = self.graph_conv2(x1)
        x2 = torch.einsum('bij,bjk->bik',
                          combined_adj.unsqueeze(1).expand(-1, seq_len, -1, -1).reshape(-1, num_joints, num_joints), x2)
        x2 = self.bn2(x2.transpose(1, 2)).transpose(1, 2)
        x2 = F.relu(x2)

        # 第三层
        x3 = self.graph_conv3(x2)
        x3 = torch.einsum('bij,bjk->bik',
                          combined_adj.unsqueeze(1).expand(-1, seq_len, -1, -1).reshape(-1, num_joints, num_joints), x3)
        x3 = self.bn3(x3.transpose(1, 2)).transpose(1, 2)

        return x3.reshape(batch_size, seq_len, num_joints, channels)


# -------------------------------
# 增强的时空图卷积网络
# -------------------------------
class SpatioTemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_joints=6, stage='early', dropout=0.1):
        """
        增强的时空图卷积网络，用于处理IMU数据的时空关系
        """
        super().__init__()
        self.sensor_joints = num_joints
        self.stage = stage

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
            self.map_sensor2joints = nn.Conv1d(in_channels=self.sensor_joints,
                                               out_channels=self.num_joints,
                                               kernel_size=1,
                                               stride=1)
        else:  # late stage
            self.graph = Graph_A(strategy="uniform")
            self.num_joints = 24
            self.A = self.graph.A_a
            self.need_mapping = False

        self.dropout_rate = dropout

        # 构建IMU节点图结构
        graph_adj = torch.tensor(self.A, dtype=torch.float32, requires_grad=False)
        if len(graph_adj.shape) == 3:
            graph_adj = graph_adj[0]
        self.register_buffer('adjacency', graph_adj)

        # 拓扑自适应层 - 学习调整节点间连接强度
        self.edge_importance = nn.Parameter(torch.ones_like(graph_adj))

        # 多层图卷积
        self.gcn1 = GraphConvBlock(in_channels, hidden_dim // 4, self.adjacency)
        self.gcn2 = GraphConvBlock(hidden_dim // 4, hidden_dim // 2, self.adjacency)
        self.gcn3 = GraphConvBlock(hidden_dim // 2, hidden_dim, self.adjacency)

        # 第三阶段专用：增强空间模块
        if stage == 'late':
            self.enhanced_spatial = EnhancedSpatialModule(hidden_dim, self.num_joints)

        # 多层时间卷积
        self.tcn1 = TemporalConvBlock(
            in_channels=hidden_dim * self.num_joints,
            out_channels=hidden_dim * self.num_joints // 2,
            kernel_size=3
        )
        self.tcn2 = TemporalConvBlock(
            in_channels=hidden_dim * self.num_joints // 2,
            out_channels=hidden_dim * self.num_joints,
            kernel_size=5
        )

        # 增强输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_joints, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        """
        增强的时空图卷积前向传播
        """
        batch_size, seq_len, input_joints, features_per_channel = x.shape

        if self.need_mapping == True:
            x_reshape = x.reshape(batch_size * seq_len, self.sensor_joints, features_per_channel)
            x_mapped = self.map_sensor2joints(x_reshape)
            x = x_mapped.reshape(batch_size, seq_len, self.num_joints, features_per_channel)

        # 加入边重要性权重
        adj_with_importance = self.adjacency * self.edge_importance

        # 多层图卷积
        x_gcn1 = self.gcn1(x, adj_with_importance)
        x_gcn2 = self.gcn2(x_gcn1, adj_with_importance)
        x_gcn3 = self.gcn3(x_gcn2, adj_with_importance)

        # 第三阶段专用增强
        if self.stage == 'late':
            x_gcn3 = self.enhanced_spatial(x_gcn3, self.adjacency)

        # 多层时间卷积
        x_reshaped = x_gcn3.reshape(batch_size, seq_len, -1).transpose(1, 2)
        x_temporal1 = self.tcn1(x_reshaped)
        x_temporal2 = self.tcn2(x_temporal1).transpose(1, 2)

        # 增强输出层
        output = self.output_layer(x_temporal2)

        return output


# -----------------------------
# 非对称滑动窗口双向GRU模块（保持原样）
# -----------------------------
class AsymmetricBidirectionalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, stage='early', frame_rate=60):
        """
        非对称滑动窗口双向GRU，针对局部动态建模优化
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stage = stage
        self.frame_rate = frame_rate

        if stage == 'early':
            self.window_size = int(self.frame_rate * 0.8)
            self.current_frame_idx = int(self.window_size * 2 // 3)
            self.droupout = 0.2
            self.stride = max(1, frame_rate // 10)
        elif stage == 'mid':
            self.window_size = int(self.frame_rate * 0.6)
            self.current_frame_idx = int(self.window_size * 2 // 3)
            self.droupout = 0.2
            self.stride = max(1, frame_rate // 15)
        else:
            self.window_size = int(self.frame_rate * 0.4)
            self.current_frame_idx = int(self.window_size * 2 // 3)
            self.droupout = 0.3
            self.stride = max(1, frame_rate // 20)

        # 前向GRU
        self.forward_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.droupout
        )

        # 后向GRU
        self.backward_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.droupout
        )

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # 时间注意力机制用于加权融合窗口内特征
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        非对称滑动窗口双向GRU前向传播 (矢量化高性能版)
        """
        batch_size, seq_len, feat_dim = x.shape
        half_left = self.current_frame_idx
        half_right = self.window_size - half_left - 1

        # 对输入序列进行一次性填充，适配中心化窗口
        x_padded = F.pad(x.transpose(1, 2), (half_left, half_right), mode='replicate').transpose(1, 2)

        # 使用 unfold 高效提取所有滑动窗口
        windows = x_padded.transpose(1, 2).unfold(dimension=2, size=self.window_size, step=self.stride)

        # 计算输出序列的长度
        output_seq_len = windows.shape[2]

        windows = windows.permute(0, 2, 3, 1)
        windows = windows.reshape(batch_size * output_seq_len, self.window_size, feat_dim)

        # 前向 GRU 处理
        _, forward_hidden = self.forward_gru(windows[:, :self.current_frame_idx + 1])
        forward_hidden = forward_hidden[-1]

        _, backward_hidden = self.backward_gru(torch.flip(windows[:, self.current_frame_idx:], dims=[1]))
        backward_hidden = backward_hidden[-1]

        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        # 将形状重塑为 [B, L_out, H]
        outputs = combined.view(batch_size, output_seq_len, self.hidden_dim)

        outputs = self.norm(outputs)
        return outputs


# ===========================================
# 轻量化自定义多头注意力模块
# ===========================================
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True, rank_ratio=0.5):
        super().__init__()
        assert batch_first, "CustomMultiheadAttention 仅支持 batch_first=True"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 修复：确保head_dim是整数，并调整embed_dim使其能被num_heads整除
        if embed_dim % num_heads != 0:
            # 向上调整embed_dim到最近的能被num_heads整除的数
            adjusted_embed_dim = ((embed_dim + num_heads - 1) // num_heads) * num_heads
            self.embed_dim = adjusted_embed_dim
            # 添加投影层来匹配维度
            self.input_proj = nn.Linear(embed_dim, adjusted_embed_dim)
            self.output_proj_final = nn.Linear(adjusted_embed_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()
            self.output_proj_final = nn.Identity()

        self.head_dim = self.embed_dim // num_heads
        self.rank = max(1, int(self.embed_dim * rank_ratio))  # 低秩维度

        # 低秩分解的QKV投影
        self.q_down = nn.Linear(self.embed_dim, self.rank, bias=False)
        self.q_up = nn.Linear(self.rank, self.embed_dim)

        self.k_down = nn.Linear(self.embed_dim, self.rank, bias=False)
        self.k_up = nn.Linear(self.rank, self.embed_dim)

        self.v_down = nn.Linear(self.embed_dim, self.rank, bias=False)
        self.v_up = nn.Linear(self.rank, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, key, value, temperature=1.0):
        batch_size, seq_len, _ = query.size()

        # 维度调整
        query = self.input_proj(query)
        key = self.input_proj(key)
        value = self.input_proj(value)

        # 低秩分解的QKV投影
        q = self.q_up(self.q_down(query))
        k = self.k_up(self.k_down(key))
        v = self.v_up(self.v_down(value))

        # 重塑为多头格式
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 线性注意力近似（避免计算完整的attention矩阵）
        scaling_factor = float(self.head_dim) ** -0.5

        # 使用核技巧计算注意力
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 计算分母
        k_sum = k.sum(dim=-2, keepdim=True)  # [B, H, 1, D]
        denominator = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(-2))

        # 计算分子
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        numerator = torch.einsum('bhnd,bhde->bhne', q, kv)

        # 最终输出
        output = numerator / (denominator.unsqueeze(-1) + 1e-6)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)

        # 维度还原
        output = self.output_proj_final(output)

        return output, None  # 返回None作为attention weights


# DualAttentionLayer 增加更多层
class DualAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, expansion_ratio=3):
        super().__init__()

        # 更多注意力头
        self.self_attn_body = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn_limb = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_body2limb = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_limb2body = CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # 增强前馈网络
        expanded_dim = embed_dim * expansion_ratio

        self.ffn_body = nn.Sequential(
            nn.Linear(embed_dim, expanded_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim // 2, expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(expanded_dim, expanded_dim, 3, padding=1, groups=expanded_dim),
            nn.BatchNorm1d(expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim, expanded_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim // 2, embed_dim),
            nn.Dropout(dropout)
        )

        self.ffn_limb = nn.Sequential(
            nn.Linear(embed_dim, expanded_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim // 2, expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(expanded_dim, expanded_dim, 3, padding=1, groups=expanded_dim),
            nn.BatchNorm1d(expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim, expanded_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim // 2, embed_dim),
            nn.Dropout(dropout)
        )

        # 增强融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU6(inplace=True),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

        # 归一化层
        self.norm1_body = nn.LayerNorm(embed_dim)
        self.norm2_body = nn.LayerNorm(embed_dim)
        self.norm3_body = nn.LayerNorm(embed_dim)
        self.norm1_limb = nn.LayerNorm(embed_dim)
        self.norm2_limb = nn.LayerNorm(embed_dim)
        self.norm3_limb = nn.LayerNorm(embed_dim)


def forward(self, body_feats, limb_feats, temperature=1.0):
    # 自注意力
    body_self, _ = self.self_attn_body(body_feats, body_feats, body_feats, temperature=temperature)
    body = self.norm1_body(body_feats + body_self)

    limb_self, _ = self.self_attn_limb(limb_feats, limb_feats, limb_feats, temperature=temperature)
    limb = self.norm1_limb(limb_feats + limb_self)

    # 交叉注意力
    body_cross, _ = self.cross_attn_body2limb(body, limb, limb, temperature=temperature)
    limb_cross, _ = self.cross_attn_limb2body(limb, body, body, temperature=temperature)

    # 自适应融合
    body_combined = torch.cat([body, body_cross], dim=-1)
    limb_combined = torch.cat([limb, limb_cross], dim=-1)

    body_gate = self.fusion_gate(body_combined)
    limb_gate = self.fusion_gate(limb_combined)

    body = self.norm2_body(body * body_gate[..., 0:1] + body_cross * body_gate[..., 1:2])
    limb = self.norm2_limb(limb * limb_gate[..., 0:1] + limb_cross * limb_gate[..., 1:2])

    # 增强前馈网络（处理维度转换）
    batch_size, seq_len, embed_dim = body.shape

    # Body FFN
    body_expanded = self.ffn_body[0](body)
    body_activated1 = self.ffn_body[1](body_expanded)
    body_expanded2 = self.ffn_body[2](body_activated1)
    body_activated2 = self.ffn_body[3](body_expanded2)

    body_conv_input = body_activated2.transpose(1, 2)
    body_conv_out = self.ffn_body[4](body_conv_input)
    body_conv_out = self.ffn_body[5](body_conv_out)
    body_conv_out = self.ffn_body[6](body_conv_out)
    body_conv_out = body_conv_out.transpose(1, 2)

    body_compressed1 = self.ffn_body[7](body_conv_out)
    body_activated3 = self.ffn_body[8](body_compressed1)
    body_ffn_out = self.ffn_body[9](body_activated3)
    body_ffn_out = self.ffn_body[10](body_ffn_out)

    # Limb FFN (同样的处理)
    limb_expanded = self.ffn_limb[0](limb)
    limb_activated1 = self.ffn_limb[1](limb_expanded)
    limb_expanded2 = self.ffn_limb[2](limb_activated1)
    limb_activated2 = self.ffn_limb[3](limb_expanded2)

    limb_conv_input = limb_activated2.transpose(1, 2)
    limb_conv_out = self.ffn_limb[4](limb_conv_input)
    limb_conv_out = self.ffn_limb[5](limb_conv_out)
    limb_conv_out = self.ffn_limb[6](limb_conv_out)
    limb_conv_out = limb_conv_out.transpose(1, 2)

    limb_compressed1 = self.ffn_limb[7](limb_conv_out)
    limb_activated3 = self.ffn_limb[8](limb_compressed1)
    limb_ffn_out = self.ffn_limb[9](limb_activated3)
    limb_ffn_out = self.ffn_limb[10](limb_ffn_out)

    body = self.norm3_body(body + body_ffn_out)
    limb = self.norm3_limb(limb + limb_ffn_out)

    return body, limb


# 增强的 DualStreamTransformerFusion
class DualStreamTransformerFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=3, output_dim=15, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DualAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 增强融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 增强姿态回归器
        self.pose_regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.ReLU6(inplace=True),
            nn.Linear(embed_dim // 8, output_dim)
        )

    def forward(self, body_feats, limb_feats, temperature=1.0):
        for layer in self.layers:
            body_feats, limb_feats = layer(body_feats, limb_feats, temperature=temperature)

        fused_features = self.fusion(torch.cat([body_feats, limb_feats], dim=-1))
        pose_params = self.pose_regressor(fused_features)
        return pose_params


# 第三阶段专用：多模态特征增强模块
class MultiModalEnhancement(nn.Module):
    """第三阶段专用的多模态特征增强模块"""

    def __init__(self, feature_dim, num_modalities=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # 模态特定的特征增强
        self.modality_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.LayerNorm(feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim * 2),
                nn.LayerNorm(feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim * 2, feature_dim)
            ) for _ in range(num_modalities)
        ])

        # 模态间交互学习
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 自适应权重学习
        self.adaptive_weights = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, modality_features):
        """
        modality_features: list of [B, L, D] tensors
        """
        enhanced_features = []

        # 模态特定增强
        for i, (features, enhancer) in enumerate(zip(modality_features, self.modality_enhancers)):
            enhanced = enhancer(features)
            enhanced_features.append(enhanced)

        # 模态间交互
        if len(enhanced_features) >= 2:
            # 使用第一个模态查询其他模态
            query = enhanced_features[0]
            for i in range(1, len(enhanced_features)):
                key_value = enhanced_features[i]
                attended, _ = self.cross_modal_attention(query, key_value, key_value)
                enhanced_features[0] = enhanced_features[0] + attended

        # 自适应加权融合
        stacked_features = torch.stack(enhanced_features, dim=-1)  # [B, L, D, M]
        concat_for_weights = torch.cat(enhanced_features, dim=-1)  # [B, L, D*M]
        weights = self.adaptive_weights(concat_for_weights).unsqueeze(-2)  # [B, L, 1, M]

        fused_features = (stacked_features * weights).sum(dim=-1)  # [B, L, D]

        return fused_features


# 主模型 DSTFPE 的增强版
class DSTFPE(nn.Module):
    def __init__(self, num_nodes=6, trunk_dim=108, limb_dim=108, hidden_dim=512, output_dim=15, num_heads=8,
                 stage='early'):
        super().__init__()
        self.stage = stage

        # 阶段特定配置 - 修正num_heads确保能被hidden_dim整除
        def find_valid_heads(hidden_dim, preferred_heads):
            """找到能被hidden_dim整除的最接近的头数"""
            for heads in [preferred_heads, preferred_heads - 1, preferred_heads + 1,
                          preferred_heads - 2, preferred_heads + 2, 8, 4, 2, 1]:
                if hidden_dim % heads == 0 and heads > 0:
                    return heads
            return 1

        stage_config = {
            'early': {'temperature': 2.0, 'dropout': 0.1, 'num_layers': 2, 'preferred_heads': 8},
            'mid': {'temperature': 1.0, 'dropout': 0.15, 'num_layers': 3, 'preferred_heads': 10},
            'late': {'temperature': 0.5, 'dropout': 0.2, 'num_layers': 4, 'preferred_heads': 12}
        }
        config = stage_config.get(self.stage, stage_config['mid'])
        self.temperature = config['temperature']
        self.dropout_rate = config['dropout']

        # 确保num_heads能被hidden_dim整除
        valid_num_heads = find_valid_heads(hidden_dim, config['preferred_heads'])

        # 增强的时空图卷积
        self.st_gcn = SpatioTemporalGCN(
            in_channels=trunk_dim // num_nodes,
            hidden_dim=hidden_dim,
            num_joints=num_nodes,
            stage=self.stage,
            dropout=self.dropout_rate
        )

        # 保持双向GRU
        self.bi_gru = AsymmetricBidirectionalGRU(
            input_dim=limb_dim,
            hidden_dim=hidden_dim,
            stage=self.stage
        )

        # 增强融合模块
        self.transformer_fusion = DualStreamTransformerFusion(
            embed_dim=hidden_dim,
            num_heads=valid_num_heads,
            num_layers=config['num_layers'],
            output_dim=output_dim,
            dropout=self.dropout_rate
        )

    def forward(self, trunk_features, limb_features):
        # 全局路径：增强的时空图卷积
        gcn_out = self.st_gcn(trunk_features)

        # 细节路径：双向GRU
        gru_out = self.bi_gru(limb_features)

        # 对GRU输出进行上采样以匹配GCN序列长度
        target_seq_len = gcn_out.shape[1]

        if gru_out.shape[1] != target_seq_len:
            gru_out_permuted = gru_out.permute(0, 2, 1)
            gru_out_upsampled = F.interpolate(
                gru_out_permuted,
                size=target_seq_len,
                mode='linear',
                align_corners=False
            )
            gru_out = gru_out_upsampled.permute(0, 2, 1)

        # 使用阶段特定温度进行增强融合
        joint_positions = self.transformer_fusion(gcn_out, gru_out, temperature=self.temperature)

        return joint_positions


# 第三阶段专用：增强版DSTFPE
class EnhancedDSTFPE(nn.Module):
    """第三阶段专用的增强版DSTFPE，包含额外的多模态增强模块"""

    def __init__(self, num_nodes=6, trunk_dim=108, limb_dim=108, hidden_dim=512, output_dim=15, num_heads=12,
                 stage='late'):
        super().__init__()
        self.stage = stage

        # 第三阶段专用配置
        self.temperature = 0.3  # 更低的温度用于精细化
        self.dropout_rate = 0.25

        # 确保num_heads能被hidden_dim整除
        def find_valid_heads(hidden_dim, preferred_heads):
            for heads in [preferred_heads, preferred_heads - 1, preferred_heads + 1,
                          preferred_heads - 2, preferred_heads + 2, 8, 4, 2, 1]:
                if hidden_dim % heads == 0 and heads > 0:
                    return heads
            return 1

        valid_num_heads = find_valid_heads(hidden_dim, num_heads)

        # 增强的时空图卷积
        self.st_gcn = SpatioTemporalGCN(
            in_channels=trunk_dim // num_nodes,
            hidden_dim=hidden_dim,
            num_joints=num_nodes,
            stage=self.stage,
            dropout=self.dropout_rate
        )

        # 双向GRU
        self.bi_gru = AsymmetricBidirectionalGRU(
            input_dim=limb_dim,
            hidden_dim=hidden_dim,
            stage=self.stage
        )

        # 第三阶段专用：多模态特征增强
        self.multimodal_enhancement = MultiModalEnhancement(
            feature_dim=hidden_dim,
            num_modalities=2
        )

        # 增强的融合模块（更多层数）
        self.transformer_fusion = DualStreamTransformerFusion(
            embed_dim=hidden_dim,
            num_heads=valid_num_heads,
            num_layers=5,  # 第三阶段使用更多层
            output_dim=output_dim,
            dropout=self.dropout_rate
        )

        # 第三阶段专用：残差精化模块
        self.residual_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim * 2, output_dim)
            ) for _ in range(3)  # 3层残差精化
        ])

    def forward(self, trunk_features, limb_features):
        # 全局路径
        gcn_out = self.st_gcn(trunk_features)

        # 细节路径
        gru_out = self.bi_gru(limb_features)

        # 序列长度匹配
        target_seq_len = gcn_out.shape[1]
        if gru_out.shape[1] != target_seq_len:
            gru_out_permuted = gru_out.permute(0, 2, 1)
            gru_out_upsampled = F.interpolate(
                gru_out_permuted,
                size=target_seq_len,
                mode='linear',
                align_corners=False
            )
            gru_out = gru_out_upsampled.permute(0, 2, 1)

        # 第三阶段专用：多模态特征增强
        enhanced_features = self.multimodal_enhancement([gcn_out, gru_out])

        # 使用增强后的特征进行融合
        joint_positions = self.transformer_fusion(enhanced_features, enhanced_features, temperature=self.temperature)

        # 第三阶段专用：残差精化
        refined_output = joint_positions
        for refine_layer in self.residual_refinement:
            residual = refine_layer(refined_output)
            refined_output = refined_output + residual

        return refined_output


# 根据阶段选择模型的工厂函数
def create_dstfpe_model(stage='early', **kwargs):
    """根据阶段创建相应的DSTFPE模型"""
    if stage == 'late':
        return EnhancedDSTFPE(stage=stage, **kwargs)
    else:
        return DSTFPE(stage=stage, **kwargs)
