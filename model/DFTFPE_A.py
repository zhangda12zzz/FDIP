# ----------------------------
# 轻量化时空图卷积网络（仅第三阶段增强表达能力）
# ----------------------------

import torch
from torch import nn
import torch.nn.functional as F
from model.graph import Graph_B, Graph_A
from torch.nn import Linear


# ===========================
# 注意力模块
# ===========================
class CustomMultiheadAttention(nn.Module):
    """低秩 + 线性注意力近似（轻量化版本，用于 early/mid）"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True, rank_ratio=0.5):
        super().__init__()
        assert batch_first, "CustomMultiheadAttention 仅支持 batch_first=True"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.rank = max(1, int(embed_dim * rank_ratio))
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 必须能被 num_heads 整除"

        # 低秩分解的QKV投影
        self.q_down = nn.Linear(embed_dim, self.rank, bias=False)
        self.q_up = nn.Linear(self.rank, embed_dim)
        self.k_down = nn.Linear(embed_dim, self.rank, bias=False)
        self.k_up = nn.Linear(self.rank, embed_dim)
        self.v_down = nn.Linear(embed_dim, self.rank, bias=False)
        self.v_up = nn.Linear(self.rank, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, temperature=1.0):
        batch_size, seq_len, _ = query.size()

        # 低秩分解的QKV投影
        q = self.q_up(self.q_down(query))
        k = self.k_up(self.k_down(key))
        v = self.v_up(self.v_down(value))

        # 重塑为多头格式
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 线性注意力近似（核技巧）
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 分母: \sum_j q_i^T k_j
        k_sum = k.sum(dim=-2, keepdim=True)  # [B, H, 1, D]
        denominator = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(-2))

        # 分子: q_i^T (\sum_j k_j v_j^T)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        numerator = torch.einsum('bhnd,bhde->bhne', q, kv)

        # 输出
        output = numerator / (denominator.unsqueeze(-1) + 1e-6)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        return output, None


class StrongMultiheadAttention(nn.Module):
    """标准多头注意力（表达力更强，用于 late）"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, query, key, value, temperature=1.0):
        # temperature 可以用于缩放 qk^T，但这里交给 Pytorch 内部处理；保留参数以保持接口一致
        out, attn = self.mha(query, key, value, need_weights=False)
        return out, attn


# ===========================
# 图卷积块
# ===========================
class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, reduction_ratio=4):
        super(GraphConvBlock, self).__init__()
        self.A = A

        # 深度可分离图卷积（Linear 代替）
        self.depthwise_conv = nn.Linear(in_channels, in_channels, bias=False)
        self.pointwise_conv = nn.Linear(in_channels, out_channels)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(out_channels, max(1, out_channels // reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, out_channels // reduction_ratio), out_channels),
            nn.Sigmoid()
        )

        # 轻量化残差
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Linear(in_channels, max(1, out_channels // 4)),
                nn.Linear(max(1, out_channels // 4), out_channels),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A=None):
        """输入: [B, T, N, C_in]"""
        batch_size, seq_len, num_nodes, _ = x.shape

        # 残差
        res = self.residual(x.reshape(batch_size * seq_len * num_nodes, -1))
        res = res.reshape(batch_size, seq_len, num_nodes, -1)

        # 深度可分离图卷积
        x_depth = self.depthwise_conv(x)
        adj_matrix = A if A is not None else self.A
        x_conv = torch.einsum('nm,bsmd->bsnd', adj_matrix, x_depth)
        x_point = self.pointwise_conv(x_conv)

        # 通道注意力
        x_flat = x_point.reshape(batch_size * seq_len, num_nodes, -1).transpose(1, 2)
        pooled = x_flat.mean(dim=2)
        attention = self.channel_attention(pooled).unsqueeze(-1)
        x_flat = x_flat * attention

        x_flat = self.bn(x_flat)
        x_flat = x_flat.transpose(1, 2)
        x_conv = x_flat.reshape(batch_size, seq_len, num_nodes, -1)

        out = self.relu(x_conv + res)
        return out


# ===========================
# 轻量化时间卷积块（Ghost Conv）
# ===========================
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=4):
        super(TemporalConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        groups = min(groups, in_channels)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size, padding=padding, groups=groups),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv1d(out_channels // 2, out_channels // 2, 3, padding=1, groups=out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2_primary = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 2, kernel_size, padding=padding, groups=min(groups, out_channels)),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2_cheap = nn.Sequential(
            nn.Conv1d(out_channels // 2, out_channels // 2, 3, padding=1, groups=out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """输入: [B, C, T] -> 输出: [B, C_out, T]"""
        res = self.residual(x)
        x1_primary = self.primary_conv(x)
        x1_cheap = self.cheap_operation(x1_primary)
        x1 = torch.cat([x1_primary, x1_cheap], dim=1)
        x2_primary = self.conv2_primary(x1)
        x2_cheap = self.conv2_cheap(x2_primary)
        x2 = torch.cat([x2_primary, x2_cheap], dim=1)
        return x2 + res


# ===========================
# 时空图卷积网络（ST-GCN）
# ===========================
class SpatioTemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_joints=6, stage='early', dropout=0.1):
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
            self.need_mapping = True    # 6 -> 24
            self.map_sensor2joints = nn.Conv1d(in_channels=self.sensor_joints,
                                               out_channels=self.num_joints,
                                               kernel_size=1,
                                               stride=1)
        else:
            self.graph = Graph_A(strategy="uniform")
            self.num_joints = 24
            self.A = self.graph.A_a
            self.need_mapping = False

        self.dropout_rate = dropout

        # 邻接矩阵
        graph_adj = torch.tensor(self.A, dtype=torch.float32, requires_grad=False)
        if len(graph_adj.shape) == 3:
            graph_adj = graph_adj[0]
        self.register_buffer('adjacency', graph_adj)

        # 可学习边重要性
        self.edge_importance = nn.Parameter(torch.ones_like(graph_adj))

        # 轻量化图卷积层
        self.gcn1 = GraphConvBlock(in_channels, hidden_dim // 2, self.adjacency)
        self.gcn2 = GraphConvBlock(hidden_dim // 2, hidden_dim, self.adjacency)

        # 时间卷积层（late 可以适当放宽感受野）
        ks = 7 if stage == 'late' else 5
        self.tcn = TemporalConvBlock(
            in_channels=hidden_dim * self.num_joints,
            out_channels=hidden_dim * self.num_joints,
            kernel_size=ks
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_joints, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        batch_size, seq_len, input_joints, features_per_channel = x.shape

        if self.need_mapping:
            x_reshape = x.reshape(batch_size * seq_len, self.sensor_joints, features_per_channel)
            x_mapped = self.map_sensor2joints(x_reshape)
            x = x_mapped.reshape(batch_size, seq_len, self.num_joints, features_per_channel)

        adj_with_importance = self.adjacency * self.edge_importance
        x_gcn1 = self.gcn1(x, adj_with_importance)
        x_gcn2 = self.gcn2(x_gcn1, adj_with_importance)

        x_reshaped = x_gcn2.reshape(batch_size, seq_len, -1).transpose(1, 2)
        x_temporal = self.tcn(x_reshaped).transpose(1, 2)
        output = self.output_layer(x_temporal)
        return output


# ===========================
# 非对称滑动窗口双向GRU（保持原样）
# ===========================
class AsymmetricBidirectionalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, stage='early', frame_rate=60):
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

        self.forward_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.droupout
        )
        self.backward_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.droupout
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        half_left = self.current_frame_idx
        half_right = self.window_size - half_left - 1

        x_padded = F.pad(x.transpose(1, 2), (half_left, half_right), mode='replicate').transpose(1, 2)
        windows = x_padded.transpose(1, 2).unfold(dimension=2, size=self.window_size, step=self.stride)
        output_seq_len = windows.shape[2]

        windows = windows.permute(0, 2, 3, 1).reshape(batch_size * output_seq_len, self.window_size, feat_dim)

        _, forward_hidden = self.forward_gru(windows[:, :self.current_frame_idx + 1])
        forward_hidden = forward_hidden[-1]
        _, backward_hidden = self.backward_gru(torch.flip(windows[:, self.current_frame_idx:], dims=[1]))
        backward_hidden = backward_hidden[-1]

        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        outputs = combined.view(batch_size, output_seq_len, self.hidden_dim)
        outputs = self.norm(outputs)
        return outputs


# ===========================
# 双流注意力融合层（late 阶段增强）
# ===========================
class DualAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, expansion_ratio=2, strength='lite'):
        super().__init__()
        self.strength = strength  # 'lite' for early/mid, 'strong' for late

        # 注意力实现：late 使用更强的标准 MHA
        Attn = StrongMultiheadAttention if strength == 'strong' else CustomMultiheadAttention
        self.self_attn_body = Attn(embed_dim, num_heads, dropout=dropout)
        self.self_attn_limb = Attn(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_body2limb = Attn(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_limb2body = Attn(embed_dim, num_heads, dropout=dropout)

        # FFN：late 扩宽
        expanded_dim = embed_dim * (4 if strength == 'strong' else expansion_ratio)

        self.ffn_body = nn.Sequential(
            nn.Linear(embed_dim, expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(expanded_dim, expanded_dim, 3, padding=1, groups=expanded_dim),
            nn.BatchNorm1d(expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_limb = nn.Sequential(
            nn.Linear(embed_dim, expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(expanded_dim, expanded_dim, 3, padding=1, groups=expanded_dim),
            nn.BatchNorm1d(expanded_dim),
            nn.ReLU6(inplace=True),
            nn.Linear(expanded_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # 归一化
        self.norm1_body = nn.LayerNorm(embed_dim)
        self.norm2_body = nn.LayerNorm(embed_dim)
        self.norm3_body = nn.LayerNorm(embed_dim)
        self.norm1_limb = nn.LayerNorm(embed_dim)
        self.norm2_limb = nn.LayerNorm(embed_dim)
        self.norm3_limb = nn.LayerNorm(embed_dim)

        # 简化门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),
            nn.ReLU6(inplace=True),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, body_feats, limb_feats, temperature=1.0):
        # 1) 自注意力
        body_self, _ = self.self_attn_body(body_feats, body_feats, body_feats, temperature=temperature)
        body = self.norm1_body(body_feats + body_self)
        limb_self, _ = self.self_attn_limb(limb_feats, limb_feats, limb_feats, temperature=temperature)
        limb = self.norm1_limb(limb_feats + limb_self)

        # 2) 交叉注意力（一次）
        body_cross, _ = self.cross_attn_body2limb(body, limb, limb, temperature=temperature)
        limb_cross, _ = self.cross_attn_limb2body(limb, body, body, temperature=temperature)

        # late：进行第二次交叉注意力以增强跨模态交互
        if self.strength == 'strong':
            body_cross2, _ = self.cross_attn_body2limb(body, limb, limb, temperature=temperature)
            limb_cross2, _ = self.cross_attn_limb2body(limb, body, body, temperature=temperature)
            body_cross = body_cross + 0.5 * body_cross2
            limb_cross = limb_cross + 0.5 * limb_cross2

        # 融合门控
        body_combined = torch.cat([body, body_cross], dim=-1)
        limb_combined = torch.cat([limb, limb_cross], dim=-1)
        body_gate = self.fusion_gate(body_combined)
        limb_gate = self.fusion_gate(limb_combined)

        body = self.norm2_body(body * body_gate[..., 0:1] + body_cross * body_gate[..., 1:2])
        limb = self.norm2_limb(limb * limb_gate[..., 0:1] + limb_cross * limb_gate[..., 1:2])

        # 3) FFN（MobileNet 风格，带深度卷积）
        # Body
        body_expanded = self.ffn_body[0](body)
        body_activated = self.ffn_body[1](body_expanded)
        body_conv_out = self.ffn_body[2](body_activated.transpose(1, 2))
        body_conv_out = self.ffn_body[3](body_conv_out)
        body_conv_out = self.ffn_body[4](body_conv_out)
        body_conv_out = body_conv_out.transpose(1, 2)
        body_ffn_out = self.ffn_body[5](body_conv_out)
        body_ffn_out = self.ffn_body[6](body_ffn_out)

        # Limb
        limb_expanded = self.ffn_limb[0](limb)
        limb_activated = self.ffn_limb[1](limb_expanded)
        limb_conv_out = self.ffn_limb[2](limb_activated.transpose(1, 2))
        limb_conv_out = self.ffn_limb[3](limb_conv_out)
        limb_conv_out = self.ffn_limb[4](limb_conv_out)
        limb_conv_out = limb_conv_out.transpose(1, 2)
        limb_ffn_out = self.ffn_limb[5](limb_conv_out)
        limb_ffn_out = self.ffn_limb[6](limb_ffn_out)

        body = self.norm3_body(body + body_ffn_out)
        limb = self.norm3_limb(limb + limb_ffn_out)
        return body, limb


class DualStreamTransformerFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=3, output_dim=15, dropout=0.1, strength='lite'):
        super().__init__()
        self.layers = nn.ModuleList([
            DualAttentionLayer(embed_dim, num_heads, dropout, expansion_ratio=2, strength=strength)
            for _ in range(num_layers)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.pose_regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU6(inplace=True),
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


# ===========================
# 顶层网络：DSTFPE
# ===========================
class DSTFPE(nn.Module):
    def __init__(self, num_nodes=6, trunk_dim=108, limb_dim=108, hidden_dim=512, output_dim=15, num_heads=8,
                 stage='early'):
        super().__init__()
        self.stage = stage

        # 阶段特定温度与 dropout
        stage_config = {
            'early': {'temperature': 2.0, 'dropout': 0.2},
            'mid':   {'temperature': 1.0, 'dropout': 0.2},
            'late':  {'temperature': 0.5, 'dropout': 0.4}
        }
        config = stage_config.get(self.stage, stage_config['mid'])
        self.temperature = config['temperature']
        self.dropout_rate = config['dropout']

        # 全局路径：ST-GCN
        self.st_gcn = SpatioTemporalGCN(
            in_channels=trunk_dim // num_nodes,
            hidden_dim=hidden_dim,
            num_joints=num_nodes,
            stage=self.stage,
            dropout=self.dropout_rate
        )

        # 细节路径：Bi-GRU
        self.bi_gru = AsymmetricBidirectionalGRU(
            input_dim=limb_dim,
            hidden_dim=hidden_dim,
            stage=self.stage
        )

        # 仅在 late 阶段增强表达力
        fusion_strength = 'strong' if stage == 'late' else 'lite'
        fusion_layers = 6 if stage == 'late' else 2

        self.transformer_fusion = DualStreamTransformerFusion(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=fusion_layers,
            output_dim=output_dim,
            dropout=self.dropout_rate,
            strength=fusion_strength
        )

    def forward(self, trunk_features, limb_features):
        # 全局（ST-GCN）
        gcn_out = self.st_gcn(trunk_features)

        # 细节（Bi-GRU）
        gru_out = self.bi_gru(limb_features)

        # 上采样 GRU 输出到与 GCN 对齐的时间长度
        target_seq_len = gcn_out.shape[1]
        if gru_out.shape[1] != target_seq_len:
            gru_out_permuted = gru_out.permute(0, 2, 1)
            gru_out_upsampled = F.interpolate(
                gru_out_permuted, size=target_seq_len, mode='linear', align_corners=False
            )
            gru_out = gru_out_upsampled.permute(0, 2, 1)

        # 融合与回归
        joint_positions = self.transformer_fusion(gcn_out, gru_out, temperature=self.temperature)
        return joint_positions
