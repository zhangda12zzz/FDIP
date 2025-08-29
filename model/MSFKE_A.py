import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 辅助模块 (被 NodeAwareMSFKE 使用)
# =============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    """
    深度可分离1D卷积模块。
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = nn.ReLU(inplace=True)

        self.mid_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.activation2 = nn.ReLU(inplace=True)

        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        # 添加最后的BatchNorm - 关键位置1
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.mid_conv(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.pointwise_conv(x)
        x = self.bn3(x)
        return x


class BranchAwareSEModule(nn.Module):
    """
    分支感知的权重计算模块 (Squeeze-and-Excitation)。
    """

    def __init__(self, base_channels, num_scales, reduction=8, module_type='trunk', stage='early'):
        super().__init__()
        self.base_channels = base_channels
        self.num_scales = num_scales
        self.module_type = module_type

        dropout_rates = {'early': 0.1, 'mid': 0.1, 'late': 0.4}
        self.dropout = nn.Dropout(dropout_rates.get(stage, 0.1))

        if module_type == 'trunk':
            self.branch_se_modules = nn.ModuleList([
                nn.Sequential(
                    DepthwiseSeparableConv1d(base_channels, base_channels // reduction,
                                             kernel_size=5 if i > num_scales // 2 else 3),
                    nn.ReLU(inplace=True),
                    self.dropout,
                    nn.Conv1d(base_channels // reduction, base_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(base_channels),  # 添加BN - 关键位置2
                    nn.Sigmoid()
                ) for i in range(num_scales)
            ])
            freq_bias_configs = {
                'early': torch.ones(num_scales) * 0.3 + torch.tensor(
                    [i / (num_scales - 1) * 0.7 for i in range(num_scales)]),
                'mid': torch.ones(num_scales) * 0.5,
                'late': torch.ones(num_scales) * 0.3 + torch.tensor(
                    [(num_scales - i - 1) / (num_scales - 1) * 0.7 for i in range(num_scales)])
            }
            self.freq_bias = nn.Parameter(freq_bias_configs.get(stage, torch.ones(num_scales) * 0.5))
        else:  # module_type == 'limb'
            self.branch_se_modules = nn.ModuleList([
                nn.Sequential(
                    DepthwiseSeparableConv1d(base_channels, base_channels // reduction,
                                             kernel_size=3 if i < num_scales // 2 else 1),
                    nn.GELU(),
                    self.dropout,
                    nn.Conv1d(base_channels // reduction, base_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(base_channels),  # 添加BN - 关键位置3
                    nn.Sigmoid()
                ) for i in range(num_scales)
            ])
            default_bias = torch.ones(num_scales) * 0.3 + torch.tensor(
                [(num_scales - 1 - i) / (num_scales - 1) * 0.7 for i in
                 range(num_scales)]) if num_scales > 1 else torch.ones(num_scales)
            freq_bias_configs = {
                'early': torch.ones(num_scales) * 0.4 + torch.tensor(
                    [(num_scales - 1 - i) / (num_scales - 1) * 0.6 for i in
                     range(num_scales)]) if num_scales > 1 else torch.ones(num_scales),
                'mid': torch.ones(num_scales) * 0.5
            }
            self.freq_bias = nn.Parameter(freq_bias_configs.get(stage, default_bias))

        concat_channels = num_scales * base_channels

        self.branch_importance = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(concat_channels, max(1, concat_channels // 2), kernel_size=1, bias=False),
            nn.BatchNorm1d(max(1, concat_channels // 2)),  # 添加BN - 关键位置4
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(max(1, concat_channels // 2), max(1, concat_channels // 4), kernel_size=1, bias=False),
            nn.BatchNorm1d(max(1, concat_channels // 4)),  # 添加BN - 关键位置5
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, concat_channels // 4), num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, branch_features):
        weighted_branches = [se_module(branch) * branch for branch, se_module in
                             zip(branch_features, self.branch_se_modules)]
        concat_features = torch.cat(weighted_branches, dim=1)

        raw_weights = self.branch_importance(concat_features)
        branch_weights = F.softmax(raw_weights * self.freq_bias.view(1, -1, 1), dim=1)

        batch_size, _, seq_len = concat_features.shape
        reshaped_features = concat_features.view(batch_size, self.num_scales, self.base_channels, seq_len)

        # Element-wise multiplication after broadcasting weights
        fused_features = (reshaped_features * branch_weights.unsqueeze(-2)).sum(dim=1)

        return fused_features, branch_weights.squeeze(-1)


class SharedMultiScaleExtractor(nn.Module):
    def __init__(self, input_channels, base_channels, dilation_rates):
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            branch = nn.Sequential(
                nn.Conv1d(input_channels, base_channels // 4, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(base_channels // 4),
                nn.ReLU(inplace=True),

                nn.Conv1d(base_channels // 4, base_channels // 2, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(base_channels // 2),
                nn.ReLU(inplace=True),

                nn.Conv1d(base_channels // 2, base_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

    def forward(self, x):
        return [branch(x) for branch in self.branches]


# =============================================================================
# 核心模型：节点感知型MSFKE (Node-Aware MSFKE)
# =============================================================================
class NodeAwareMSFKE(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_channels: int,
                 base_channels: int = 24,
                 num_scales: int = 5,
                 node_feature_dim: int = 18,
                 stage: str = 'early'):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_channels = input_channels
        self.base_channels_config = base_channels
        self.num_scales = num_scales
        self.node_feature_dim = node_feature_dim
        self.stage = stage
        self.node_embedding_dim = 16

        # --- 1. 根据 num_nodes 和 stage 配置模型 ---
        self.base_channels = self.base_channels_config if self.stage == 'early' else int(
            self.base_channels_config * 1.5)

        if self.num_nodes == 6:
            self.num_groups = 3
            self.adapted_input_channels = self.input_channels
            self.dimension_adapter = None
            self.node_to_group_map = {0: 0, 3: 1, 1: 2, 2: 2, 4: 2, 5: 2}
        elif self.num_nodes == 24:
            self.num_groups = 24
            self.adapted_input_channels = 3
            self.dimension_adapter = nn.Sequential(
                nn.Linear(288, self.num_nodes * self.adapted_input_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(self.num_nodes * self.adapted_input_channels)  # 保持LayerNorm
            )
            self.node_to_group_map = {node_idx: node_idx for node_idx in range(24)}
        else:
            raise ValueError(f"Unsupported number of nodes: {self.num_nodes}. Only 6 and 24 are supported.")

        self.dilation_rates = [1, 3, 7, 15, 31]
        self.fused_channels = self.base_channels

        # --- 2. 定义模型组件 ---
        self.node_embedding = nn.Parameter(torch.randn(24, self.node_embedding_dim))

        self.shared_feature_extractor = SharedMultiScaleExtractor(self.adapted_input_channels, self.base_channels,
                                                                  self.dilation_rates)

        self.trunk_se_modules = nn.ModuleList([
            BranchAwareSEModule(self.base_channels, num_scales, module_type='trunk', stage=stage) for _ in
            range(self.num_groups)
        ])
        self.limb_se_modules = nn.ModuleList([
            BranchAwareSEModule(self.base_channels, num_scales, module_type='limb', stage=stage) for _ in
            range(self.num_groups)
        ])

        projection_input_dim = self.fused_channels + self.node_embedding_dim
        self.shared_projector = nn.Sequential(
            nn.Conv1d(projection_input_dim,
                      max(8, projection_input_dim // 2), kernel_size=1, bias=False),
            nn.BatchNorm1d(max(8, projection_input_dim // 2)),  # 保持原有的BN
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(max(8, projection_input_dim // 2),
                      self.node_feature_dim, kernel_size=1)  # 最后一层不加BN，保持输出灵活性
        )

    def forward(self, imu_data):
        batch_size, seq_len, _ = imu_data.shape

        # --- 1. 输入预处理和重塑 ---
        if self.dimension_adapter is not None:
            imu_data = self.dimension_adapter(imu_data)

        node_inputs = imu_data.view(batch_size, seq_len, self.num_nodes, self.adapted_input_channels)
        parallel_inputs = node_inputs.permute(0, 2, 3, 1).reshape(batch_size * self.num_nodes,
                                                                  self.adapted_input_channels, seq_len)

        # --- 2. 并行多尺度特征提取 ---
        branch_features_list = self.shared_feature_extractor(parallel_inputs)

        # --- 3. 并行节点感知SE处理 ---
        all_trunk_features, all_limb_features = [], []
        all_trunk_weights, all_limb_weights = [], []

        organized_features = [feat.view(batch_size, self.num_nodes, self.base_channels, seq_len) for feat in
                              branch_features_list]

        for node_idx in range(self.num_nodes):
            node_branch_features = [feat[:, node_idx] for feat in organized_features]
            group_idx = self.node_to_group_map[node_idx]

            trunk_feat, trunk_w = self.trunk_se_modules[group_idx](node_branch_features)
            limb_feat, limb_w = self.limb_se_modules[group_idx](node_branch_features)

            all_trunk_features.append(trunk_feat)
            all_limb_features.append(limb_feat)
            all_trunk_weights.append(trunk_w)
            all_limb_weights.append(limb_w)

        # --- 4. 并行融合节点编码和共享投射 ---
        trunk_stacked = torch.stack(all_trunk_features, dim=1)
        limb_stacked = torch.stack(all_limb_features, dim=1)

        node_embeds_sliced = self.node_embedding[:self.num_nodes]
        node_embeds = node_embeds_sliced.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, seq_len)

        trunk_with_embeds = torch.cat([trunk_stacked, node_embeds], dim=2)
        limb_with_embeds = torch.cat([limb_stacked, node_embeds], dim=2)

        c_proj = self.fused_channels + self.node_embedding_dim
        parallel_trunk_to_proj = trunk_with_embeds.reshape(batch_size * self.num_nodes, c_proj, seq_len)
        parallel_limb_to_proj = limb_with_embeds.reshape(batch_size * self.num_nodes, c_proj, seq_len)

        trunk_outputs_parallel = self.shared_projector(parallel_trunk_to_proj)
        limb_outputs_parallel = self.shared_projector(parallel_limb_to_proj)

        # --- 5. 组装最终输出 ---
        c_out = self.node_feature_dim
        trunk_features = trunk_outputs_parallel.view(batch_size, self.num_nodes, c_out, seq_len).permute(0, 3, 1, 2)
        limb_features = limb_outputs_parallel.view(batch_size, self.num_nodes, c_out, seq_len).permute(0, 3, 1, 2)

        trunk_features_combined = trunk_features.reshape(batch_size, seq_len, -1)
        limb_features_combined = limb_features.reshape(batch_size, seq_len, -1)

        return {
            'trunk_features': trunk_features,
            'limb_features': limb_features,
            'trunk_features_combined': trunk_features_combined,
            'limb_features_combined': limb_features_combined,
            'trunk_branch_weights': torch.stack(all_trunk_weights, dim=1),
            'limb_branch_weights': torch.stack(all_limb_weights, dim=1),
        }
