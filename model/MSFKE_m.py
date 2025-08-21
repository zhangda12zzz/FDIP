import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # 增加中间层
        self.mid_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.activation2 = nn.ReLU(inplace=True)

        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.mid_conv(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.pointwise_conv(x)
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

        dropout_rates = {'early': 0.2, 'mid': 0.2, 'late': 0.3}
        self.dropout = nn.Dropout(dropout_rates.get(stage, 0.3))

        if module_type == 'trunk':
            self.branch_se_modules = nn.ModuleList([
                nn.Sequential(
                    DepthwiseSeparableConv1d(base_channels, base_channels // reduction,
                                             kernel_size=5 if i > num_scales // 2 else 3),
                    nn.ReLU(inplace=True),
                    self.dropout,
                    # 增加中间层
                    nn.Conv1d(base_channels // reduction, base_channels // reduction, kernel_size=3, padding=1),
                    nn.BatchNorm1d(base_channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(base_channels // reduction, base_channels // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(base_channels // reduction, base_channels, kernel_size=1),
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
                    # 增加中间层
                    nn.Conv1d(base_channels // reduction, base_channels // reduction, kernel_size=3, padding=1),
                    nn.BatchNorm1d(base_channels // reduction),
                    nn.GELU(),
                    nn.Conv1d(base_channels // reduction, base_channels // reduction, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(base_channels // reduction, base_channels, kernel_size=1),
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

        # 修正：确保branch_importance的输入维度正确计算
        concat_channels = num_scales * base_channels

        self.branch_importance = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(concat_channels, max(1, concat_channels // 2), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(max(1, concat_channels // 2), max(1, concat_channels // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, concat_channels // 4), num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, branch_features):
        # 添加维度检查
        print(f"[DEBUG] Input branch_features shapes: {[f.shape for f in branch_features]}")

        weighted_branches = []
        for i, (branch, se_module) in enumerate(zip(branch_features, self.branch_se_modules)):
            try:
                weighted = se_module(branch) * branch
                weighted_branches.append(weighted)
                print(f"[DEBUG] Branch {i} weighted shape: {weighted.shape}")
            except Exception as e:
                print(f"[ERROR] Branch {i} processing failed: {e}")
                print(f"[DEBUG] Branch {i} shape: {branch.shape}")
                raise e

        concat_features = torch.cat(weighted_branches, dim=1)
        print(f"[DEBUG] Concat features shape: {concat_features.shape}")

        raw_weights = self.branch_importance(concat_features)
        branch_weights = F.softmax(raw_weights * self.freq_bias.view(1, -1, 1), dim=1)

        batch_size, _, seq_len = concat_features.shape
        reshaped_features = concat_features.view(batch_size, self.num_scales, self.base_channels, seq_len)

        # Element-wise multiplication after broadcasting weights
        fused_features = (reshaped_features * branch_weights.unsqueeze(-2)).sum(dim=1)

        return fused_features, branch_weights.squeeze(-1)


class SharedMultiScaleExtractor(nn.Module):
    """
    共享的多尺度特征提取器。
    """

    def __init__(self, input_channels, base_channels, dilation_rates):
        super().__init__()
        print(f"[DEBUG] SharedMultiScaleExtractor - input_channels: {input_channels}, base_channels: {base_channels}")

        self.branches = nn.ModuleList()
        for i, dilation in enumerate(dilation_rates):
            branch = nn.Sequential(
                # 第一层卷积组
                nn.Conv1d(input_channels, base_channels // 4, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(base_channels // 4),
                nn.ReLU(inplace=True),

                # 第二层卷积组
                nn.Conv1d(base_channels // 4, base_channels // 2, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(base_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),

                # 第三层卷积组
                nn.Conv1d(base_channels // 2, base_channels // 2, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(base_channels // 2),
                nn.ReLU(inplace=True),

                # 第四层卷积组 - 确保输出是base_channels
                nn.Conv1d(base_channels // 2, base_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
            print(f"[DEBUG] Branch {i} created for dilation {dilation}")

    def forward(self, x):
        print(f"[DEBUG] SharedMultiScaleExtractor input shape: {x.shape}")
        results = []
        for i, branch in enumerate(self.branches):
            output = branch(x)
            results.append(output)
            print(f"[DEBUG] Branch {i} output shape: {output.shape}")
        return results


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

        print(f"[DEBUG] NodeAwareMSFKE init - num_nodes: {num_nodes}, input_channels: {input_channels}")
        print(f"[DEBUG] base_channels: {base_channels}, stage: {stage}")

        # --- 1. 根据 num_nodes 和 stage 配置模型 ---
        self.base_channels = self.base_channels_config if self.stage == 'early' else int(
            self.base_channels_config * 1.5)

        if self.num_nodes == 6:
            self.num_groups = 3
            self.adapted_input_channels = self.input_channels
            self.dimension_adapter = None
            self.node_to_group_map = {0: 0, 3: 1, 1: 2, 2: 2, 4: 2, 5: 2}
        elif self.num_nodes == 24:
            self.num_groups = 5
            self.adapted_input_channels = 3
            # 修正维度适配器计算
            expected_output_dim = self.num_nodes * self.adapted_input_channels
            print(f"[DEBUG] Dimension adapter output dim: {expected_output_dim}")

            self.dimension_adapter = nn.Sequential(
                nn.Linear(self.input_channels * self.num_nodes, 512),  # 修正输入维度
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, expected_output_dim),
                nn.LayerNorm(expected_output_dim)
            )
            groups_late = [[20, 21, 22, 23, 7, 8, 11, 10], [4, 5, 18, 19], [12, 15, 16, 17], [0],
                           [1, 2, 3, 6, 9, 13, 14]]
            self.node_to_group_map = {node_idx: group_idx for group_idx, nodes in enumerate(groups_late) for node_idx in
                                      nodes}
        else:
            raise ValueError(f"Unsupported number of nodes: {self.num_nodes}. Only 6 and 24 are supported.")

        self.dilation_rates = [1, 3, 7, 15, 31]
        self.fused_channels = self.base_channels

        print(f"[DEBUG] Final config - adapted_input_channels: {self.adapted_input_channels}")
        print(f"[DEBUG] base_channels: {self.base_channels}, fused_channels: {self.fused_channels}")

        # --- 2. 定义模型组件 ---
        self.node_embedding = nn.Parameter(torch.randn(24, self.node_embedding_dim))

        self.shared_feature_extractor = SharedMultiScaleExtractor(
            self.adapted_input_channels,
            self.base_channels,
            self.dilation_rates
        )

        # SE模块数量根据配置的 num_groups 动态创建
        self.trunk_se_modules = nn.ModuleList([
            BranchAwareSEModule(self.base_channels, self.num_scales, module_type='trunk', stage=stage)
            for _ in range(self.num_groups)
        ])
        self.limb_se_modules = nn.ModuleList([
            BranchAwareSEModule(self.base_channels, self.num_scales, module_type='limb', stage=stage)
            for _ in range(self.num_groups)
        ])

        # 修正：确保projection层的输入维度正确
        projection_input_dim = self.fused_channels + self.node_embedding_dim
        print(f"[DEBUG] Projection input dim: {projection_input_dim}")

        self.shared_projector = nn.Sequential(
            nn.Conv1d(projection_input_dim,
                      max(8, projection_input_dim // 2), kernel_size=1),
            nn.BatchNorm1d(max(8, projection_input_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(max(8, projection_input_dim // 2),
                      self.node_feature_dim, kernel_size=1)
        )

    def forward(self, imu_data):
        batch_size, seq_len, original_features = imu_data.shape
        print(f"[DEBUG] Input shape: {imu_data.shape}")

        # --- 1. 输入预处理和重塑 ---
        if self.dimension_adapter is not None:
            print(f"[DEBUG] Before dimension adapter: {imu_data.shape}")
            # 重塑为 (batch_size * seq_len, features)
            imu_reshaped = imu_data.reshape(batch_size * seq_len, -1)
            adapted_data = self.dimension_adapter(imu_reshaped)
            imu_data = adapted_data.reshape(batch_size, seq_len, -1)
            print(f"[DEBUG] After dimension adapter: {imu_data.shape}")

        node_inputs = imu_data.view(batch_size, seq_len, self.num_nodes, self.adapted_input_channels)
        parallel_inputs = node_inputs.permute(0, 2, 3, 1).reshape(
            batch_size * self.num_nodes, self.adapted_input_channels, seq_len
        )
        print(f"[DEBUG] Parallel inputs shape: {parallel_inputs.shape}")

        # --- 2. 并行多尺度特征提取 ---
        branch_features_list = self.shared_feature_extractor(parallel_inputs)
        print(f"[DEBUG] Branch features list shapes: {[f.shape for f in branch_features_list]}")

        # --- 3. 并行节点感知SE处理 ---
        all_trunk_features, all_limb_features = [], []
        all_trunk_weights, all_limb_weights = [], []

        organized_features = [
            feat.view(batch_size, self.num_nodes, self.base_channels, seq_len)
            for feat in branch_features_list
        ]
        print(f"[DEBUG] Organized features shapes: {[f.shape for f in organized_features]}")

        for node_idx in range(self.num_nodes):
            node_branch_features = [feat[:, node_idx] for feat in organized_features]
            group_idx = self.node_to_group_map[node_idx]

            print(f"[DEBUG] Node {node_idx}, Group {group_idx}")
            print(f"[DEBUG] Node branch features shapes: {[f.shape for f in node_branch_features]}")

            try:
                trunk_feat, trunk_w = self.trunk_se_modules[group_idx](node_branch_features)
                limb_feat, limb_w = self.limb_se_modules[group_idx](node_branch_features)

                all_trunk_features.append(trunk_feat)
                all_limb_features.append(limb_feat)
                all_trunk_weights.append(trunk_w)
                all_limb_weights.append(limb_w)
            except Exception as e:
                print(f"[ERROR] Node {node_idx} processing failed: {e}")
                raise e

        # --- 4. 并行融合节点编码和共享投射 ---
        trunk_stacked = torch.stack(all_trunk_features, dim=1)
        limb_stacked = torch.stack(all_limb_features, dim=1)
        print(f"[DEBUG] Trunk stacked shape: {trunk_stacked.shape}")
        print(f"[DEBUG] Limb stacked shape: {limb_stacked.shape}")

        # 节点嵌入
        node_embeds_sliced = self.node_embedding[:self.num_nodes]
        node_embeds = node_embeds_sliced.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, seq_len)
        print(f"[DEBUG] Node embeds shape: {node_embeds.shape}")

        trunk_with_embeds = torch.cat([trunk_stacked, node_embeds], dim=2)
        limb_with_embeds = torch.cat([limb_stacked, node_embeds], dim=2)
        print(f"[DEBUG] Trunk with embeds shape: {trunk_with_embeds.shape}")

        c_proj = self.fused_channels + self.node_embedding_dim
        parallel_trunk_to_proj = trunk_with_embeds.reshape(batch_size * self.num_nodes, c_proj, seq_len)
        parallel_limb_to_proj = limb_with_embeds.reshape(batch_size * self.num_nodes, c_proj, seq_len)

        print(f"[DEBUG] Parallel trunk to proj shape: {parallel_trunk_to_proj.shape}")
        print(f"[DEBUG] Expected projection input channels: {c_proj}")

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
