import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# 分支感知的权重计算模块
# ------------------------
class BranchAwareSEModule(nn.Module):
    def __init__(self, base_channels, num_scales, reduction=4, module_type='trunk',stage='early'):
        super().__init__()
        self.base_channels = base_channels
        self.num_scales = num_scales
        self.module_type = module_type

        # 阶段特定的Dropout
        if stage == 'early':
            self.dropout_rate = 0.1
        elif stage == 'mid':
            self.dropout_rate = 0.2
        else:  # late
            self.dropout_rate = 0.3

        self.dropout = nn.Dropout(self.dropout_rate)

        # 为躯干和肢体模块设置不同的特性
        if module_type == 'trunk':
            # 躯干模块：偏好低频特征，使用更大的卷积核获取全局信息
            self.branch_se_modules = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(base_channels, base_channels // reduction,
                              kernel_size=5 if i > num_scales // 2 else 3,  # 较大卷积核用于低频分支
                              padding='same'),
                    nn.ReLU(inplace=True),
                    self.dropout,
                    nn.Conv1d(base_channels // reduction, base_channels, kernel_size=1),
                    nn.Sigmoid()
                ) for i in range(num_scales)
            ])

            # 低频分支初始权重略高 --
            if stage == 'early':
                # 早期阶段：更关注低频全局特征
                self.freq_bias = nn.Parameter(
                    torch.ones(num_scales) * 0.3 +
                    torch.tensor([i / (num_scales - 1) * 0.7 for i in range(num_scales)])
                )
            elif stage == 'mid':
                # 中期阶段：平衡处理
                self.freq_bias = nn.Parameter(torch.ones(num_scales) * 0.5)
            else:  # late stage
                # 后期阶段：更关注高频细节
                self.freq_bias = nn.Parameter(
                    torch.ones(num_scales) * 0.3 +
                    torch.tensor([(num_scales - i - 1) / (num_scales - 1) * 0.7 for i in range(num_scales)])
                )
        else:
            # 肢体模块：偏好高频特征，使用较小的卷积核关注局部细节
            self.branch_se_modules = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(base_channels, base_channels // reduction,
                              kernel_size=3 if i < num_scales // 2 else 1,  # 小卷积核用于高频分支
                              padding='same'),
                    nn.GELU(),  # 肢体用GELU激活，更好地处理小信号
                    self.dropout,
                    nn.Conv1d(base_channels // reduction, base_channels, kernel_size=1),
                    nn.Sigmoid()
                ) for i in range(num_scales)
            ])
            # 高频分支初始权重略高
            if stage == 'early':
                # Early stage: Start by focusing on high-frequency local details (lower branch indices)
                self.freq_bias = nn.Parameter(
                    torch.ones(num_scales) * 0.4 +
                    torch.tensor([(num_scales - 1 - i) / (num_scales - 1) * 0.6 for i in range(num_scales)]) if num_scales > 1 else torch.ones(num_scales)
                )
            elif stage == 'mid':
                # Middle stage: Balanced, but still leaning towards high-frequency
                self.freq_bias = nn.Parameter(torch.ones(num_scales) * 0.5)
            else:  # late stage
                # Late stage: Strongest focus on high-frequency details for precision
                self.freq_bias = nn.Parameter(
                    torch.ones(num_scales) * 0.3 +
                    torch.tensor([(num_scales - 1 - i) / (num_scales - 1) * 0.7 for i in range(num_scales)]) if num_scales > 1 else torch.ones(num_scales)
                )

        # 分支间重要性权重，考虑频率偏好
        self.branch_importance = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(num_scales * base_channels, num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, branch_features):
        # 为每个分支单独应用SE
        weighted_branches = []
        for i, (branch, se_module) in enumerate(zip(branch_features, self.branch_se_modules)):
            # 应用SE权重
            channel_weights = se_module(branch)
            weighted_branch = branch * channel_weights
            weighted_branches.append(weighted_branch)

        # 拼接所有分支特征
        concat_features = torch.cat(weighted_branches, dim=1)

        # 计算分支间重要性，结合频率偏好偏置
        raw_weights = self.branch_importance(concat_features)
        branch_weights = raw_weights * self.freq_bias.view(1, -1, 1)   #拓展-便于广播
        branch_weights = F.softmax(branch_weights, dim=1)  # 重新归一化

        # 将分支重要性应用到对应特征组
        batch_size, _, seq_len = concat_features.shape
        reshaped_weights = branch_weights.view(batch_size, self.num_scales, 1, 1)
        reshaped_features = concat_features.view(batch_size, self.num_scales, self.base_channels, seq_len)

        # 加权求和
        weighted_features = (reshaped_features * reshaped_weights).view(batch_size, -1, seq_len)

        return weighted_features, branch_weights.squeeze(-1)


# -----------------------------------------------------------------------------
# 修改后的 TrunkSEModule（无全局池化，使用1D卷积）
# -----------------------------------------------------------------------------

class TrunkSEModule(nn.Module):

    # 640个通道上权重都不一样
    def __init__(self, total_channels, reduction=16):
        super().__init__()
        self.total_channels = total_channels
        self.se_net = nn.Sequential(
            nn.Conv1d(total_channels, total_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(total_channels // reduction, total_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.trunk_specific_conv = nn.Conv1d(total_channels, total_channels, kernel_size=5, padding=2)

    def forward(self, multi_freq_features):
        trunk_processed = self.trunk_specific_conv(multi_freq_features)
        frequency_weights = self.se_net(trunk_processed)
        weighted_features = trunk_processed * frequency_weights
        return weighted_features, frequency_weights


# -----------------------------------------------------------------------------
# 修改后的 LimbSEModule（无全局池化，使用1D卷积）
# -----------------------------------------------------------------------------

class LimbSEModule(nn.Module):
    def __init__(self, total_channels, reduction=16):
        super().__init__()
        self.total_channels = total_channels
        self.local_feature_extractor = nn.Conv1d(total_channels, total_channels, kernel_size=3, padding=1, groups=total_channels)
        self.se_net = nn.Sequential(
            nn.Conv1d(total_channels, total_channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(total_channels // reduction, total_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, multi_freq_features):
        local_features = self.local_feature_extractor(multi_freq_features)
        frequency_weights = self.se_net(local_features)
        weighted_features = local_features * frequency_weights
        return weighted_features, frequency_weights


# -----------------------------------------------------------------------------
# 频率权重分析器（增加均值处理）
# -----------------------------------------------------------------------------

class FrequencyWeightAnalyzer:
    @staticmethod
    def analyze_frequency_weights(frequency_weights, base_channels=64):
        frequency_weights = frequency_weights.mean(dim=-1)  # [B, C]
        batch_size, total_channels = frequency_weights.shape
        num_freq_bands = total_channels // base_channels
        reshaped = frequency_weights.view(batch_size, num_freq_bands, base_channels)
        mean_weights = reshaped.mean(dim=2)

        freq_labels = ['高频(d=1)', '中高频(d=2)', '中频(d=4)', '中低频(d=8)', '低频(d=16)']
        analysis = {}
        for i in range(num_freq_bands):
            analysis[freq_labels[i]] = {
                'mean_weight': mean_weights[:, i].mean().item(),
                'std_weight': mean_weights[:, i].std().item(),
                'importance_rank': None
            }
        sorted_indices = sorted(range(len(freq_labels)), key=lambda i: analysis[freq_labels[i]]['mean_weight'], reverse=True)
        for rank, idx in enumerate(sorted_indices):
            analysis[freq_labels[idx]]['importance_rank'] = rank + 1
        return analysis, mean_weights

# --------------------------
# NodeBranch
# --------------------------
class NodeBranch(nn.Module):
    def __init__(self, input_channels, base_channels, num_scales, dilation_rates, module_type='trunk'):
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            branch = nn.Sequential(
                nn.Conv1d(
                    input_channels, base_channels // 2,
                    kernel_size=3, dilation=dilation, padding=dilation
                ),
                nn.BatchNorm1d(base_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    base_channels // 2, base_channels,
                    kernel_size=3, dilation=dilation, padding=dilation
                ),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        # 为该节点构建分支感知SE模块
        self.se_module = BranchAwareSEModule(base_channels, num_scales, module_type=module_type)

    def forward(self, x):
        branch_features = [branch(x) for branch in self.branches]
        node_features, branch_weights = self.se_module(branch_features)
        return node_features, branch_weights


# --------------------------------
# 节点感知型(MSFKE)
# --------------------------------
class NodeAwareMSFKE(nn.Module):
    def __init__(self, input_channels=9, num_nodes=6, base_channels=128, num_scales=5,
                 node_feature_dim=18,stage='early'):
        super().__init__()
        self.input_channels = input_channels  # 每个节点的输入特征
        self.num_nodes = num_nodes  # 节点数量(6个IMU)
        self.num_scales = num_scales
        self.node_feature_dim = node_feature_dim  # 每个节点的输出特征
        self.stage = stage

        if stage == 'early':
            # 早期：更关注全局特征，使用更大的膨胀率
            self.dilation_rates = [2, 4, 8, 16, 32]
            self.base_channels = base_channels
        elif stage == 'mid':
            # 中期：平衡全局和局部
            self.dilation_rates = [1, 2, 4, 8, 16]
            self.base_channels = int(base_channels * 1.2)  # 增加容量
        else:  # late
            # 后期：更关注细节，使用更小的膨胀率
            self.dilation_rates = [1, 1, 2, 4, 8]  # 更多局部特征
            self.base_channels = int(base_channels * 1.5)

        # 节点编码 - 每个节点有唯一embedding
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, 16))   #16是嵌入量大小，用多少维度来表示节点

        # 为每个节点构建多尺度分支
        # self.node_branches = nn.ModuleList([
        #     NodeBranch(self.input_channels, self.base_channels, self.num_scales,
        #                self.dilation_rates, module_type='trunk' if i in [4] else 'limb' ) for i in range(num_nodes)
        # ])
        self.node_feature_extractors = nn.ModuleList([
            self._build_multi_scale_extractor() for _ in range(num_nodes)
        ])

        self.trunk_se_modules = nn.ModuleList([
            BranchAwareSEModule(self.base_channels, num_scales, module_type='trunk',stage=stage)
            for _ in range(num_nodes)
        ])

        self.limb_se_modules = nn.ModuleList([
            BranchAwareSEModule(self.base_channels, num_scales, module_type='limb',stage=stage)
            for _ in range(num_nodes)
        ])

        # # 节点间交互层 - 可选
        # self.node_interaction = nn.MultiheadAttention(
        #     embed_dim=self.base_channels * num_scales,
        #     num_heads=8,
        #     batch_first=True
        # )

        # 输出映射层
        self.node_projectors = nn.ModuleList([
            nn.Conv1d(self.base_channels * num_scales+16, node_feature_dim, kernel_size=1)    # 16是节点嵌入维度，18是输出特征维度
            for _ in range(num_nodes)
        ])

    def _build_multi_scale_extractor(self):
        branches = nn.ModuleList()
        for dilation in self.dilation_rates:
            branch = nn.Sequential(
                nn.Conv1d(
                    self.input_channels, self.base_channels // 2,
                    kernel_size=3, dilation=dilation, padding=dilation
                ),
                nn.BatchNorm1d(self.base_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    self.base_channels // 2, self.base_channels,
                    kernel_size=3, dilation=dilation, padding=dilation
                ),
                nn.BatchNorm1d(self.base_channels),
                nn.ReLU(inplace=True)
            )
            branches.append(branch)
        return branches


    def forward(self, imu_data):
        """
        输入: [batch_size, seq_len, 54] (54=6节点×9特征)
        输出: [batch_size, seq_len, 108] (108=6节点×18特征), 保证节点顺序
        """
        batch_size, seq_len, _ = imu_data.shape

        # 重组输入为每个节点单独处理
        node_inputs = imu_data.view(batch_size, seq_len, self.num_nodes, self.input_channels)

        # 节点级处理
        trunk_outputs = []
        limb_outputs = []
        trunk_weights_list = []
        limb_weights_list = []

        for i in range(self.num_nodes):
            # 提取当前节点的输入
            x_node = node_inputs[:, :, i, :].transpose(1, 2)  # [B, 9, T]

            # 直接使用NodeBranch实例(里边是节点不同)
            branch_features = [branch(x_node) for branch in self.node_feature_extractors[i]]

            # 2. 同时经过trunk和limb SE模块
            trunk_features, trunk_weights = self.trunk_se_modules[i](branch_features)
            limb_features, limb_weights = self.limb_se_modules[i](branch_features)

            # 节点编码融合 - 将节点位置信息注入特征
            node_embed = self.node_embedding[i].unsqueeze(0).unsqueeze(-1)  # [1, 16, 1]添加维度
            node_embed = node_embed.expand(batch_size, -1, seq_len)  # [B, 16, T]
            trunk_features = torch.cat([trunk_features, node_embed], dim=1)  # 加入位置编码
            limb_features = torch.cat([limb_features, node_embed], dim=1)

            # 投影到输出维度
            trunk_output = self.node_projectors[i](trunk_features)  # [B, 18, T]
            limb_output = self.node_projectors[i](limb_features)


            trunk_outputs.append(trunk_output)
            limb_outputs.append(limb_output)

            trunk_weights_list.append(trunk_weights)
            limb_weights_list.append(limb_weights)

        # 堆叠所有节点结果，保持节点顺序
        # 堆叠处理
        trunk_features = torch.stack(trunk_outputs, dim=2).permute(0, 3, 2, 1)  # [B, T, 6, 18]
        limb_features = torch.stack(limb_outputs, dim=2).permute(0, 3, 2, 1)  # [B, T, 6, 18]


        # 输出形状调整为模型期望的格式
        trunk_features_combined = trunk_features.reshape(batch_size, seq_len, -1)  # [B, T, 108]
        limb_features_combined = limb_features.reshape(batch_size, seq_len, -1)  # [B, T, 108]

        return {
            'trunk_features': trunk_features,
            'limb_features': limb_features,  # 在节点感知模型中，可以为肢体设计单独处理
            'trunk_features_combined': trunk_features_combined,  # [B, T, 6, 18] - 保留节点维度的特征
            'limb_features_combined': limb_features_combined,
            'trunk_branch_weights': torch.stack(trunk_weights_list, dim=1),  # [B, 6, 5]
            'limb_branch_weights': torch.stack(limb_weights_list, dim=1),
            'dilation_rates': self.dilation_rates
        }

