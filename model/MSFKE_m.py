import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# 压缩点1: 低秩分解卷积层
# ------------------------
class LowRankConv1d(nn.Module):
    """使用低秩分解减少卷积参数量"""

    def __init__(self, in_channels, out_channels, kernel_size, rank_ratio=0.4, **kwargs):
        super().__init__()
        rank = max(1, int(min(in_channels, out_channels) * rank_ratio))

        self.conv1 = nn.Conv1d(in_channels, rank, 1)  # 降维
        self.conv2 = nn.Conv1d(rank, out_channels, kernel_size, **kwargs)  # 升维

    def forward(self, x):
        return self.conv2(self.conv1(x))


# ------------------------
# 压缩点2: 压缩的分支感知SE模块
# ------------------------
class CompressedBranchAwareSEModule(nn.Module):
    def __init__(self, base_channels, num_scales, reduction=8, module_type='trunk',
                 stage='early'):  # 压缩点2a: 增加reduction
        super().__init__()
        self.base_channels = base_channels
        self.num_scales = num_scales
        self.module_type = module_type

        # 压缩点2b: 统一dropout率，减少条件分支
        self.dropout_rate = 0.15  # 统一dropout率
        self.dropout = nn.Dropout(self.dropout_rate)

        # 压缩点2c: 简化SE模块结构，使用低秩分解
        kernel_size = 5 if module_type == 'trunk' else 3
        activation = nn.ReLU(inplace=True) if module_type == 'trunk' else nn.GELU()

        self.branch_se_modules = nn.ModuleList([
            nn.Sequential(
                LowRankConv1d(base_channels, base_channels // reduction,
                              kernel_size=kernel_size,
                              rank_ratio=0.3, padding='same'),
                activation,
                self.dropout,
                nn.Conv1d(base_channels // reduction, base_channels, kernel_size=1),
                nn.Sigmoid()
            ) for i in range(num_scales)
        ])

        # 压缩点2d: 简化频率偏置初始化
        if module_type == 'trunk':
            if stage == 'early':
                bias_pattern = torch.linspace(0.3, 1.0, num_scales)
            elif stage == 'mid':
                bias_pattern = torch.ones(num_scales) * 0.5
            else:  # late
                bias_pattern = torch.linspace(1.0, 0.3, num_scales)
        else:  # limb
            if stage == 'early':
                bias_pattern = torch.linspace(1.0, 0.4, num_scales) if num_scales > 1 else torch.ones(num_scales)
            elif stage == 'mid':
                bias_pattern = torch.ones(num_scales) * 0.5
            else:  # late
                bias_pattern = torch.linspace(1.0, 0.3, num_scales) if num_scales > 1 else torch.ones(num_scales)

        self.freq_bias = nn.Parameter(bias_pattern)

        # 压缩点2e: 使用分组卷积减少参数
        self.branch_importance = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(num_scales * base_channels, num_scales, kernel_size=1,
                      groups=min(4, num_scales)),
            nn.Softmax(dim=1)
        )

    def forward(self, branch_features):
        weighted_branches = []
        for i, (branch, se_module) in enumerate(zip(branch_features, self.branch_se_modules)):
            channel_weights = se_module(branch)
            weighted_branch = branch * channel_weights
            weighted_branches.append(weighted_branch)

        concat_features = torch.cat(weighted_branches, dim=1)
        raw_weights = self.branch_importance(concat_features)
        branch_weights = raw_weights * self.freq_bias.view(1, -1, 1)
        branch_weights = F.softmax(branch_weights, dim=1)

        batch_size, _, seq_len = concat_features.shape
        reshaped_weights = branch_weights.view(batch_size, self.num_scales, 1, 1)
        reshaped_features = concat_features.view(batch_size, self.num_scales, self.base_channels, seq_len)

        weighted_features = (reshaped_features * reshaped_weights).view(batch_size, -1, seq_len)
        return weighted_features, branch_weights.squeeze(-1)


# ------------------------
# 压缩点3: 共享权重的SE模块
# ------------------------
class SharedSEModule(nn.Module):
    """trunk和limb模块共享部分权重"""

    def __init__(self, total_channels, reduction=16, module_type='trunk'):
        super().__init__()
        self.total_channels = total_channels

        # 压缩点3a: 共享的基础特征提取器，使用分组卷积
        if module_type == 'trunk':
            self.feature_processor = nn.Conv1d(total_channels, total_channels,
                                               kernel_size=5, padding=2,
                                               groups=max(1, total_channels // 8))  # 分组卷积
        else:
            self.feature_processor = nn.Conv1d(total_channels, total_channels,
                                               kernel_size=3, padding=1,
                                               groups=max(1, total_channels // 4))  # 分组卷积

        # 压缩点3b: 使用低秩分解的SE网络
        self.se_net = nn.Sequential(
            LowRankConv1d(total_channels, total_channels // reduction,
                          kernel_size=1, rank_ratio=0.3),
            nn.GELU() if module_type == 'limb' else nn.ReLU(inplace=True),
            LowRankConv1d(total_channels // reduction, total_channels,
                          kernel_size=1, rank_ratio=0.3),
            nn.Sigmoid()
        )

    def forward(self, multi_freq_features):
        processed_features = self.feature_processor(multi_freq_features)
        frequency_weights = self.se_net(processed_features)
        weighted_features = processed_features * frequency_weights
        return weighted_features, frequency_weights


# ------------------------
# 压缩点4: 权重共享的节点分支
# ------------------------
class SharedNodeBranch(nn.Module):
    def __init__(self, input_channels, base_channels, num_scales, dilation_rates, num_node_groups=2):
        super().__init__()

        # 压缩点4a: 创建共享的分支组
        self.shared_branches = nn.ModuleList()
        for group_id in range(num_node_groups):
            group_branches = nn.ModuleList()
            for dilation in dilation_rates:
                branch = nn.Sequential(
                    LowRankConv1d(input_channels, base_channels // 2,
                                  kernel_size=3, dilation=dilation, padding=dilation,
                                  rank_ratio=0.4),
                    nn.BatchNorm1d(base_channels // 2),
                    nn.ReLU(inplace=True),
                    LowRankConv1d(base_channels // 2, base_channels,
                                  kernel_size=3, dilation=dilation, padding=dilation,
                                  rank_ratio=0.4),
                    nn.BatchNorm1d(base_channels),
                    nn.ReLU(inplace=True)
                )
                group_branches.append(branch)
            self.shared_branches.append(group_branches)

        # 压缩点4b: 节点到组的映射 (4是躯干，其他是肢体)
        self.node_to_group_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0}

    def forward(self, x, node_id):
        group_id = self.node_to_group_map[node_id]
        branch_features = [branch(x) for branch in self.shared_branches[group_id]]
        return branch_features


# ------------------------
# 压缩点5: 频率权重分析器（简化版）
# ------------------------
class FrequencyWeightAnalyzer:
    @staticmethod
    def analyze_frequency_weights(frequency_weights, base_channels=24):  # 压缩点5a: 减少base_channels
        frequency_weights = frequency_weights.mean(dim=-1)
        batch_size, total_channels = frequency_weights.shape
        num_freq_bands = total_channels // base_channels
        reshaped = frequency_weights.view(batch_size, num_freq_bands, base_channels)
        mean_weights = reshaped.mean(dim=2)

        freq_labels = ['高频(d=1)', '中高频(d=3)', '中频(d=7)', '中低频(d=15)', '低频(d=31)']
        analysis = {}
        for i in range(min(num_freq_bands, len(freq_labels))):
            analysis[freq_labels[i]] = {
                'mean_weight': mean_weights[:, i].mean().item(),
                'std_weight': mean_weights[:, i].std().item(),
                'importance_rank': None
            }

        sorted_indices = sorted(range(len(analysis)),
                                key=lambda i: list(analysis.values())[i]['mean_weight'],
                                reverse=True)
        for rank, idx in enumerate(sorted_indices):
            list(analysis.values())[idx]['importance_rank'] = rank + 1
        return analysis, mean_weights


# --------------------------------
# # 压缩点6: 主要的压缩模型
# # --------------------------------
class CompressedNodeAwareMSFKE(nn.Module):
    def __init__(self, input_channels=9, num_nodes=6, base_channels=24, num_scales=5,  # 压缩点6a: 减少base_channels
                 node_feature_dim=18, stage='early'):
        super().__init__()
        self.input_channels = input_channels
        self.num_nodes = num_nodes
        self.num_scales = num_scales
        self.node_feature_dim = node_feature_dim
        self.stage = stage

        if stage == 'late':
            self.dimension_adapter = nn.Sequential(
                nn.Linear(288, 72),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.adapted_input_channels = 3
        else:
            self.dimension_adapter = None
            self.adapted_input_channels = input_channels

        self.dilation_rates = [1, 3, 7, 15, 31]
        if stage == 'early':
            self.base_channels = base_channels
        elif stage == 'mid':
            self.base_channels = int(base_channels * 1.2)
        else:  # late
            self.base_channels = int(base_channels * 1.2)

        # 位置编码
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, 8))

        # 压缩点6f: 使用共享的节点分支
        self.shared_node_branch = SharedNodeBranch(
            self.adapted_input_channels, self.base_channels,
            self.num_scales, self.dilation_rates, num_node_groups=2
        )

        # 压缩点6g: 只使用两个共享的SE模块
        self.trunk_se_module = CompressedBranchAwareSEModule(
            self.base_channels, num_scales, module_type='trunk', stage=stage
        )
        self.limb_se_module = CompressedBranchAwareSEModule(
            self.base_channels, num_scales, module_type='limb', stage=stage
        )

        # 压缩点6h: 节点特定的轻量级适配器
        self.node_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.base_channels * num_scales,
                          self.base_channels * num_scales,
                          kernel_size=1, groups=min(4, self.base_channels * num_scales)),  # 分组卷积
                nn.BatchNorm1d(self.base_channels * num_scales),
                nn.ReLU(inplace=True)
            ) for _ in range(num_nodes)
        ])

        # 压缩点6i: 共享输出投影器
        self.output_projector = nn.Conv1d(self.base_channels * num_scales + 8,
                                          node_feature_dim, kernel_size=1)  # 从16降到8

    def forward(self, imu_data):
        batch_size, seq_len, _ = imu_data.shape

        if self.stage == 'late' and self.dimension_adapter is not None:
            imu_data = self.dimension_adapter(imu_data)
            adapted_input_channels = self.adapted_input_channels
        else:
            adapted_input_channels = self.input_channels

        node_inputs = imu_data.view(batch_size, seq_len, self.num_nodes, adapted_input_channels)

        trunk_outputs = []
        limb_outputs = []
        trunk_weights_list = []
        limb_weights_list = []

        for i in range(self.num_nodes):
            x_node = node_inputs[:, :, i, :].transpose(1, 2)  # [B, adapted_input_channels, T]

            # 使用共享的分支提取器
            branch_features = self.shared_node_branch(x_node, i)

            # 根据节点类型选择SE模块
            if i == 4:  # 躯干节点
                se_features, se_weights = self.trunk_se_module(branch_features)
            else:  # 肢体节点
                se_features, se_weights = self.limb_se_module(branch_features)

            # 节点特定适配
            adapted_features = self.node_adapters[i](se_features)

            # 添加节点编码
            node_embed = self.node_embedding[i].unsqueeze(0).unsqueeze(-1)
            node_embed = node_embed.expand(batch_size, -1, seq_len)
            features_with_embed = torch.cat([adapted_features, node_embed], dim=1)

            # 使用共享的输出投影器
            output = self.output_projector(features_with_embed)

            trunk_outputs.append(output)
            limb_outputs.append(output)  # 简化：trunk和limb使用相同输出
            trunk_weights_list.append(se_weights)
            limb_weights_list.append(se_weights)

        # 堆叠结果
        trunk_features = torch.stack(trunk_outputs, dim=2).permute(0, 3, 2, 1)  # [B, T, 6, 18]
        limb_features = torch.stack(limb_outputs, dim=2).permute(0, 3, 2, 1)  # [B, T, 6, 18]

        trunk_features_combined = trunk_features.reshape(batch_size, seq_len, -1)  # [B, T, 108]
        limb_features_combined = limb_features.reshape(batch_size, seq_len, -1)  # [B, T, 108]

        return {
            'trunk_features': trunk_features,
            'limb_features': limb_features,
            'trunk_features_combined': trunk_features_combined,
            'limb_features_combined': limb_features_combined,
            'trunk_branch_weights': torch.stack(trunk_weights_list, dim=1),  # [B, 6, 5]
            'limb_branch_weights': torch.stack(limb_weights_list, dim=1),
            'dilation_rates': self.dilation_rates
        }


# ------------------------
# 压缩点7: 知识蒸馏训练框架
# ------------------------
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.6):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_outputs, teacher_outputs, true_labels, criterion):
        # 硬标签损失
        student_predictions = student_outputs.get('predictions', student_outputs['trunk_features_combined'])
        hard_loss = criterion(student_predictions, true_labels)

        # 软标签损失（特征蒸馏）
        soft_loss = 0
        count = 0
        for key in ['trunk_features_combined', 'limb_features_combined']:
            if key in student_outputs and key in teacher_outputs:
                soft_loss += F.mse_loss(student_outputs[key], teacher_outputs[key].detach())
                count += 1

        if count > 0:
            soft_loss = soft_loss / count

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# ------------------------
# 压缩点8: 渐进式剪枝器
# ------------------------
class ProgressivePruner:
    def __init__(self, model, target_sparsity=0.3, total_steps=1000):
        self.model = model
        self.target_sparsity = target_sparsity
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        if self.current_step < self.total_steps:
            current_sparsity = self.target_sparsity * (self.current_step / self.total_steps)
            self._prune_weights(current_sparsity)
            self.current_step += 1

    def _prune_weights(self, sparsity):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d) and hasattr(module, 'weight'):
                weight = module.weight.data
                # 计算权重重要性
                importance = weight.abs().view(weight.size(0), -1).sum(dim=1)
                num_to_prune = int(len(importance) * sparsity)

                if num_to_prune > 0:
                    _, indices = torch.topk(importance, num_to_prune, largest=False)
                    weight[indices] = 0


# ------------------------
# 保留原始模块以便比较
# ------------------------
class TrunkSEModule(nn.Module):
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


class LimbSEModule(nn.Module):
    def __init__(self, total_channels, reduction=16):
        super().__init__()
        self.total_channels = total_channels
        self.local_feature_extractor = nn.Conv1d(total_channels, total_channels,
                                                 kernel_size=3, padding=1, groups=total_channels)
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


# 使用示例和训练函数
def create_compressed_model(stage='early'):
    """创建压缩后的模型"""
    return CompressedNodeAwareMSFKE(
        input_channels=9,
        num_nodes=6,
        base_channels=24,  # 压缩点: 从32降到24
        num_scales=5,
        node_feature_dim=18,
        stage=stage
    )


def train_with_distillation(teacher_model, student_model, train_loader, optimizer, device):
    """知识蒸馏训练函数"""
    teacher_model.eval()
    student_model.train()

    distill_loss_fn = KnowledgeDistillationLoss()
    pruner = ProgressivePruner(student_model, target_sparsity=0.2)

    for epoch, batch in enumerate(train_loader):
        batch_input = batch['input'].to(device)
        batch_labels = batch['labels'].to(device)

        # 教师模型推理
        with torch.no_grad():
            teacher_outputs = teacher_model(batch_input)

        # 学生模型推理
        student_outputs = student_model(batch_input)

        # 计算蒸馏损失
        loss = distill_loss_fn(student_outputs, teacher_outputs, batch_labels, F.mse_loss)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 渐进式剪枝
        if epoch % 10 == 0:
            pruner.step()


def compare_model_sizes():
    """比较原始模型和压缩模型的参数量"""
    # 这里需要原始模型进行比较
    compressed_model = create_compressed_model()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    compressed_params = count_parameters(compressed_model)
    print(f"压缩模型参数量: {compressed_params:,}")

    return compressed_model


if __name__ == "__main__":
    # 创建并测试压缩模型
    model = compare_model_sizes()

    # 测试前向传播
    batch_size, seq_len, input_dim = 8, 100, 54
    test_input = torch.randn(batch_size, seq_len, input_dim)

    with torch.no_grad():
        outputs = model(test_input)
        print(f"输出形状: {outputs['trunk_features_combined'].shape}")
        print(f"分支权重形状: {outputs['trunk_branch_weights'].shape}")
