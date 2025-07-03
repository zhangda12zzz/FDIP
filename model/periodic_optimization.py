import torch
from torch import nn


def forward_with_periodic_optimization(self, imu_data):
    """
    集成周期性优化的前向传播
    """
    # 1. 多尺度频域编码
    multi_scale_features = self.msfke(imu_data)

    # 2. 提取SE权重向量
    trunk_weights = self.trunk_se_module.get_weights()
    limb_weights = self.limb_se_module.get_weights()

    # 3. 周期性检测与缓存
    should_use_cache, cached_data = self.periodic_cache.detect_and_cache(
        torch.cat([trunk_weights, limb_weights]),
        multi_scale_features
    )

    # 4. 选择计算路径
    if should_use_cache and isinstance(cached_data, dict):
        # 使用缓存的特征
        trunk_features = cached_data['features']['trunk_features']
        limb_features = cached_data['features']['limb_features']
    else:
        # 正常计算
        trunk_features = self.st_gcn(multi_scale_features['trunk'])
        limb_features = self.bi_gru(multi_scale_features['limb'])

    # 5. 交叉注意力融合
    fused_features = self.cross_attention_fusion(trunk_features, limb_features)

    # 6. 最终回归
    pose_output = self.regression_head(fused_features)

    return pose_output


class PeriodDetectionGRU(nn.Module):
    def __init__(self, weight_dim=256, hidden_dim=64, max_period=100):
        super().__init__()
        self.weight_encoder = nn.Linear(weight_dim, 32)

        # 单向GRU用于实时周期检测
        self.period_gru = nn.GRU(32, hidden_dim, batch_first=True)

        # 周期检测分类器
        self.period_classifier = nn.Linear(hidden_dim, 2)  # 是否周期性
        self.period_length_predictor = nn.Linear(hidden_dim, max_period)

    def forward(self, weight_sequence):
        # weight_sequence: [batch, seq_len, weight_dim]
        encoded = torch.relu(self.weight_encoder(weight_sequence))

        gru_out, h_n = self.period_gru(encoded)

        # 使用最后时刻的隐状态
        is_periodic = torch.softmax(self.period_classifier(h_n[-1]), dim=-1)
        period_length = torch.softmax(self.period_length_predictor(h_n[-1]), dim=-1)

        return is_periodic, period_length


class PeriodicWeightCacheSystem:
    def __init__(self, similarity_threshold=0.85):
        self.is_periodic = False
        self.period_length = None
        self.first_period_weights = {}  # 存储第一个周期的权重
        self.period_features = {}  # 存储第一个周期的特征
        self.detection_window = 50  # 检测窗口大小
        self.similarity_threshold = similarity_threshold
        self.frame_count = 0

    def detect_and_cache(self, weight_vector, features):
        """
        检测周期性并缓存第一个周期的参数
        """
        self.frame_count += 1

        # 第一阶段：收集足够的数据进行周期检测
        if not self.is_periodic and self.frame_count >= self.detection_window:
            period_detected, period_len = self.detect_periodicity(weight_vector)

            if period_detected:
                self.is_periodic = True
                self.period_length = period_len
                print(f"检测到周期性运动，周期长度: {period_len}")

                # 立即开始缓存第一个周期
                self.start_first_period_caching()

        # 第二阶段：缓存第一个完整周期
        if self.is_periodic and self.period_length is not None:
            return self.cache_first_period(weight_vector, features)

        return False, None

    def detect_periodicity(self, weight_sequence):
        """
        基于权重向量序列检测周期性
        """
        # 使用GRU进行周期检测
        model_output = self.period_detection_model(weight_sequence)
        is_periodic_prob, period_length_dist = model_output

        # 判断是否为周期性
        if is_periodic_prob[1] > 0.8:  # 周期性概率大于0.8
            predicted_period = torch.argmax(period_length_dist) + 1
            return True, predicted_period.item()

        return False, None

    def cache_first_period(self, weight_vector, features):
        """
        缓存第一个周期的权重和特征
        """
        current_position = self.frame_count % self.period_length

        if len(self.first_period_weights) < self.period_length:
            # 还在收集第一个周期的数据
            self.first_period_weights[current_position] = weight_vector.clone()
            self.first_period_features[current_position] = features.clone()

            if len(self.first_period_weights) == self.period_length:
                print("第一个周期缓存完成")
                return True, "first_period_cached"

        else:
            # 第一个周期已完成，进行相似性检查
            cached_weight = self.first_period_weights[current_position]
            similarity = self.compute_similarity(weight_vector, cached_weight)

            if similarity > self.similarity_threshold:
                # 相似度足够高，使用缓存
                return True, self.first_period_features[current_position]
            else:
                # 相似度不够，更新缓存
                self.update_cache(current_position, weight_vector, features)
                return True, features

        return False, None

