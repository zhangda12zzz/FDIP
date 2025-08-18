import torch
import os
import glob
import pandas as pd
from config import paths
import numpy as np

# 1. 收集 .pt 文件路径
datasets = {
    'DIP-IMU': r"D:\Dataset\DIPIMUandOthers\DIP_6\Detail",
    'TotalCapture': r"D:\Dataset\TotalCapture_Real_60FPS\KaPt\split_actions",
    'Hva': r"D:\Dataset\AMASS\HumanEva\pt",
    'DanceDB': r"D:\Dataset\AMASS\DanceDBa\pt",
    "SingleOne": r"D:\Dataset\SingleOne\Pt"
}

records = []
for ds_name, ds_dir in datasets.items():
    pt_files = glob.glob(os.path.join(ds_dir, '*.pt'))
    for file_path in pt_files:
        data = torch.load(file_path)
        # 判断形状
        if isinstance(data, torch.Tensor):
            shapes = [tuple(data.shape)]
        elif isinstance(data, (list, tuple)):
            shapes = [tuple(x.shape) for x in data]
        elif isinstance(data, dict):
            shapes = {k: tuple(v.shape) for k, v in data.items()}
        else:
            shapes = ['Unknown Type']
        records.append({
            'Dataset': ds_name,
            'Filename': os.path.basename(file_path),
            'Shapes': shapes
        })

# 以 DataFrame 形式展示文件与形状
df = pd.DataFrame(records)
import ace_tools as tools;tools.display_dataframe_to_user(name="PT 文件形状一览", dataframe=df)

# # 2. 对每个文件进行标准化并保存
# for ds_name, ds_dir in datasets.items():
#     pt_files = glob.glob(os.path.join(ds_dir, '*.pt'))
#     for file_path in pt_files:
#         data = torch.load(file_path)
#
#         # 将列表/tuple/dict 转为统一处理结构
#         if isinstance(data, torch.Tensor):
#             tensors = [data]
#         elif isinstance(data, (list, tuple)):
#             tensors = list(data)
#         elif isinstance(data, dict):
#             tensors = list(data.values())
#         else:
#             continue
#
#         # 拼接所有数据以计算全局均值和标准差
#         all_data = torch.cat([t.reshape(-1) for t in tensors], dim=0).float()
#         mean = all_data.mean().item()
#         std = all_data.std().item()
#
#
#         # 归一化处理
#         def normalize_tensor(t):
#             return (t.float() - mean) / (std + 1e-8)
#
#
#         if isinstance(data, torch.Tensor):
#             norm_data = normalize_tensor(data)
#         elif isinstance(data, (list, tuple)):
#             norm_data = [normalize_tensor(t) for t in data]
#         elif isinstance(data, dict):
#             norm_data = {k: normalize_tensor(v) for k, v in data.items()}
#
#         # 保存归一化文件
#         norm_path = file_path.replace('.pt', '_norm.pt')
#         torch.save(norm_data, norm_path)
#
# print("归一化处理完成，文件以 *_norm.pt 命名保存。")
