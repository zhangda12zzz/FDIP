import os
import glob
import torch
import pandas as pd
import re
from collections import defaultdict

datasets = {
    'DIP-IMU': r"D:\Dataset\DIPIMUandOthers\DIP_6\Detail",
    'TotalCapture': r"D:\Dataset\TotalCapture_Real_60FPS\KaPt\split_actions",
    'Hva': r"D:\Dataset\AMASS\HumanEva\pt",
    'DanceDB': r"D:\Dataset\AMASS\DanceDB\pt",
    "SingleOne": r"D:\Dataset\SingleOne\Pt"
}


def get_file_type(filename):
    """从文件名提取文件类型，例如从'pose_123.pt'提取'pose'"""
    # 从文件名中移除序号和扩展名
    base_name = os.path.splitext(filename)[0]  # 移除扩展名

    # 尝试通过下划线或其他分隔符提取文件类型
    # 例如: "pose_123" -> "pose", "tran_data_01" -> "tran"
    match = re.match(r'^([a-zA-Z]+)', base_name)
    if match:
        return match.group(1).lower()
    return "unknown"


def analyze_tensor(tensor, prefix=""):
    """分析单个张量并返回统计信息"""
    try:
        if not isinstance(tensor, torch.Tensor):
            return {
                'Type': f"{prefix}Not_Tensor",
                'DataType': type(tensor).__name__,
                'Shape': 'N/A',
                'Stats': 'Not applicable'
            }

        shape = tuple(tensor.shape)

        # 检查是否是数值型张量
        if tensor.dtype in [torch.float32, torch.float64, torch.float16,
                            torch.int32, torch.int64, torch.int16, torch.int8]:
            return {
                'Type': f"{prefix}Tensor",
                'DataType': str(tensor.dtype),
                'Shape': shape,
                'Mean': tensor.float().mean().item() if tensor.numel() > 0 else 'Empty',
                'Std': tensor.float().std().item() if tensor.numel() > 0 else 'Empty',
                'Min': tensor.float().min().item() if tensor.numel() > 0 else 'Empty',
                'Max': tensor.float().max().item() if tensor.numel() > 0 else 'Empty'
            }
        else:
            return {
                'Type': f"{prefix}Tensor",
                'DataType': str(tensor.dtype),
                'Shape': shape,
                'Stats': 'Non-numeric tensor'
            }
    except Exception as e:
        return {
            'Type': f"{prefix}Tensor",
            'DataType': str(getattr(tensor, 'dtype', 'Unknown')),
            'Shape': getattr(tensor, 'shape', 'Unknown'),
            'Stats': f'Error: {str(e)}'
        }


def analyze_nested_structure(data, prefix=""):
    """递归分析嵌套数据结构"""
    results = []

    if isinstance(data, torch.Tensor):
        results.append(analyze_tensor(data, prefix))

    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            if isinstance(item, torch.Tensor):
                results.append(analyze_tensor(item, f"{prefix}List/Tuple[{i}]_"))
            elif isinstance(item, (list, tuple, dict)):
                results.extend(analyze_nested_structure(item, f"{prefix}List/Tuple[{i}]_"))

    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                results.append(analyze_tensor(value, f"{prefix}Dict[{key}]_"))
            elif isinstance(value, (list, tuple, dict)):
                results.extend(analyze_nested_structure(value, f"{prefix}Dict[{key}]_"))

    else:
        # 其他类型的数据
        results.append({
            'Type': f"{prefix}Other",
            'DataType': type(data).__name__,
            'Shape': 'N/A',
            'Stats': 'Not applicable'
        })

    return results


# 创建输出目录
output_dir = "pt_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# 存储所有文件的分析结果
all_files_analysis = []

# 分析每个数据集中的每个PT文件
for ds_name, ds_dir in datasets.items():
    pt_files = glob.glob(os.path.join(ds_dir, '*.pt'))
    print(f"处理数据集 {ds_name}: 找到 {len(pt_files)} 个.pt文件")

    for file_path in pt_files:
        filename = os.path.basename(file_path)
        file_type = get_file_type(filename)
        print(f"  分析 {filename} (类型: {file_type})")

        try:
            # 加载PT文件
            data = torch.load(file_path)

            # 分析数据结构
            analysis_results = analyze_nested_structure(data)

            # 将数据集、文件名和文件类型添加到每个结果中
            for result in analysis_results:
                result['Dataset'] = ds_name
                result['Filename'] = filename
                result['FileType'] = file_type
                all_files_analysis.append(result)

        except Exception as e:
            # 记录无法加载或分析的文件
            all_files_analysis.append({
                'Dataset': ds_name,
                'Filename': filename,
                'FileType': file_type,
                'Type': 'Error',
                'DataType': 'N/A',
                'Shape': 'N/A',
                'Stats': f'Error loading file: {str(e)}'
            })

# 创建DataFrame并显示结果
analysis_df = pd.DataFrame(all_files_analysis)

# 重新排列列顺序，确保Dataset、Filename和FileType在前面
columns = ['Dataset', 'Filename', 'FileType', 'Type', 'DataType', 'Shape']
for col in analysis_df.columns:
    if col not in columns:
        columns.append(col)
analysis_df = analysis_df[columns]

# 显示和保存结果
print("\n===== 所有PT文件分析结果 =====")
print(analysis_df.head(20))  # 只显示前20行
print(f"\n[总计 {len(analysis_df)} 行数据]")
analysis_df.to_csv(os.path.join(output_dir, "all_pt_files_analysis.csv"), index=False, encoding="utf-8-sig")
print(f"完整结果已保存至: {os.path.join(output_dir, 'all_pt_files_analysis.csv')}")

# 按数据集和文件类型统计张量信息
file_type_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for _, row in analysis_df.iterrows():
    file_type_summary[row['Dataset']][row['FileType']][row['Type']] += 1

summary_rows = []
for ds, type_dict in file_type_summary.items():
    for file_type, tensor_types in type_dict.items():
        for tensor_type, count in tensor_types.items():
            summary_rows.append({
                'Dataset': ds,
                'FileType': file_type,
                'TensorType': tensor_type,
                'Count': count
            })

summary_df = pd.DataFrame(summary_rows)
print("\n===== 按文件类型的数据统计 =====")
print(summary_df.head(20))
print(f"\n[总计 {len(summary_df)} 行数据]")
summary_df.to_csv(os.path.join(output_dir, "file_type_summary.csv"), index=False, encoding="utf-8-sig")
print(f"完整结果已保存至: {os.path.join(output_dir, 'file_type_summary.csv')}")

# 为每种文件类型计算形状统计
shape_stats = defaultdict(lambda: defaultdict(list))
for _, row in analysis_df.iterrows():
    if row['Shape'] != 'N/A' and row['Shape'] != 'Unknown':
        # 将形状转换为字符串以便分组
        shape_str = str(row['Shape'])
        shape_stats[row['FileType']][shape_str].append(row)

shape_summary = []
for file_type, shapes in shape_stats.items():
    for shape_str, rows in shapes.items():
        shape_summary.append({
            'FileType': file_type,
            'Shape': shape_str,
            'Count': len(rows),
            'Datasets': ', '.join(sorted(set(row['Dataset'] for row in rows)))
        })

shape_df = pd.DataFrame(shape_summary)
shape_df = shape_df.sort_values(['FileType', 'Count'], ascending=[True, False])
print("\n===== 文件类型形状统计 =====")
print(shape_df.head(20))
print(f"\n[总计 {len(shape_df)} 行数据]")
shape_df.to_csv(os.path.join(output_dir, "shape_statistics.csv"), index=False, encoding="utf-8-sig")
print(f"完整结果已保存至: {os.path.join(output_dir, 'shape_statistics.csv')}")

# 每种文件类型的常见数据统计
file_stats_summary = []
for file_type in set(analysis_df['FileType']):
    # 过滤出当前文件类型的数值型张量
    numeric_tensors = analysis_df[
        (analysis_df['FileType'] == file_type) &
        (analysis_df['Mean'] != 'Empty') &
        (analysis_df['Mean'] != 'N/A') &
        (~analysis_df['Mean'].astype(str).str.contains('Error'))
        ]

    if len(numeric_tensors) > 0:
        try:
            # 计算统计值的统计
            means = pd.to_numeric(numeric_tensors['Mean'], errors='coerce')
            stds = pd.to_numeric(numeric_tensors['Std'], errors='coerce')
            mins = pd.to_numeric(numeric_tensors['Min'], errors='coerce')
            maxs = pd.to_numeric(numeric_tensors['Max'], errors='coerce')

            file_stats_summary.append({
                'FileType': file_type,
                'NumTensors': len(numeric_tensors),
                'MeanOfMeans': means.mean(),
                'MeanOfStds': stds.mean(),
                'OverallMin': mins.min(),
                'OverallMax': maxs.max()
            })
        except Exception as e:
            print(f"计算{file_type}统计信息时出错: {str(e)}")

if file_stats_summary:
    stats_summary_df = pd.DataFrame(file_stats_summary)
    print("\n===== 文件类型统计摘要 =====")
    print(stats_summary_df)
    stats_summary_df.to_csv(os.path.join(output_dir, "file_type_statistics.csv"), index=False, encoding="utf-8-sig")
    print(f"完整结果已保存至: {os.path.join(output_dir, 'file_type_statistics.csv')}")

print(f"\n分析完成! 所有结果已保存到 {output_dir} 目录")
