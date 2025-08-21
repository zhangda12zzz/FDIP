# TRAIN_DATA_FOLDERS = [
#     os.path.join("F:\\", "IMUdata", "TotalCapture_Real_60FPS", "KaPt"),
#     os.path.join("F:\\", "IMUdata", "DIPIMUandOthers", "DIP_6"),
#     os.path.join("F:\\", "IMUdata", "AMASS", "DanceDB", "pt"),
#     os.path.join("F:\\", "IMUdata", "AMASS", "HumanEva", "pt"),
# ]
# VAL_DATA_FOLDERS = [
#     os.path.join("F:\\", "IMUdata", "SingleOne", "pt"),
# ]


import torch


def find_global_max_abs_in_nested_structure(data):
    """递归查找嵌套数据结构中所有tensor的全局最大绝对值"""
    max_values = []

    if isinstance(data, torch.Tensor):
        max_val = data.abs().max().item()
        max_values.append(max_val)

    elif isinstance(data, (list, tuple)):
        for item in data:
            sub_max = find_global_max_abs_in_nested_structure(item)
            max_values.extend(sub_max)

    elif isinstance(data, dict):
        for value in data.values():
            sub_max = find_global_max_abs_in_nested_structure(value)
            max_values.extend(sub_max)

    return max_values


# 将此路径替换为您的实际路径
actual_vacc_file_path = r"F:\IMUdata\AMASS\DanceDB\pt\vacc.pt"

try:
    loaded_data = torch.load(actual_vacc_file_path)

    # 查找所有tensor的最大绝对值
    max_values = find_global_max_abs_in_nested_structure(loaded_data)

    if max_values:
        overall_max = max(max_values)
        print(f"Verification: Global max absolute value in {actual_vacc_file_path}: {overall_max:.6f}")
        print(f"Total tensors analyzed: {len(max_values)}")
    else:
        print(f"Verification: No tensors found in {actual_vacc_file_path}")

except FileNotFoundError:
    print(f"Error: File not found at {actual_vacc_file_path}. Double-check your path.")
except Exception as e:
    print(f"An error occurred during loading/checking: {e}")
