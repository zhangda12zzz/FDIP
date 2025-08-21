import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_vacc_file_structure(data_folders, target_filename="vacc.pt"):
    """专门分析vacc.pt文件的结构"""
    print(f"=== Analyzing {target_filename} file structure ===")

    for folder_idx, folder in enumerate(data_folders):
        print(f"\nFolder {folder_idx + 1}: {folder}")
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        vacc_file_path = os.path.join(folder, target_filename)

        if not os.path.exists(vacc_file_path):
            print(f"❌ {target_filename} not found in this folder")
            continue

        print(f"✅ Found {target_filename}")

        try:
            data = torch.load(vacc_file_path)
            print(f"\n  File: {target_filename}")

            if isinstance(data, dict):
                print("  Structure: Dictionary")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(
                            f"    {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
                    else:
                        print(f"    {key}: {type(value)}")
            elif isinstance(data, torch.Tensor):
                print(
                    f"  Structure: Single Tensor, shape={data.shape}, dtype={data.dtype}, range=[{data.min():.3f}, {data.max():.3f}]")
            elif isinstance(data, (list, tuple)):
                print(f"  Structure: {type(data).__name__} with {len(data)} elements")
                for j, item in enumerate(data):
                    if isinstance(item, torch.Tensor):
                        print(
                            f"    [{j}]: shape={item.shape}, dtype={item.dtype}, range=[{item.min():.3f}, {item.max():.3f}]")
                    else:
                        print(f"    [{j}]: {type(item)}")
            else:
                print(f"  Structure: {type(data)}")

        except Exception as e:
            print(f"  ❌ Error loading {target_filename}: {e}")


def analyze_vacc_distribution(data_folders, target_filename="vacc.pt", threshold=300):
    """专门分析vacc.pt文件中加速度数据的分布（支持张量列表）"""
    print(f"=== Analyzing acceleration data distribution in {target_filename} files ===")

    all_acc_values = []
    outlier_files = []
    stats_by_folder = {}

    for folder_idx, folder in enumerate(data_folders):
        folder_name = os.path.basename(folder)
        stats_by_folder[folder_name] = {
            'has_file': False,
            'total_tensors': 0,
            'tensors_with_outliers': 0,
            'max_acc_overall': 0,
            'total_outliers': 0,
            'total_values': 0,
            'outlier_ratio': 0
        }

        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        vacc_file_path = os.path.join(folder, target_filename)

        if not os.path.exists(vacc_file_path):
            print(f"❌ {target_filename} not found in {folder_name}")
            continue

        stats_by_folder[folder_name]['has_file'] = True
        print(f"\nProcessing {target_filename} in folder: {folder_name}")

        try:
            data = torch.load(vacc_file_path)

            # 处理不同的数据结构
            tensor_list = []

            if isinstance(data, list):
                tensor_list = [item for item in data if isinstance(item, torch.Tensor)]
            elif isinstance(data, tuple):
                tensor_list = [item for item in data if isinstance(item, torch.Tensor)]
            elif isinstance(data, torch.Tensor):
                tensor_list = [data]
            elif isinstance(data, dict):
                # 从字典中提取张量
                for key in ['acc', 'acceleration', 'vacc', 'data']:
                    if key in data:
                        if isinstance(data[key], list):
                            tensor_list = [item for item in data[key] if isinstance(item, torch.Tensor)]
                        elif isinstance(data[key], torch.Tensor):
                            tensor_list = [data[key]]
                        break

                # 如果上面没找到，取第一个列表或张量
                if not tensor_list:
                    for key, value in data.items():
                        if isinstance(value, list):
                            tensor_list = [item for item in value if isinstance(item, torch.Tensor)]
                            break
                        elif isinstance(value, torch.Tensor):
                            tensor_list = [value]
                            break

            if not tensor_list:
                print(f"  ⚠️ Could not find tensor data in {target_filename}")
                continue

            stats_by_folder[folder_name]['total_tensors'] = len(tensor_list)
            print(f"  📊 Found {len(tensor_list)} tensors")

            # 分析每个张量
            tensors_with_outliers = 0
            total_outliers = 0
            total_values = 0
            max_acc_overall = 0

            for tensor_idx, tensor in enumerate(tensor_list):
                if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float64]:
                    acc_values = tensor.flatten().numpy()
                    all_acc_values.extend(acc_values)

                    max_acc_tensor = np.abs(acc_values).max()
                    outlier_count_tensor = np.sum(np.abs(acc_values) > threshold)

                    total_values += len(acc_values)
                    total_outliers += outlier_count_tensor
                    max_acc_overall = max(max_acc_overall, max_acc_tensor)

                    if outlier_count_tensor > 0:
                        tensors_with_outliers += 1
                        print(
                            f"    Tensor[{tensor_idx}] shape={tensor.shape}: max={max_acc_tensor:.1f}, outliers={outlier_count_tensor}/{len(acc_values)}")
                    else:
                        print(f"    Tensor[{tensor_idx}] shape={tensor.shape}: max={max_acc_tensor:.1f}, no outliers")

            # 更新统计信息
            stats_by_folder[folder_name]['tensors_with_outliers'] = tensors_with_outliers
            stats_by_folder[folder_name]['max_acc_overall'] = max_acc_overall
            stats_by_folder[folder_name]['total_outliers'] = total_outliers
            stats_by_folder[folder_name]['total_values'] = total_values
            stats_by_folder[folder_name]['outlier_ratio'] = total_outliers / total_values if total_values > 0 else 0

            if total_outliers > 0:
                outlier_files.append({
                    'folder': folder_name,
                    'file': target_filename,
                    'total_tensors': len(tensor_list),
                    'tensors_with_outliers': tensors_with_outliers,
                    'max_acc': max_acc_overall,
                    'total_outliers': total_outliers,
                    'total_values': total_values,
                    'outlier_ratio': total_outliers / total_values
                })

            print(
                f"  📊 Overall: max_acc={max_acc_overall:.1f}, total_outliers={total_outliers}/{total_values} ({total_outliers / total_values:.3f})")

        except Exception as e:
            print(f"❌ Error processing {target_filename} in {folder_name}: {e}")

    # 打印统计结果
    print(f"\n=== Overall Statistics ===")
    print(f"Files with outliers: {len(outlier_files)}")

    for folder_name, stats in stats_by_folder.items():
        print(f"\n{folder_name}:")
        if stats['has_file']:
            print(f"  ✅ {target_filename} found")
            print(f"  📊 Total tensors: {stats['total_tensors']}")
            print(f"  📊 Tensors with outliers: {stats['tensors_with_outliers']}")
            print(f"  📊 Max acceleration: {stats['max_acc_overall']:.1f}")
            if stats['total_outliers'] > 0:
                print(
                    f"  ⚠️ Total outliers: {stats['total_outliers']}/{stats['total_values']} ({stats['outlier_ratio']:.1%})")
            else:
                print(f"  ✅ No outliers (threshold: {threshold})")
        else:
            print(f"  ❌ {target_filename} not found")

    # 绘制分布图
    if all_acc_values:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(all_acc_values, bins=100, alpha=0.7)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.axvline(x=-threshold, color='r', linestyle='--')
        plt.xlabel('Acceleration Value')
        plt.ylabel('Frequency')
        plt.title(f'Acceleration Distribution in {target_filename} Files (Full Range)')
        plt.legend()

        plt.subplot(1, 2, 2)
        # 只显示 -500 到 500 的范围以便更好观察
        filtered_values = [v for v in all_acc_values if -500 <= v <= 500]
        plt.hist(filtered_values, bins=100, alpha=0.7)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.axvline(x=-threshold, color='r', linestyle='--')
        plt.xlabel('Acceleration Value')
        plt.ylabel('Frequency')
        plt.title(f'Acceleration Distribution in {target_filename} Files (-500 to 500)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{target_filename}_acceleration_distribution.png', dpi=150)
        plt.show()

    return outlier_files, stats_by_folder


def clean_vacc_files_preserve_shape(data_folders, target_filename="vacc.pt", acc_threshold=300, backup=True):
    """专门清理vacc.pt文件中的加速度异常值（保持张量形状不变）"""
    print(f"=== Cleaning {target_filename} files (threshold: ±{acc_threshold}) ===")
    print("🔧 Mode: Clipping outliers while preserving tensor shapes")

    processing_stats = {
        'total_folders': len(data_folders),
        'files_found': 0,
        'files_processed': 0,
        'files_with_outliers': 0,
        'files_clipped': 0,
        'total_tensors_processed': 0,
        'tensors_clipped': 0,
        'total_values_clipped': 0,
        'errors': 0
    }

    for folder in data_folders:
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        folder_name = os.path.basename(folder)
        vacc_file_path = os.path.join(folder, target_filename)

        if not os.path.exists(vacc_file_path):
            print(f"❌ {target_filename} not found in {folder_name}")
            continue

        processing_stats['files_found'] += 1
        print(f"\n📁 Processing {target_filename} in folder: {folder_name}")

        # 创建备份文件夹
        if backup:
            backup_folder = os.path.join(folder, 'backup_original')
            os.makedirs(backup_folder, exist_ok=True)

        try:
            # 加载原始数据
            data = torch.load(vacc_file_path)
            original_data = data  # 保存原始引用用于备份

            # 深拷贝用于处理
            if isinstance(data, dict):
                processed_data = {k: v for k, v in data.items()}
            elif isinstance(data, list):
                processed_data = data.copy()
            else:
                processed_data = data

            # 查找张量列表
            tensor_list = None
            tensor_path = None

            if isinstance(processed_data, list):
                tensor_list = processed_data
                tensor_path = 'root_list'
            elif isinstance(processed_data, torch.Tensor):
                tensor_list = [processed_data]
                tensor_path = 'single_tensor'
            elif isinstance(processed_data, dict):
                # 从字典中查找张量列表
                for key in ['acc', 'acceleration', 'vacc', 'data']:
                    if key in processed_data:
                        if isinstance(processed_data[key], list):
                            tensor_list = processed_data[key]
                            tensor_path = key
                            break
                        elif isinstance(processed_data[key], torch.Tensor):
                            tensor_list = [processed_data[key]]
                            tensor_path = key
                            break

                # 如果没找到，取第一个列表或张量
                if tensor_list is None:
                    for key, value in processed_data.items():
                        if isinstance(value, list):
                            tensor_list = value
                            tensor_path = key
                            break
                        elif isinstance(value, torch.Tensor):
                            tensor_list = [value]
                            tensor_path = key
                            break

            if tensor_list is None:
                print(f"⚠️ No tensor data found in {target_filename}")
                continue

            print(f"🔍 Found tensor list at path: {tensor_path}")
            print(f"📊 Processing {len(tensor_list)} tensors")

            # 创建备份
            if backup:
                backup_path = os.path.join(backup_folder, target_filename)
                torch.save(original_data, backup_path)
                print(f"💾 Backup created: {backup_path}")

            # 处理每个张量
            file_has_outliers = False
            tensors_clipped_in_file = 0
            total_values_clipped_in_file = 0

            for tensor_idx, tensor in enumerate(tensor_list):
                if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float64]:
                    processing_stats['total_tensors_processed'] += 1
                    original_shape = tensor.shape

                    # 检查是否有异常值
                    max_acc = torch.abs(tensor).max().item()
                    outlier_mask = torch.abs(tensor) > acc_threshold
                    outlier_count = outlier_mask.sum().item()
                    total_count = tensor.numel()

                    if outlier_count > 0:
                        file_has_outliers = True
                        tensors_clipped_in_file += 1
                        processing_stats['tensors_clipped'] += 1
                        processing_stats['total_values_clipped'] += outlier_count
                        total_values_clipped_in_file += outlier_count

                        print(
                            f"  ✂️ Tensor[{tensor_idx}] shape={original_shape}: clipping {outlier_count}/{total_count} values (max was {max_acc:.1f})")

                        # 裁剪异常值 - 保持原始形状
                        clipped_tensor = torch.clamp(tensor, -acc_threshold, acc_threshold)
                        tensor_list[tensor_idx] = clipped_tensor

                        # 验证形状没有改变
                        assert clipped_tensor.shape == original_shape, f"Shape mismatch! Original: {original_shape}, Clipped: {clipped_tensor.shape}"

                        # 验证裁剪效果
                        new_max = torch.abs(clipped_tensor).max().item()
                        print(f"    ✅ After clipping: max={new_max:.1f} (should be ≤{acc_threshold})")

                    else:
                        print(f"  ✅ Tensor[{tensor_idx}] shape={original_shape}: no outliers (max={max_acc:.1f})")

            if file_has_outliers:
                processing_stats['files_with_outliers'] += 1
                processing_stats['files_clipped'] += 1

                print(
                    f"📊 File summary: {tensors_clipped_in_file} tensors clipped, {total_values_clipped_in_file} values modified")

                # 保存处理后的数据
                torch.save(processed_data, vacc_file_path)
                print(f"✅ Clipped data saved to {target_filename}")
            else:
                print(f"✅ No outliers found in any tensor")

            processing_stats['files_processed'] += 1

        except Exception as e:
            print(f"❌ Error processing {target_filename} in {folder_name}: {e}")
            processing_stats['errors'] += 1

    # 打印处理结果
    print(f"\n=== Processing Summary ===")
    print(f"Total folders: {processing_stats['total_folders']}")
    print(f"Files found: {processing_stats['files_found']}")
    print(f"Files processed: {processing_stats['files_processed']}")
    print(f"Files with outliers: {processing_stats['files_with_outliers']}")
    print(f"Files clipped: {processing_stats['files_clipped']}")
    print(f"Total tensors processed: {processing_stats['total_tensors_processed']}")
    print(f"Tensors clipped: {processing_stats['tensors_clipped']}")
    print(f"Total values clipped: {processing_stats['total_values_clipped']}")
    print(f"Errors: {processing_stats['errors']}")

    if processing_stats['files_found'] > 0:
        success_rate = (processing_stats['files_processed'] - processing_stats['errors']) / processing_stats[
            'files_found'] * 100
        print(f"Success rate: {success_rate:.1f}%")

    return processing_stats


def verify_vacc_cleaning_results(data_folders, target_filename="vacc.pt", acc_threshold=300):
    """验证vacc.pt文件的清理结果（支持张量列表）"""
    print(f"=== Verifying {target_filename} cleaning results ===")

    for folder in data_folders:
        if not os.path.exists(folder):
            continue

        folder_name = os.path.basename(folder)
        vacc_file_path = os.path.join(folder, target_filename)

        print(f"\nVerifying folder: {folder_name}")

        if not os.path.exists(vacc_file_path):
            print(f"❌ {target_filename} not found (may have been removed)")

            # 检查备份是否存在
            backup_folder = os.path.join(folder, 'backup_original')
            backup_path = os.path.join(backup_folder, target_filename)
            if os.path.exists(backup_path):
                print(f"💾 Original backup exists: {backup_path}")
            continue

        try:
            data = torch.load(vacc_file_path)

            # 提取张量列表
            tensor_list = []
            if isinstance(data, list):
                tensor_list = [item for item in data if isinstance(item, torch.Tensor)]
            elif isinstance(data, torch.Tensor):
                tensor_list = [data]
            elif isinstance(data, dict):
                for key in ['acc', 'acceleration', 'vacc', 'data']:
                    if key in data:
                        if isinstance(data[key], list):
                            tensor_list = [item for item in data[key] if isinstance(item, torch.Tensor)]
                        elif isinstance(data[key], torch.Tensor):
                            tensor_list = [data[key]]
                        break

                if not tensor_list:
                    for key, value in data.items():
                        if isinstance(value, list):
                            tensor_list = [item for item in value if isinstance(item, torch.Tensor)]
                            break
                        elif isinstance(value, torch.Tensor):
                            tensor_list = [value]
                            break

            if tensor_list:
                print(f"📊 Found {len(tensor_list)} tensors")
                max_acc_overall = 0
                total_outliers = 0

                for tensor_idx, tensor in enumerate(tensor_list):
                    if isinstance(tensor, torch.Tensor):
                        max_acc_tensor = torch.abs(tensor).max().item()
                        outlier_count = (torch.abs(tensor) > acc_threshold).sum().item()
                        max_acc_overall = max(max_acc_overall, max_acc_tensor)
                        total_outliers += outlier_count

                        if outlier_count > 0:
                            print(
                                f"  ⚠️ Tensor[{tensor_idx}]: still has {outlier_count} outliers (max={max_acc_tensor:.1f})")
                        else:
                            print(f"  ✅ Tensor[{tensor_idx}]: clean (max={max_acc_tensor:.1f})")

                if total_outliers > 0:
                    print(f"⚠️ Total outliers remaining: {total_outliers}")
                else:
                    print(f"✅ All tensors clean: overall max acceleration = {max_acc_overall:.1f}")
            else:
                print(f"⚠️ Could not extract tensor data for verification")

        except Exception as e:
            print(f"❌ Error verifying {target_filename}: {e}")

        # 检查备份文件夹
        backup_folder = os.path.join(folder, 'backup_original')
        backup_path = os.path.join(backup_folder, target_filename)
        if os.path.exists(backup_path):
            print(f"💾 Backup file exists: {backup_path}")


def main_vacc_processing_pipeline():
    """专门处理vacc.pt文件的流程（支持张量列表）"""

    data_folders = [
        "F:\\IMUdata\\TotalCapture_Real_60FPS\\KaPt",
        "F:\\IMUdata\\DIPIMUandOthers\\DIP_6",
        "F:\\IMUdata\\AMASS\\DanceDB\\pt",
        "F:\\IMUdata\\AMASS\\HumanEva\\pt",
        "F:\\IMUdata\\SingleOne\\pt"
    ]

    target_filename = "vacc.pt"
    threshold = 300

    print(f"🚀 Starting {target_filename} processing pipeline...")
    print(f"🔧 Mode: Preserve tensor shapes, clip outliers to ±{threshold}")

    # 步骤1: 分析文件结构
    print("\n" + "=" * 50)
    print(f"STEP 1: Analyzing {target_filename} file structure")
    print("=" * 50)
    analyze_vacc_file_structure(data_folders, target_filename)

    # 步骤2: 分析数据分布
    print("\n" + "=" * 50)
    print(f"STEP 2: Analyzing acceleration distribution in {target_filename}")
    print("=" * 50)
    outlier_files, folder_stats = analyze_vacc_distribution(data_folders, target_filename, threshold)

    # 步骤3: 用户确认
    outlier_count = len(outlier_files)
    if outlier_count > 0:
        print(f"\n⚠️ Found {outlier_count} {target_filename} files with acceleration > {threshold}")
        print("This will:")
        print(f"  - Create backup copies of original {target_filename} files")
        print(f"  - Clip acceleration values to ±{threshold} while preserving tensor shapes")
        print(f"  - Process each tensor in the list individually")

        confirm = input(f"\nProceed with cleaning {target_filename} files? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Operation cancelled")
            return

        # 步骤4: 执行清理
        print("\n" + "=" * 50)
        print(f"STEP 3: Cleaning {target_filename} files")
        print("=" * 50)
        processing_stats = clean_vacc_files_preserve_shape(data_folders, target_filename, threshold, backup=True)

        # 步骤5: 验证结果
        print("\n" + "=" * 50)
        print(f"STEP 4: Verifying {target_filename} results")
        print("=" * 50)
        verify_vacc_cleaning_results(data_folders, target_filename, threshold)

    else:
        print(f"\n✅ No {target_filename} files with outliers found!")
        processing_stats = None

    print(f"\n🎉 {target_filename} processing pipeline completed!")
    return processing_stats


# 执行专门针对vacc.pt的完整流程
if __name__ == "__main__":
    stats = main_vacc_processing_pipeline()
