import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pickle
import pandas as pd
from datetime import datetime
from data.dataset_posReg import ImuDataset
from model.net_zd import FDIP_1, FDIP_2, FDIP_3
from train.evaluator import PerFramePoseEvaluator
import gc

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE_TEST = 32

# --- Test Data Path ---
TEST_DATA_FOLDERS = [
    os.path.join("D:\\", "Dataset", "TotalCapture_Real_60FPS", "KaPt", "split_actions"),
    os.path.join("D:\\", "Dataset", "DIPIMUandOthers", "DIP_6", "Detail"),
]

# --- Model Checkpoint Paths ---
CHECKPOINT_TIMESTAMP = "20250803_143022"  # 请替换为实际的时间戳
# CHECKPOINT_BASE_DIR = os.path.join("train","GGIP", f"checkpoints_{CHECKPOINT_TIMESTAMP}")
CHECKPOINT_BASE_DIR = os.path.join("train","GGIP", f"checkpoints")

MODEL1_PATH = os.path.join(CHECKPOINT_BASE_DIR, 'ggip1', 'best_model_fdip1.pth')
MODEL2_PATH = os.path.join(CHECKPOINT_BASE_DIR, 'ggip2', 'best_model_fdip2.pth')
MODEL3_PATH = os.path.join(CHECKPOINT_BASE_DIR, 'ggip3', 'best_model_fdip3.pth')

# --- Output Directory ---
TEST_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TEST_RESULTS_DIR = os.path.join("eval_GGIP", f"test_results_{TEST_TIMESTAMP}")
TEST_PLOTS_DIR = os.path.join(TEST_RESULTS_DIR, "plots")
TEST_DATA_DIR = os.path.join(TEST_RESULTS_DIR, "data")


def clear_memory():
    """清理GPU和CPU内存"""
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


def create_test_directories():
    """创建测试结果目录"""
    dirs = [TEST_RESULTS_DIR, TEST_PLOTS_DIR, TEST_DATA_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Test directories created with timestamp {TEST_TIMESTAMP}:")
    for dir_path in dirs:
        print(f"  - {dir_path}")


def load_test_data():
    """加载测试数据集"""
    print("Loading test dataset...")

    try:
        test_dataset = ImuDataset(TEST_DATA_FOLDERS)
        print(f"Test dataset loaded: {len(test_dataset)} samples")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        print("Please ensure your test dataset paths are correct.")
        sys.exit(1)

    # 数据加载器设置
    num_workers = 0 if sys.platform == "win32" else 4

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,  # 测试时不需要打乱
        pin_memory=True,
        num_workers=num_workers
    )

    print(f"Test data loader created: {len(test_loader)} batches")
    return test_loader


def load_models():
    """加载三个阶段的最佳模型"""
    print("Loading trained models...")

    # 检查模型文件是否存在
    model_paths = [MODEL1_PATH, MODEL2_PATH, MODEL3_PATH]
    model_names = ["FDIP_1", "FDIP_2", "FDIP_3"]

    for path, name in zip(model_paths, model_names):
        if not os.path.exists(path):
            print(f"Error: {name} model file not found at {path}")
            print("Please check the checkpoint paths and ensure training is completed.")
            sys.exit(1)

    # 初始化模型
    model1 = FDIP_1(input_dim=6 * 9, output_dim=5 * 3).to(DEVICE)
    model2 = FDIP_2(input_dim=6 * 12, output_dim=24 * 3).to(DEVICE)
    model3 = FDIP_3(input_dim=288, output_dim=24 * 6).to(DEVICE)

    # 加载模型权重
    try:
        checkpoint1 = torch.load(MODEL1_PATH, map_location=DEVICE)
        model1.load_state_dict(checkpoint1['model_state_dict'])
        print(f"✓ FDIP_1 model loaded from {MODEL1_PATH}")
        print(f"  - Best epoch: {checkpoint1.get('epoch', 'N/A')}")
        print(f"  - Best val loss: {checkpoint1.get('val_loss_min', 'N/A'):.6f}")
        del checkpoint1

        checkpoint2 = torch.load(MODEL2_PATH, map_location=DEVICE)
        model2.load_state_dict(checkpoint2['model_state_dict'])
        print(f"✓ FDIP_2 model loaded from {MODEL2_PATH}")
        print(f"  - Best epoch: {checkpoint2.get('epoch', 'N/A')}")
        print(f"  - Best val loss: {checkpoint2.get('val_loss_min', 'N/A'):.6f}")
        del checkpoint2

        checkpoint3 = torch.load(MODEL3_PATH, map_location=DEVICE)
        model3.load_state_dict(checkpoint3['model_state_dict'])
        print(f"✓ FDIP_3 model loaded from {MODEL3_PATH}")
        print(f"  - Best epoch: {checkpoint3.get('epoch', 'N/A')}")
        print(f"  - Best val loss: {checkpoint3.get('val_loss_min', 'N/A'):.6f}")
        del checkpoint3

    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    # 设置为评估模式
    model1.eval()
    model2.eval()
    model3.eval()

    return model1, model2, model3


def test_stage_losses(model1, model2, model3, test_loader):
    """测试三个阶段的损失"""
    print("\n" + "=" * 80)
    print("Testing Stage Losses on Test Dataset")
    print("=" * 80)

    criterion = nn.MSELoss()

    stage1_losses = []
    stage2_losses = []
    stage3_losses = []

    model1.eval()
    model2.eval()
    model3.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Computing Stage Losses"):
            try:
                acc = data[0].to(DEVICE, non_blocking=True).float()
                ori_6d = data[2].to(DEVICE, non_blocking=True).float()
                p_leaf = data[3].to(DEVICE, non_blocking=True).float()
                p_all = data[4].to(DEVICE, non_blocking=True).float()
                pose_6d_gt = data[6].to(DEVICE, non_blocking=True).float()

                # --- Stage 1: FDIP_1 Loss ---
                input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                target1 = p_leaf.view(-1, p_leaf.shape[1], 15)  # 5个叶节点，每个3D位置

                logits1 = model1(input1)
                loss1 = torch.sqrt(criterion(logits1, target1))  # RMSE
                stage1_losses.append(loss1.item())

                # --- Stage 2: FDIP_2 Loss ---
                p_leaf_logits = model1(input1)
                zeros = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
                p_leaf_pred = torch.cat([zeros, p_leaf_logits.view(p_leaf_logits.shape[0], p_leaf_logits.shape[1], -1)],
                                        dim=2)

                input2 = torch.cat([acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 6, 3)],
                                   dim=-1).view(acc.shape[0], acc.shape[1], -1)
                target2 = torch.cat([torch.zeros_like(p_all[:, :, 0:1, :]), p_all], dim=2).view(p_all.shape[0],
                                                                                                p_all.shape[1],
                                                                                                -1)  # 24*3

                logits2 = model2(input2)
                loss2 = torch.sqrt(criterion(logits2, target2))  # RMSE
                stage2_losses.append(loss2.item())

                # --- Stage 3: FDIP_3 Loss ---
                p_all_pos_flattened = model2(input2)  # FDIP_2 输出的所有24个关节的3D位置，展平
                input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                target3 = pose_6d_gt.view(pose_6d_gt.shape[0], pose_6d_gt.shape[1], -1)  # 24*6 = 144

                logits3 = model3(input_base, p_all_pos_flattened)
                loss3 = torch.sqrt(criterion(logits3, target3))  # RMSE
                stage3_losses.append(loss3.item())

            except Exception as e:
                print(f"Warning: Error processing batch in stage loss testing: {e}")
                continue

    # 计算平均损失
    avg_stage1_loss = np.mean(stage1_losses) if stage1_losses else 0.0
    avg_stage2_loss = np.mean(stage2_losses) if stage2_losses else 0.0
    avg_stage3_loss = np.mean(stage3_losses) if stage3_losses else 0.0

    std_stage1_loss = np.std(stage1_losses) if stage1_losses else 0.0
    std_stage2_loss = np.std(stage2_losses) if stage2_losses else 0.0
    std_stage3_loss = np.std(stage3_losses) if stage3_losses else 0.0

    # 打印结果
    print("\nStage-wise Test Losses (RMSE):")
    print(f"  Stage 1 (FDIP_1 - Leaf Positions): {avg_stage1_loss:.6f} ± {std_stage1_loss:.6f}")
    print(f"  Stage 2 (FDIP_2 - All Positions):  {avg_stage2_loss:.6f} ± {std_stage2_loss:.6f}")
    print(f"  Stage 3 (FDIP_3 - Pose 6D):        {avg_stage3_loss:.6f} ± {std_stage3_loss:.6f}")

    # 保存损失数据
    loss_data = {
        'stage1_losses': stage1_losses,
        'stage2_losses': stage2_losses,
        'stage3_losses': stage3_losses,
        'averages': {
            'stage1': avg_stage1_loss,
            'stage2': avg_stage2_loss,
            'stage3': avg_stage3_loss
        },
        'std_devs': {
            'stage1': std_stage1_loss,
            'stage2': std_stage2_loss,
            'stage3': std_stage3_loss
        }
    }

    return loss_data


def test_final_metrics(model1, model2, model3, test_loader):
    """测试最终的评估指标"""
    print("\n" + "=" * 80)
    print("Testing Final Pipeline Metrics on Test Dataset")
    print("=" * 80)

    clear_memory()

    try:
        evaluator = PerFramePoseEvaluator()
        model1.eval()
        model2.eval()
        model3.eval()

        all_errors = {
            "pos_err": [],
            "mesh_err": [],
            "angle_err": [],
            "jitter_err": []
        }

        print("Running final pipeline evaluation...")
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Computing Final Metrics"):
                try:
                    # --- 完整流水线前向传播 ---
                    acc, ori_6d, pose_6d_gt = [d.to(DEVICE, non_blocking=True).float() for d in
                                               (data[0], data[2], data[6])]

                    # Stage 1: FDIP_1
                    input1 = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                    p_leaf_logits = model1(input1)

                    # Stage 2: FDIP_2
                    zeros1 = torch.zeros(p_leaf_logits.shape[0], p_leaf_logits.shape[1], 3, device=DEVICE)
                    p_leaf_pred = torch.cat([zeros1, p_leaf_logits], dim=2)

                    input2 = torch.cat(
                        (acc, ori_6d, p_leaf_pred.view(p_leaf_pred.shape[0], p_leaf_pred.shape[1], 6, 3)),
                        -1).view(acc.shape[0], acc.shape[1], -1)
                    p_all_pos_flattened = model2(input2)

                    # Stage 3: FDIP_3
                    input_base = torch.cat((acc, ori_6d), -1).view(acc.shape[0], acc.shape[1], -1)
                    pose_pred_flat = model3(input_base, p_all_pos_flattened)

                    # 重塑输出为正确格式
                    batch_size, seq_len = pose_pred_flat.shape[:2]
                    pose_pred = pose_pred_flat.view(batch_size, seq_len, 24, 6)

                    # 计算评估指标
                    errs_dict = evaluator.eval(pose_pred, pose_6d_gt)

                    for key in all_errors.keys():
                        if errs_dict[key].numel() > 0:
                            all_errors[key].append(errs_dict[key].flatten().cpu())

                except Exception as e:
                    print(f"Warning: Error processing batch in final metrics: {e}")
                    continue

        clear_memory()

        # --- 汇总结果 ---
        if all_errors["mesh_err"]:
            print("Processing final evaluation results...")

            # 拼接所有误差数据
            final_errors = {key: torch.cat(val, dim=0) for key, val in all_errors.items() if val}
            avg_errors = {key: val.mean().item() for key, val in final_errors.items()}
            std_errors = {key: val.std().item() for key, val in final_errors.items()}

            # 打印结果
            print("\nFinal Pipeline Test Results:")
            print(
                f"  - Positional Error (cm):      {avg_errors.get('pos_err', 'N/A'):.4f} ± {std_errors.get('pos_err', 'N/A'):.4f}")
            print(
                f"  - Mesh Error (cm):            {avg_errors.get('mesh_err', 'N/A'):.4f} ± {std_errors.get('mesh_err', 'N/A'):.4f}")
            print(
                f"  - Angular Error (deg):        {avg_errors.get('angle_err', 'N/A'):.4f} ± {std_errors.get('angle_err', 'N/A'):.4f}")
            print(
                f"  - Jitter Error (cm/s²):       {avg_errors.get('jitter_err', 'N/A'):.4f} ± {std_errors.get('jitter_err', 'N/A'):.4f}")

            return final_errors, avg_errors, std_errors

        else:
            print("No final evaluation results were generated.")
            return None, None, None

    except Exception as e:
        print(f"Critical error in final metrics evaluation: {e}")
        return None, None, None


def save_results(loss_data, final_errors, avg_errors, std_errors):
    """保存测试结果"""
    print("\nSaving test results...")

    try:
        # 1. 保存损失数据 (JSON格式)
        loss_results = {
            "timestamp": TEST_TIMESTAMP,
            "checkpoint_timestamp": CHECKPOINT_TIMESTAMP,
            "stage_losses": {
                "stage1_fdip1": {
                    "mean": loss_data['averages']['stage1'],
                    "std": loss_data['std_devs']['stage1'],
                    "description": "Leaf positions prediction (RMSE)"
                },
                "stage2_fdip2": {
                    "mean": loss_data['averages']['stage2'],
                    "std": loss_data['std_devs']['stage2'],
                    "description": "All joint positions prediction (RMSE)"
                },
                "stage3_fdip3": {
                    "mean": loss_data['averages']['stage3'],
                    "std": loss_data['std_devs']['stage3'],
                    "description": "6D pose prediction (RMSE)"
                }
            }
        }

        if avg_errors is not None:
            loss_results["final_metrics"] = {
                "pos_err": {"mean": avg_errors.get('pos_err', 'N/A'), "std": std_errors.get('pos_err', 'N/A'),
                            "unit": "cm"},
                "mesh_err": {"mean": avg_errors.get('mesh_err', 'N/A'), "std": std_errors.get('mesh_err', 'N/A'),
                             "unit": "cm"},
                "angle_err": {"mean": avg_errors.get('angle_err', 'N/A'), "std": std_errors.get('angle_err', 'N/A'),
                              "unit": "degrees"},
                "jitter_err": {"mean": avg_errors.get('jitter_err', 'N/A'), "std": std_errors.get('jitter_err', 'N/A'),
                               "unit": "cm/s²"}
            }

        loss_path = os.path.join(TEST_DATA_DIR, f"test_results_{TEST_TIMESTAMP}.json")
        with open(loss_path, 'w') as f:
            json.dump(loss_results, f, indent=2)
        print(f"Test results saved to: {loss_path}")

        # 2. 保存原始数据 (pickle格式)
        if final_errors is not None:
            raw_data = {
                'stage_losses': loss_data,
                'final_errors': final_errors
            }
            raw_path = os.path.join(TEST_DATA_DIR, f"raw_test_data_{TEST_TIMESTAMP}.pkl")
            with open(raw_path, 'wb') as f:
                pickle.dump(raw_data, f)
            print(f"Raw test data saved to: {raw_path}")

        # 3. 保存CSV格式
        csv_data = []

        # 添加阶段损失数据
        for stage, losses in [('FDIP_1', loss_data['stage1_losses']),
                              ('FDIP_2', loss_data['stage2_losses']),
                              ('FDIP_3', loss_data['stage3_losses'])]:
            for loss_val in losses:
                csv_data.append({
                    'metric_type': 'stage_loss',
                    'metric_name': stage,
                    'value': loss_val,
                    'timestamp': TEST_TIMESTAMP
                })

        # 添加最终指标数据
        if final_errors is not None:
            for key, values in final_errors.items():
                for value in values.numpy():
                    csv_data.append({
                        'metric_type': 'final_metric',
                        'metric_name': key,
                        'value': value,
                        'timestamp': TEST_TIMESTAMP
                    })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(TEST_DATA_DIR, f"test_data_{TEST_TIMESTAMP}.csv")
            df.to_csv(csv_path, index=False)
            print(f"CSV test data saved to: {csv_path}")

        # 4. 生成可视化图表
        generate_plots(loss_data, final_errors)

    except Exception as e:
        print(f"Warning: Error saving test results: {e}")


def generate_plots(loss_data, final_errors):
    """生成测试结果的可视化图表"""
    print("Generating visualization plots...")

    try:
        # 1. 阶段损失对比图
        plt.figure(figsize=(10, 6))
        stages = ['FDIP_1\n(Leaf Pos)', 'FDIP_2\n(All Pos)', 'FDIP_3\n(6D Pose)']
        means = [loss_data['averages']['stage1'], loss_data['averages']['stage2'], loss_data['averages']['stage3']]
        stds = [loss_data['std_devs']['stage1'], loss_data['std_devs']['stage2'], loss_data['std_devs']['stage3']]

        bars = plt.bar(stages, means, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Stage-wise Test Losses (RMSE)', fontsize=14, fontweight='bold')
        plt.ylabel('RMSE Loss', fontsize=12)
        plt.xlabel('Pipeline Stages', fontsize=12)

        # 在柱子上添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + std,
                     f'{mean:.4f}±{std:.4f}',
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        stage_loss_path = os.path.join(TEST_PLOTS_DIR, f"stage_losses_{TEST_TIMESTAMP}.png")
        plt.savefig(stage_loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Stage losses plot saved: {stage_loss_path}")

        # 2. 最终指标误差分布图
        if final_errors is not None:
            error_names_map = {
                "pos_err": "Positional Error (cm)",
                "mesh_err": "Mesh Error (cm)",
                "angle_err": "Angular Error (deg)",
                "jitter_err": "Jitter Error (cm/s²)"
            }

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for idx, (key, full_name) in enumerate(error_names_map.items()):
                if key in final_errors:
                    data = final_errors[key].numpy()
                    axes[idx].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[idx].set_title(full_name, fontsize=12, fontweight='bold')
                    axes[idx].set_xlabel('Error Value', fontsize=10)
                    axes[idx].set_ylabel('Frequency', fontsize=10)

                    # 添加统计信息
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    axes[idx].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    axes[idx].legend()
                else:
                    axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(full_name, fontsize=12, fontweight='bold')

            plt.tight_layout()
            final_metrics_path = os.path.join(TEST_PLOTS_DIR, f"final_metrics_distribution_{TEST_TIMESTAMP}.png")
            plt.savefig(final_metrics_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  - Final metrics distribution plot saved: {final_metrics_path}")

    except Exception as e:
        print(f"Warning: Error generating plots: {e}")


def save_summary_report(loss_data, avg_errors, std_errors):
    """保存测试总结报告"""
    try:
        report_path = os.path.join(TEST_RESULTS_DIR, f"test_report_{TEST_TIMESTAMP}.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GGIP Pipeline Test Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Timestamp: {TEST_TIMESTAMP}\n")
            f.write(f"Model Checkpoint Used: {CHECKPOINT_TIMESTAMP}\n")
            f.write(f"Test Dataset: {TEST_DATA_FOLDERS}\n\n")

            f.write("Stage-wise Test Losses (RMSE):\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"  Stage 1 (FDIP_1 - Leaf Positions): {loss_data['averages']['stage1']:.6f} ± {loss_data['std_devs']['stage1']:.6f}\n")
            f.write(
                f"  Stage 2 (FDIP_2 - All Positions):  {loss_data['averages']['stage2']:.6f} ± {loss_data['std_devs']['stage2']:.6f}\n")
            f.write(
                f"  Stage 3 (FDIP_3 - 6D Pose):        {loss_data['averages']['stage3']:.6f} ± {loss_data['std_devs']['stage3']:.6f}\n\n")

            if avg_errors is not None:
                f.write("Final Pipeline Metrics:\n")
                f.write("-" * 50 + "\n")
                unit_labels = {"pos_err": "cm", "mesh_err": "cm", "angle_err": "deg", "jitter_err": "cm/s²"}
                for key, value in avg_errors.items():
                    unit = unit_labels.get(key, "")
                    std = std_errors.get(key, 0.0)
                    f.write(f"  - {key}: {value:.4f} ± {std:.4f} {unit}\n")

            f.write(f"\nGenerated Files:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  - Test results: test_results_{TEST_TIMESTAMP}.json\n")
            f.write(f"  - Raw data: raw_test_data_{TEST_TIMESTAMP}.pkl\n")
            f.write(f"  - CSV data: test_data_{TEST_TIMESTAMP}.csv\n")
            f.write(f"  - Plots: Located in plots/ subdirectory\n")

        print(f"Test report saved to: {report_path}")

    except Exception as e:
        print(f"Warning: Error saving test report: {e}")


def main():
    """主函数"""
    print("=" * 80)
    print("GGIP Pipeline Testing")
    print("=" * 80)

    # 创建输出目录
    create_test_directories()

    # 加载测试数据
    test_loader = load_test_data()

    # 加载训练好的模型
    model1, model2, model3 = load_models()

    # 清理内存
    clear_memory()

    # 测试阶段损失
    loss_data = test_stage_losses(model1, model2, model3, test_loader)

    # 测试最终指标
    final_errors, avg_errors, std_errors = test_final_metrics(model1, model2, model3, test_loader)

    # 保存结果
    save_results(loss_data, final_errors, avg_errors, std_errors)

    # 保存总结报告
    save_summary_report(loss_data, avg_errors, std_errors)

    print("\n" + "=" * 80)
    print("Testing completed successfully!")
    print(f"Results saved in: {TEST_RESULTS_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
