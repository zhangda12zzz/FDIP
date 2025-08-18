import os
import shutil
from pathlib import Path


def move_npz_files(source_dir):
    source_path = Path(source_dir)
    # 创建与源目录同级的 npz 目录
    npz_dir = source_path.parent / "npz"
    npz_dir.mkdir(exist_ok=True)

    # 递归查找所有 npz 文件
    for npz_file in source_path.rglob("*.npz"):
        # 移动文件到 npz 目录
        destination = npz_dir / npz_file.name

        # 如果目标文件已存在，添加数字后缀
        counter = 1
        original_dest = destination
        while destination.exists():
            stem = original_dest.stem
            suffix = original_dest.suffix
            destination = npz_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.move(str(npz_file), str(destination))
        print(f"移动: {npz_file} -> {destination}")


# 使用示例
source_directory = "D:\Dataset\AMASS\MoSh\MPI_mosh"  # 替换为实际路径
move_npz_files(source_directory)
