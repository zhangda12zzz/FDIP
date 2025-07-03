import os
import sys


DIRECTORIES_TO_CLEAN = [
    'GGIP',      # 用于存放检查点 (checkpoints)
    'log',       # 用于存放 TensorBoard 日志
]


def clean_directory_contents(dir_path):
    """
    递归地删除一个目录及其所有子目录中的所有文件，但保留完整的文件夹结构。
    Recursively deletes all files within a directory and its subdirectories,
    but keeps the entire folder structure intact.
    """
    # 首先检查目标路径是否存在且确实是一个目录
    if not os.path.isdir(dir_path):
        print(f"  [信息] 目录不存在，跳过清理: {dir_path}")
        return

    print(f"--- 正在清理目录 '{dir_path}' 内部的文件 ---")
    files_deleted_count = 0

    # os.walk() 是遍历目录树的完美工具
    # 它会为我们提供每个子目录的路径(root)、其中的文件夹列表(dirs)和文件列表(files)
    for root, dirs, files in os.walk(dir_path):
        # 我们只关心文件，所以遍历文件列表
        for file_name in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file_name)
            try:
                # 删除文件
                os.remove(file_path)
                print(f"  [成功] 已删除文件: {file_path}")
                files_deleted_count += 1
            except OSError as e:
                # 如果出现错误（如权限问题），则打印错误信息
                print(f"  [错误] 删除文件 {file_path} 时出错: {e}", file=sys.stderr)

    if files_deleted_count == 0:
        print(f"  在 '{dir_path}' 中未找到需要清理的文件。")
    else:
        print(f"--- 在 '{dir_path}' 中总共删除了 {files_deleted_count} 个文件 ---")
    print("") # 打印一个空行，让输出更清晰


def main():
    """
    主函数，执行所有清理任务。
    Main function to run all cleaning tasks.
    """
    print("===================================================")
    print("=    开始清理训练生成的日志和检查点文件...    =")
    print("=    (注意: 仅删除文件，文件夹结构将被保留)     =")
    print("===================================================\n")

    # 获取脚本所在的当前目录
    current_directory = os.getcwd()
    print(f"将在以下目录执行清理操作: {current_directory}\n")

    # 遍历配置好的文件夹列表，并对每个文件夹执行清理操作
    for dir_to_clean in DIRECTORIES_TO_CLEAN:
        full_path = os.path.join(current_directory, dir_to_clean)
        clean_directory_contents(full_path)

    print("=======================================")
    print("=            所有清理任务完成!          =")
    print("=======================================\n")


if __name__ == '__main__':
    # 运行主函数
    main()

