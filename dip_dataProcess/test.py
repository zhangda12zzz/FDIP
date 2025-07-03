
import torch

def test_gpu():
    print(f"PyTorch is installed. Version: {torch.__version__}")
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        print("GPU is available!")
        device = torch.device("cuda")
        print(f"当前设备: {device}")

        # 创建一个随机张量并将其移动到GPU上
        x = torch.randn(5, 3).to(device)
        print("在GPU上创建的张量:")
        print(x)

        # 进行一个简单的计算操作，确保GPU正常工作
        y = torch.randn(3, 4).to(device)
        z = torch.mm(x, y)
        print("矩阵乘法结果:")
        print(z)

        # 获取GPU的名称和属性信息
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)} GB")
        print(f"GPU可用内存: {torch.cuda.memory_allocated(0) / (1024 ** 3)} GB")
    else:
        print("GPU is not available.")

import os

def get_current_directory():
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 显示当前工作目录
    print("当前工作目录是:", current_directory)


import pickle
def load_pkl_file():
    a = r'F:\CodeForPaper\Dataset\TotalCapture_Real_60FPS\s2_acting1.pkl'    #(6,3) (72,) (6,3,3)

    with open(a, 'rb') as file:
        data = pickle.load(file, encoding='latin1')  # 或者 encoding='bytes', 'rb') as file:
    print(type(data))
    print([f"{key}: {value[0]}" for key, value in data.items()])


if __name__ == "__main__":
    #test_gpu()
    get_current_directory()
    load_pkl_file()

