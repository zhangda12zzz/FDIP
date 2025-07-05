# FDIP

FDIP源代码

## 使用说明

### 依赖环境

我们使用 `python 3.8.20`。你可以通过 `requirements.py` 安装所需的包。

### SMPL人体模型

从[这里](https://smpl.is.tue.mpg.de/)下载SMPL模型。你需要点击 `SMPL for Python` 并下载 `version 1.0.0 for Python 2.7 (10 shape PCs)`。然后解压缩。

### 预训练网络权重

从[这里](https://drive.google.com/drive/folders/1ufzKzhfHsYxi-6UeefW3ufu4lFrVcprC?usp=sharing)下载权重文件。最新版本已更新(2024.3.28)。

### 数据集

1. 公开数据集：AMASS、DIP-IMU和TotalCapture数据集可以按照[Transpose](https://github.com/Xinyu-Yi/TransPose)的说明获取。
2. SingleOne-IMU数据集可从[这里](https://drive.google.com/drive/folders/1XYgswm7g_ijSmogk5Fbr3BoxFw8pG9B7?usp=sharing)获取。
3. Miaxmo-IMU数据集可从[这里](https://drive.google.com/drive/folders/13_W1M7mGwCVUJWew0oWnKKUv2dcagZ1I?usp=sharing)获取。
关于我们自建数据集的详细说明将很快整理完成。

### 运行评估

直接下载数据集的`.npy`文件，然后在VS Code中运行`eval.py`。完整的项目执行流程正在整理中。
根据`dataset_eval.py`中设置的不同数据集路径，将会输出不同的结果。
