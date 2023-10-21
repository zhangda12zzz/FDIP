# GAIP

Source code of Graph-Based Adversarial Inertial Poser (GAIP).

## Usage

### Dependencies

We use `python 3.7.6`. You can install the package as `requirements.py`.


### SMPL body model

Download SMPL model from [here](https://smpl.is.tue.mpg.de/). You should click `SMPL for Python` and download the `version 1.0.0 for Python 2.7 (10 shape PCs)`. Then unzip it.

### Pre-trained network weights

Download weights from [here](https://drive.google.com/drive/folders/1ufzKzhfHsYxi-6UeefW3ufu4lFrVcprC?usp=sharing).


### Datasets

1. Public dataset: dataset AMASS, DIP-IMU and TotalCapture can be obtained following the instructions from [Transpose](https://github.com/Xinyu-Yi/TransPose).
2. SingleOne-IMU dataset is available from [here](https://drive.google.com/drive/folders/1XYgswm7g_ijSmogk5Fbr3BoxFw8pG9B7?usp=sharing).
3. Miaxmo-IMU dataset is available from [here](https://drive.google.com/drive/folders/13_W1M7mGwCVUJWew0oWnKKUv2dcagZ1I?usp=sharing).
Detailed descriptions about our own dataset will be sorted out soon.

### Run the evaluation

Directly download `.npy` files of datasets, then run `eval.py` in VS Code. Robust project execution process is being sorted out.
According to the different dataset path set in `dataset_eval.py`, different results will be printed.
