import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import articulate as art
import config as conf

def cal_mean(a_list):
    means = []
    mediums = []
    for ten in a_list:
        res = ten.mean()
        med = ten.quantile(q=0.5)
        means.append(res)
        mediums.append(med)
    return sum(means)/len(means), sum(mediums)/len(mediums)

def cal_mean_30(a_list):
    means = []
    mediums = []
    for ten in a_list:
        ten = ten/30.0
        res = ten.mean()
        med = ten.quantile(q=0.5)
        means.append(res)
        mediums.append(med)
    return sum(means)/len(means), sum(mediums)/len(mediums)


amass_acc = torch.load('data/dataset_work/AMASS/train/vacc.pt')
amass_acc_mean = cal_mean(amass_acc)    # 0.0028, -0.0169
dip_train_acc = torch.load('data/dataset_work/DIP_IMU/train/vacc.pt')
dip_train_acc_mean = cal_mean_30(dip_train_acc)    # -0.0207, -0.0396
tc_train_acc = torch.load('data/dataset_work/TotalCapture/train/vacc.pt')
tc_train_acc_mean = cal_mean(tc_train_acc)  # -0.0018, -0.0227

dip_test = torch.load('data/dataset_work/DIP_IMU/test.pt')
dip_test_acc = dip_test['acc']
dip_test_acc_mean = cal_mean_30(dip_test_acc)  # -0.0252, -0.0481
tc_test = torch.load('data/dataset_work/TotalCapture/test.pt')
tc_test_acc = tc_test['acc']
tc_test_acc_mean = cal_mean(tc_test_acc)    # -0.0018, -0.0227

print(amass_acc)
