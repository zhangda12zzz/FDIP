import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import config as conf
from dataset import ImuDataset
from articulate.utils.torch import *
from net import PIP, TransPoseNet

epochs = 10


def load_model_para(save_model, aim_model_dict, prefix):
    for key in save_model:
        value = save_model[key]
        
        aim_key = prefix + '.' + key
        if aim_key in aim_model_dict.keys():
            aim_model_dict[aim_key] = value
            # print(value)
        


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # path_rnn1_tp = 'checkpoints/trial1203/rnn1_biRNN/rnn1__checkpoint_best_4_epoch_0.06968730688095093.pkl'
    path_rnn1 = 'checkpoints/trial1210/standard_biRNN23/rnn1/rnn1__checkpoint_best_51_epoch_0.04690026864409447.pkl'
    path_rnn2 = 'checkpoints/trial1210/standard_biRNN23/rnn2/rnn2__checkpoint_best_20_epoch_0.026802437379956245.pkl'
    # path_rnn2_tp = 'checkpoints/trial1203/rnn2_biRNN_256/rnn2__checkpoint_0_epoch_0.07210943847894669.pkl'
    path_rnn3 = 'checkpoints/trial1210/standard_biRNN23/rnn3/rnn3__checkpoint_best_18_epoch_0.7491500973701477.pkl'
    # path_rnn3_tp = 'checkpoints/trial1203/rnn3_biRNN_256/rnn3__checkpoint_best_8_epoch_0.7864381074905396.pkl'
    path_rnn4 = 'checkpoints/trial1210/standard_biRNN23/rnn4/rnn4__checkpoint_best_8_epoch_2.816467046737671.pkl'
    path_rnn5 = 'checkpoints/trial1210/standard_biRNN23/rnn5_1011/rnn5__checkpoint_best_27_epoch_0.31060364842414856.pkl'

    model = PIP().float().to(device)
    # model = TransPoseNet().float().to(device)
    model_dict = model.state_dict()
    # state_dict = model_dict

    # save_model_rnn1_tp = torch.load(path_rnn1_tp)['model_state_dict']
    # save_model_rnn2_tp = torch.load(path_rnn2_tp)['model_state_dict']
    # save_model_rnn3_tp = torch.load(path_rnn3_tp)['model_state_dict']
    save_model_rnn1 = torch.load(path_rnn1)['model_state_dict']
    save_model_rnn2 = torch.load(path_rnn2)['model_state_dict']
    save_model_rnn3 = torch.load(path_rnn3)['model_state_dict']
    save_model_rnn4 = torch.load(path_rnn4)['model_state_dict']
    save_model_rnn5 = torch.load(path_rnn5)['model_state_dict']
    
    # PIP
    load_model_para(save_model_rnn1, model_dict, 'rnn1')
    load_model_para(save_model_rnn2, model_dict, 'rnn2')
    load_model_para(save_model_rnn3, model_dict, 'rnn3')
    load_model_para(save_model_rnn4, model_dict, 'rnn4')
    load_model_para(save_model_rnn5, model_dict, 'rnn5')
    # tp
    # load_model_para(save_model_rnn1_tp, model_dict, 'pose_s1')
    # load_model_para(save_model_rnn2_tp, model_dict, 'pose_s2')
    # load_model_para(save_model_rnn3_tp, model_dict, 'pose_s3')


    # model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    torch.save(model.state_dict(), 'data/weights/trial1210_refine45_1011.pt') 
    
    