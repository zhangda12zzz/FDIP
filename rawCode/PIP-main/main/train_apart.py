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
        
        if(key[0:4] == prefix):
            aim_key = key[5:]
            if aim_key in aim_model_dict.keys():
                aim_model_dict[aim_key] = value
                # print(value)
        


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    all_path = conf.paths.weights_file_origin
    
    model = PIP().float().to(device)
    # model = TransPoseNet().float().to(device)
    model_dict = model.state_dict()
    # state_dict = model_dict

    model_all = torch.load(all_path)
    
    
    rnn1 = RNNWithInit(input_size=72,
                            output_size=conf.joint_set.n_leaf * 3,
                            hidden_size=256,
                            num_rnn_layer=2,
                            dropout=0.4).float().to(device)
    rnn2 = RNN(input_size=72 + conf.joint_set.n_leaf * 3,
                    output_size=conf.joint_set.n_full * 3,
                    hidden_size=256,
                    num_rnn_layer=2,
                    dropout=0.4).float().to(device)
    rnn3 = RNN(input_size=72 + conf.joint_set.n_full * 3,
                    output_size=conf.joint_set.n_reduced * 6,
                    hidden_size=256,
                    num_rnn_layer=2,
                    dropout=0.4).float().to(device)
    rnn4 = RNNWithInit(input_size=72 + conf.joint_set.n_full * 3,
                            output_size=24 * 3,
                            hidden_size=256,
                            num_rnn_layer=2,
                            dropout=0.4).float().to(device)
    rnn5 = RNN(input_size=72 + conf.joint_set.n_full * 3,
                    output_size=2,
                    hidden_size=64,
                    num_rnn_layer=2,
                    dropout=0.4).float().to(device)
    rnn1_dict = rnn1.state_dict()
    rnn2_dict = rnn2.state_dict()
    rnn3_dict = rnn3.state_dict()
    rnn4_dict = rnn4.state_dict()
    rnn5_dict = rnn5.state_dict()
    
    
    # PIP
    load_model_para(model_all, rnn1_dict, 'rnn1')
    load_model_para(model_all, rnn2_dict, 'rnn2')
    load_model_para(model_all, rnn3_dict, 'rnn3')
    load_model_para(model_all, rnn4_dict, 'rnn4')
    load_model_para(model_all, rnn5_dict, 'rnn5')

    # model_dict.update(state_dict)
    rnn1.load_state_dict(rnn1_dict)
    rnn2.load_state_dict(rnn2_dict)
    rnn3.load_state_dict(rnn3_dict)
    rnn4.load_state_dict(rnn4_dict)
    rnn5.load_state_dict(rnn5_dict)
    
    
    torch.save(rnn1.state_dict(), 'data/weights/rnnSon/pip_rnn1.pt') 
    torch.save(rnn2.state_dict(), 'data/weights/rnnSon/pip_rnn2.pt') 
    torch.save(rnn3.state_dict(), 'data/weights/rnnSon/pip_rnn3.pt') 
    torch.save(rnn4.state_dict(), 'data/weights/rnnSon/pip_rnn4.pt') 
    torch.save(rnn5.state_dict(), 'data/weights/rnnSon/pip_rnn5.pt') 