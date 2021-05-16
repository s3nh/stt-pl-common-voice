import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from model.network import SpeechRecognitionModel
from data.commonvoice_scheme import CommonVoice
from src.train import IterMeter, train, test
from typing import Dict, List, Union
from utils.data_transform import data_processing

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('resume', type = bool, default = False, help = "Resuming from checkpoint")
    args = parser.parse_args()
    return args

def read_config(path: str) -> Dict:
    with open(path, 'r') as confile:
        config = yaml.safe_load(confile)
    return config

def main():

    args = get_args()
    resume = args.resume
    PATH = 'config/config.yaml'
    config = read_config(PATH)
    
    #Initialize params
    n_epochs = config.get('EPOCHS')
    _optim = getattr(torch.optim, config.get('OPTIMIZER'))
    hparams = config.copy()

    train_dataset = CommonVoice()
    device = torch.device("cuda")

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        batch_size = config.get('BATCH_SIZE'), 
        shuffle = True, 
        collate_fn = lambda x : data_processing(x, 'train'), 
    )

    model = SpeechRecognitionModel(
        n_cnn_layers= config.get('N_CNN_LAYERS'), 
        n_rnn_layers= config.get('N_RNN_LAYERS'), 
        rnn_dim = config.get('RNN_DIM'), 
        n_class = config.get('N_CLASS'), 
        n_feats = config.get('N_FEATS'), 
        stride = config.get('STRIDE'), 
        dropout = config.get('DROPOUT') 
    )

