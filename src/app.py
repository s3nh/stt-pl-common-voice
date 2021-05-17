import io
import json
import numpy as np
import scipy.io.wavfile
import scipy.signal

import torch
import torchaudio

from flask import Flask, jsonify, request
from typing import List, Dict, Text, Union
from utils.utils import read_config

from model.network import SpeechRecognitionModel
from utils.text_transform import TextTransform

app = Flask(__name__)


config = read_config('config/config.yaml')
tt = TextTransform()

def load_model(config: Dict = config):
    state_dict_path = config.get('CHECKPOINT_PATH')
    checkpoint = torch.load(state_dict_path)
    model = SpeechRecognitionModel(
        n_cnn_layers =  config.get('N_CNN_LAYERS'), 
        n_rnn_layers = config.get('N_RNN_LAYERS'),
        rnn_dim = config.get('RNN_DIM'),
        n_class = config.get('N_CLASS'), 
        n_feats = config.get('N_FEATS'), 
        stride = config.get('STRIDE'),
        dropout = config.get('DROPOUT')
    )



