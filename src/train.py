import logging
import os
import threading
import time
import torch 
import torch.nn.functional as F

from model.network import SpeechRecognitionModel
from typing import Dict
from utils.data_transform import wer, cer



def worker(arg, **kwargs):
    while not arg['stop']:
        logging.debug(f"Epoch number : {kwargs.get('epoch')}: Loss function value {kwargs.get('loss')}")


class IterMeter(object):
    def __init__(self):
        self.val : int = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model : SpeechRecognitionModel, device: str, train_loader, 
        criterion, optimizer, epoch: int, item_meter: int, config: Dict) -> None:
        model.train()
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lenghts, label_lengths = _data
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(spectrograms)
            output = F.log_softmax(output, dim = 2)
            output = output.transpose(0, 1)
            loss = criterion(output, labels, input_lenghts, label_lenghts)

            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_meter.step()

            if batch_idx % 100 == 0 or batch_idx == len(data):
                print(f"train epoch {epoch} {loss.item()} {100.0 * batch_idx/len(train_loader)}")

        os.makedirs('saved_models', exist_ok = True)
        if epoch % 5 == 0:
            PATH = os.path.join('saved_models', f'COMMON_VOICE_PROPER_DICT_{config.get("epoch")}_{config.get("LANGUAGE")}.pt')
            torch.save({
                'epoch' : epoch, 
                'model_state_dict' : model.state_dict(), 
                'optimizer_state_dict' : optimizer.state_dict(), 
                'loss' : loss
            },
            PATH)

def test(model, device, test_loader, criterion, epoch, iter_meter):
    model.eval()
    test_loss = 0.0
    test_cer, test_wer = [], []
    with torch.no_grad():
        
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            output = model(spectrograms)
            output = F.log_softmax(output, dim = 2)
            output = output.transpose(0, 1) #(time, batch, n_class)
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)
            decoded_preds, decoded_target = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for el in range(len(decoded_preds)):
                test_cer.append(cer(decoded_target[el], decoded_preds[el]))
                test_wer.append(wer(decoded_target[el], decoded_preds[el]))
    avg_cer = sum(tes_cer)/len(test_cer)
    avg_ver = sum(test_wer)/len(test_wer)
    return avg_cer, avg_ver 