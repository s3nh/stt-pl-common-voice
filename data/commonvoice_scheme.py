import numpy as np
import os
import pandas as pd
import re
import sys
import torchaudio

from typing import Dict, List, Union, Tuple
from torch import Tensor
from torch.utils.data import Dataset

def load_commonvoice_item(fileid : str, utter_file: pd.DataFrame) -> Tuple[List, int, str]:
    """Load CommonVOice Item based on full name path

    Parameters
    -----------
        fileid: full path name
        utter_file: pd.DataFrame object which contain columns as follows:
            path, utter, path_wav
    
    Returns
    ---------
        waveform: np.ndarray
            input file waveform
        sample_rate : int 
            Sample rate of processed file
        utter: str
            Utterance
    """
    waveform, sample_rate = torchaudio.load(fileid)
    sound_id = os.path.basename(fileid)
    _row = utter_file[utter_file.path == sound_id].sentence_clean.values
    if len(_row) > 0:
        utter = _row[0]
        return waveform, sample_rate, utter, fileid
    else:
        raise FileNotFoundError("File not found")


class CommonVoice(Dataset):
    _ext_audio: str = '.mp3'

    def __init__(self, 
    root : str = 'data/common-voice/clips', 
    utter_root : str = 'data/common-voice',
    _ext_txt : str = '.csv', 
    _ext_audio : str = '.mp3' , 
    _utter_file : str = 'validated_cut.csv') -> None:
        self.root = root
        self._ext_txt = _ext_txt
        self._ext_audio = _ext_audio
        self.utter_root = utter_root
        self._utter_file = _utter_file

        self.utterances = self.load_utterance(path = os.path.join(self.utter_root, self._utter_file), sep = '\t')
        _soundfiles = os.listdir(self.root)
        _soundfiles = list(set(_soundfiles).intersection(self.utterances.path.values))
        walker = self.walk_files(_soundfiles)
        self._walker = list(walker)


    def walk_files(self, _soundfiles: List) -> None:
        _soundfiles.sort()
        for dirs in _soundfiles:
            if dirs.endswith(self._ext_audio):
                yield os.path.join(self.root, dirs)

    def load_utterance(self, path: str, sep: str):
        return pd.read_cs(path, sep = sep)
    
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        fileid = self._walker[n]
        return load_commonvoice_item(fileid, utter_file = self.utterances)


    def __len__(self):
        return len(self._walker)
