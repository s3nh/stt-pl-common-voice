import torchaudio
import torch.nn as nn
import os

from typing import Tuple, Union, Iterable
from torch import Tensor
from torch.utils.data import Dataset

def load_librispeech_item(fieldid: str, path: str, ext_audio: str, ext_txt: str = '.trans.txt') -> Tuple[Tensor, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fieldid.split("-")

    file_text = speaker_id + '-' + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)
    
    fileid_audio = speaker_id + '-' + chapter_id + '-' + utterance_id
    file_audio = fileid_audio + ext_audio
    file__audio = os.path.join(path, speaker_id, chapter_id, file_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(' ', 1)
            if fileid_audio == fileid_text:
                break
        else:
            raise FileNotFoundError(f"Translation not found for {fileid_audio}")
    return (
        waveform, 
        sample_rate, 
        utterance, 
        int(speaker_id), 
        int(chapter_id),
        int(utterance_id),
    )

def walk_files(root : str, suffix: Union[str, Tuple[str]],
                            prefix: bool = False, 
                            remove_suffix: bool = False) -> Iterable[str]:

    root = os.path.expanduser(root)
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        for f in files:
            if f.endswith(suffix):
                if remove_suffix:
                    f = f[: -len(suffix)]
                if prefix:
                    f = os.path.join(dirpath, f)
                yield f

class LIBRISPEECH(Dataset):
    def __init__(self, root: str, url: str, folder_in_archive: str, _ext_txt: str = 'trans.txt', _ext_audio: str = '.flac') -> None:
        self.root = root
        self.url = url
        self.folder_in_archive = folder_in_archive
        self.basename = os.path.basename(url)
        self.basename = self.basename.split(".")[0]
        self.folder_in_archive = os.path.join(self.folder_in_archive, self.basename)
        self._ext_txt = _ext_txt
        self._ext_audio = _ext_audio
        self._path = os.path.join(self.root, self.folder_in_archive)

        walker = walk_files(
            self._path, suffix = self._ext_audio, prefix = False, remove_suffix = True
        ) 
        self.walker = list(walker)

    def __getitem__(self, n : int) -> Tuple[Tensor, int, str, int, int, int]:
        fileid = self._walker[n]
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self):
        return len(self._walker)