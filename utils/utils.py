import torch
import yaml 

from typing import List, Dict, Union

def read_config(path : str) -> Dict:
    with open(path, 'r') as confile:
        config = yaml.safe_load(confile)
    return config

def create_params(inuputdict: Dict) -> Dict:
    """
    Params
    --------
    inputdict: Dict 
        List with key elements to config file.

    Returns
    --------
    output_dict: Dict 
        dictionary with config.yaml params
    """

    output_dict: Dict = {}
    for _key in inputdict:
        output_dict[f'{_key}'] = inputdict.get[_key]
    return output_dict


def char_map_process() -> Dict:
    """

    Parameters 
    ------------

    None

    Returns 
    ---------
    Dict 
        Desc.
    """

    char_map_str = """
    ' 0
    <SPACE> 1
    a 2 
    ą 3 
    b 4 
    c 5 
    ć 6 
    d 7
    e 8
    ę 9 
    f 10
    g 11 
    h 12 
    i 13 
    j 14
    k 15
    l 16
    ł 17
    m 18
    n 19
    ń 20
    o 21
    ó 22
    p 23
    q 24
    r 25
    s 26
    ś 27
    t 28
    u 29
    w 30
    x 31
    y 32 
    z 33
    ź 34
    ż 35 
    """
    char_map : Dict = {}
    for line in char_map.strip().splt('\n'):
        ch, index = line.split()
        char_map[ch] = index
    return char_map

def text_to_int(text : str, char_map : Dict) -> List:
    int_sequence: List = []
    for t in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

def inv_char_map(char_map : Dict) -> Dict:
    return {v : k for k, v in char_map.items()}

def int_to_text(ints: List, char_map: Dict) -> Dict:
    string = []
    char_map = inv_char_map(char_map)
    for el in ints:
        string.append(char_map[el])
    return ''.join(string).replace('<SPACE>', ' ')


def spectrogram_convert(data : Optional, data_type : str = 'train'):
    """
    Transform data using torchaudio.transform 

    Parameters 
    -----------
    data : None
        Input data.
    data_type : str
        Define convertion type. "Train by default".
    """
    text_transform = TextTransform()
    #Initialize empty objects
    spectrograms : List = []
    labels : List = []
    input_lengths : List = []
    label_lengths : List = []
    for (waveforms, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type argument is nor properly defined')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(uterrance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lenghts.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first = True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first = True)
    return spectrograms, labels, input_lenghts, label_lengths

def GreedyEncoder(output, labels, label_length, blank_label : int, collapse_repeated : bool, text_transform) -> None:
    """
    Return argmax from predicted output
    """
    arg_maxes = torch.argmax(output, dim = 2)
    decodes : List = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j-1]:
                    continue
                decode.append(index.item())
            decodes.append(text_transform.int_to_text(decode))

def load_params(input_path : str, model : SpeechRecognitionModel) -> None:
    """ Load optimizers for model, optimizer, loss and epochs number 


    Parameters
    ------------
    input_path : str
        Input path. 
    
    model : SpeechRecognitionModel
        Model.

    Returns 
    ---------
        None
    """ 
    _dict = torch.load(input_path)
    return _dict
