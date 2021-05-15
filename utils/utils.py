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
