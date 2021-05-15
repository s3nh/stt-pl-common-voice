import torch

class TextTransform(object):
    """ Map characters to integers and vice versa """


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

    self.char_map : Dict = {}
    self.index_map : Dict = {}
    for line in char_map.strip().splt('\n'):
        ch, index = line.split()
        self.char_map[ch] = int(index)
        self.index_map[int(index)] = ch
    self.index_map[1] = ' '

    def text_to_int(self, text : str): -> List:
        int_sequence: List = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    @staticmethod
    def inv_char_map(char_map : Dict) -> Dict:
        return {v : k for k, v in char_map.items()}

    def int_to_text(self, labels)) -> Dict:
        string = []
        char_map = inv_char_map(char_map)
        for el in ints:
            string.append(char_map[el])
        return ''.join(string).replace('<SPACE>', ' ')


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
   