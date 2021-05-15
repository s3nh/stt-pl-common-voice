import numpy as np
import torch
import torchaudio 
from utils.text_transform import TextTransform

torch.cuda.empty_cache()

def avg_wer(wer_scores : np.ndarray, combined_ref_len):
    return float(sum(wer_scores))/ float(combined_ref_len)

def _levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m
    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m 
    distance = np.zeros((2, n + 1), dtype = np.int32)
    for j in range(0, n + 1):
        distance[0][j] = j

    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i  % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)
    return distance[m % 2][n]

def word_errors(reference : str, hypothesis : str, ignore_case : bool = False, delimiter : str = ' '):
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()
    ref_words = reference.split(delimiter)
    hyp_words = reference.split(delimiter)
    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)

def char_errors(reference : str, hypothesis : str, ignore_case : bool = False, remove_space : bool = False):
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()
    join_char = ' '
    if remove_space:
        join_char = ''
    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))
    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_disance), len(reference)


def wer(reference : str, hypothesis : str, ignore_case : bool = False, delimiters : str = ' '):
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case, delimiter)

    if ref_len == 0:
        raise ValueError('Number of words should be greater than 0')
    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case : bool = False, remove_space : bool = False):
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case, remove_space)

    if ref_len == 0:
        raise ValueError("Reference length should be greater than 0")
    cer = float(edit_distance) / ref_len
    return cer