'''
 @Date  : 12/09/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import json
import torch
from typing import List
import numpy as np
from Constants import *

max_num_candidates = 0
max_num_tokens = 0
max_num_sents = 0

def count_maximum():

    def load_data(json_file):
        global max_num_candidates
        global max_num_tokens
        global max_num_sents
        data = json.load(open(json_file, 'r', encoding='utf-8'))
        print('number of instances: ', len(data))
        for instance in data:
            num_candidates = instance['total_loc_candidates']
            num_tokens = instance['total_tokens']
            num_sents = instance['total_sents']
            if num_candidates > max_num_candidates:
                max_num_candidates = num_candidates
            if num_tokens > max_num_tokens:
                max_num_tokens = num_tokens
            if num_sents > max_num_sents:
                max_num_sents = num_sents

    load_data('./data/train.json')
    print('train')
    load_data('./data/dev.json')
    print('dev')
    load_data('./data/test.json')
    print('test')

    print(f'max number of candidates: {max_num_candidates}')
    print(f'max number of tokens: {max_num_tokens}')
    print(f'max number of sentences: {max_num_sents}')


def find_allzero_rows(vector: torch.IntTensor) -> torch.BoolTensor:
    """
    Find all-zero rows of a given tensor, which is of size (batch, max_sents, max_tokens).
    This function is used to find unmentioned sentences of a certain entity/location.
    So the input tensor is typically a entity_mask or loc_mask.
    Return:
        a BoolTensor indicating that a all-zero row is True. Convenient for masked_fill.
    """
    assert vector.dtype == torch.int
    column_sum = torch.sum(vector, dim = -1)
    return column_sum == 0


def compute_state_accuracy(pred: List[List[int]], gold: List[List[int]], pad_value: int) -> (int, int):
    """
    Given the predicted tags and gold tags, compute the prediction accuracy.
    Note that we first need to deal with the padded parts of the gold tags.
    """
    assert len(pred) == len(gold)
    unpad_gold = [unpad(li, pad_value = pad_value) for li in gold]
    correct_pred = 0
    total_pred = 0

    for i in range(len(pred)):
        assert len(pred[i]) == len(unpad_gold[i])
        total_pred += len(pred[i])
        correct_pred += np.sum(np.equal(pred[i], unpad_gold[i]))

    return correct_pred.item(), total_pred


def compute_loc_accuracy(logits: torch.FloatTensor, gold: torch.IntTensor, pad_value: int) -> (int, int):
    """
    Given the generated location logits and the gold location sequence, compute the location prediction accuracy.
    Args:
        logits - size (batch, max_sents, max_cands)
        gold - size (batch, max_sents)
        pad_value - elements with this value will not count in accuracy
    """
    pred = torch.argmax(logits, dim = -1)
    assert pred.size() == gold.size()

    total_pred = torch.sum(gold != pad_value)  # total number of valid elements
    correct_pred = torch.sum(pred == gold)  # the model cannot predict PAD, NIL or UNK, so all padded positions should be false

    return correct_pred.item(), total_pred.item()


def get_report_time(total_batches: int, report_times: int) -> List[int]:
    """
    Given the total number of batches in an epoch and the report times per epoch,
    compute on which timesteps do we need to report
    e.g. total_batches = 25, report_times = 3, then we should report on batch number [8, 16, 25]
    Batch numbers start from one.
    """
    report_span = round(total_batches / report_times)
    report_batch = [i * report_span for i in range(1, report_times)]
    report_batch.append(total_batches)
    return report_batch


def unpad(source: List[int], pad_value: int) -> List[int]:
    """
    Remove padded elements from a list
    """
    return [x for x in source if x != pad_value]


def mean(source: List) -> float:
    return sum(source) / len(source)

