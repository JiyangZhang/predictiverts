import torch
from typing import List, NamedTuple
import json

class DiffTestBatchData(NamedTuple):
    """Stores tensorized batch used in test prediction model."""
    code_seqs: torch.Tensor
    code_lengths: torch.Tensor
    pos_test_seqs: torch.Tensor
    pos_test_lengths: torch.Tensor
    neg_test_seqs: torch.Tensor
    neg_test_lengths: torch.Tensor
    label: torch.Tensor

class DiffPairBatchData(NamedTuple):
    """Stores tensorized batch used in test prediction model."""
    code_seqs: torch.Tensor
    code_lengths: torch.Tensor
    pos_test_seqs: torch.Tensor
    pos_test_lengths: torch.Tensor
    neg_test_seqs: torch.Tensor
    neg_test_lengths: torch.Tensor

def read_data_from_file(filename):
    """Reads in data in the format used for model."""
    with open(filename) as f:
        data = json.load(f)
    return data

def hinge_loss(S_pos, S_neg, hinge_margin):
    """ calculate the hinge loss
        S_pos: pos score Variable (BS,)
        S_neg: neg score Variable (BS,)
        hinge_margin: hinge margin
        returns: batch-averaged loss value
    """
    cost = torch.mean((hinge_margin - (S_pos - S_neg)) *
                      ((hinge_margin - (S_pos - S_neg)) > 0).float())
    return cost

def compute_score(predicted_labels, gold_labels, verbose=True):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    assert(len(predicted_labels) == len(gold_labels))

    for i in range(len(gold_labels)):
        if gold_labels[i]:
            if predicted_labels[i]:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_labels[i]:
                false_positives += 1
            else:
                true_negatives += 1
    
    if verbose:
        print('True positives: {}'.format(true_positives))
        print('False positives: {}'.format(false_positives))
        print('True negatives: {}'.format(true_negatives))
        print('False negatives: {}'.format(false_negatives))
    
    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        precision = 0.0
    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        recall = 0.0
    try:
        f1 = 2*((precision * recall)/(precision + recall))
    except:
        f1 = 0.0
    print(f"precision: {precision}, recall: {recall}, f1: {f1}")
    return precision, recall, f1
