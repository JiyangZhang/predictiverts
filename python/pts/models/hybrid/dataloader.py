from __future__ import print_function

from random import random

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from seutil import IOUtils

from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.models.hybrid.utils import DiffTestBatchData


class CodeClassDataLoader(object):




    def __init__(self, path_file: Path, word_to_index, batch_size=32, max_len=200, mode="train", input="simple"):
        """
        Args:
            path_file:
            word_to_index:
            batch_size:
        """
        self.batch_size = batch_size
        self.word_to_index = word_to_index
        self.index_to_word = {v: k for k, v in word_to_index.items()}

        # read file
        objs = IOUtils.load_json_stream(path_file)
        self.samples = list()
        for obj in objs:
            # index the inputs
            new_inp_idx = list()
            # index code diff
            diff_token_index = self._indexify(obj["code_diff"][:max_len])
            new_inp_idx.append(diff_token_index)
            # index test class and code
            test_input = []
            for test_code in [obj["test_class"]] + obj["test_class_methods"]:
                test_code_tokens: List[str] = SubTokenizer.sub_tokenize_java_like(test_code)
                test_input += test_code_tokens
            # end for
            test_token_indx = self._indexify(test_input[: max_len])
            new_inp_idx.append(test_token_indx)
            self.samples.append([obj["label"], new_inp_idx])
        # end for
        self.mode = mode

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = int(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        if mode == "train":
            self._shuffle_indices()
        self.report()

    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self) -> int:
        length = 0
        for sample in self.samples:
            for seq in sample[1]:
                length = max(length, len(seq))
        return length

    def _indexify(self, lst_text: List[str]) -> List[int]:
        indices = []
        for word in lst_text:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
            else:
                indices.append(self.word_to_index['__UNK__'])
        return indices

    def _textify(self, lst_indx: List[int]) -> List[str]:
        texts = []
        for id in lst_indx:
            texts.append(self.index_to_word[id])
        return texts

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] = batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        labels, strings = tuple(zip(*batch))

        # get the length of each seq in your batch
        diff_seq_lengths = list()
        test_seq_lengths = list()
        for seq in strings:
            diff_seq_lengths.append(len(seq[0]))
            test_seq_lengths.append(len(seq[1]))
        code_lengths = torch.LongTensor(diff_seq_lengths)
        test_lengths = torch.LongTensor(test_seq_lengths)

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        code_seq_tensor = torch.zeros((len(strings), code_lengths.max())).long()
        test_seq_tensor = torch.zeros((len(strings), test_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(strings, code_lengths)):
            code_seq_tensor[idx, :seqlen] = torch.LongTensor(seq[0])
        for idx, (seq, seqlen) in enumerate(zip(strings, test_lengths)):
            test_seq_tensor[idx, :seqlen] = torch.LongTensor(seq[1])

        # get the label
        true_labels = torch.zeros((len(labels))).long()

        for idx, tgt in enumerate(labels):
            true_labels[idx] = torch.LongTensor([tgt[0]])

        return DiffTestBatchData(code_seq_tensor,
                                 code_lengths,
                                 test_seq_tensor,
                                 test_lengths,
                                 true_labels
                                 )

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.mode == "train":
            self._shuffle_indices()
        else:
            self.indices = np.arange(self.n_samples)
            self.index = 0
            self.batch_index = 0
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
