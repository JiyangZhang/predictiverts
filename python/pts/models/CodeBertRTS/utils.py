import os

import torch
from typing import List, NamedTuple
import json
from seutil import IOUtils
from pts.Macros import Macros


class BertBatchData(NamedTuple):
    """Stores tensorized batch used in test prediction model."""
    code_seqs: torch.Tensor
    pos_test_seqs: torch.Tensor
    neg_test_seqs: torch.Tensor


class BertTestBatchData(NamedTuple):
    """Stores tensorized batch used in test prediction model."""
    code_seqs: torch.Tensor
    pos_test_seqs: torch.Tensor
    neg_test_seqs: torch.Tensor
    label: torch.Tensor
    ekstazi_label: torch.Tensor
    starts_label: torch.Tensor
