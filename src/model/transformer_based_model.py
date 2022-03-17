import pytorch_lightning as pl
import torch
from torch import nn
from transformers import BertModel, BertConfig, AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from typing import Dict, List
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import namedtuple
from dataclasses import asdict, dataclass, make_dataclass


"""
Input:  (rule, sentence)
Output: 
"""
class TransformerBasedScorer(nn.Module):
    def __init__(self):
        pass
