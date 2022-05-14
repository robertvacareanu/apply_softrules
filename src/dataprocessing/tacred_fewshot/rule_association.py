"""
Associate rules pregenerated with odinsynth
Helpful to avoid look-up on the fly
"""

from collections import defaultdict
import json
import random
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import tqdm
import numpy as np

@dataclass
class Sentence:
    relation  : str
    token     : List[str]
    subj_start: int
    subj_end  : int
    obj_start : int
    obj_end   : int
    subj_type : str
    obj_type  : str

    def get_tokens_with_entity_types(self) -> List[str]:
        subj_end   = self.subj_end + 1
        obj_end    = self.obj_end + 1
        if self.subj_start < self.obj_start:
            new_token = self.token[:self.subj_start] + [self.subj_type.lower()] + self.token[subj_end:self.obj_start] + [self.obj_type.lower()] + self.token[obj_end:]
        else:
            new_token = self.token[:self.obj_start] + [self.obj_type.lower()] + self.token[obj_end:self.subj_start] + [self.subj_type.lower()] + self.token[subj_end:]
        
        return new_token

    def get_tokens_inbetween_entities(self) -> List[str]:
        subj_end   = self.subj_end + 1
        obj_end    = self.obj_end + 1
        if self.subj_start < self.obj_start:
            new_token = [self.subj_type.lower()] + self.token[subj_end:self.obj_start] + [self.obj_type.lower()]
        else:
            new_token = [self.obj_type.lower()] + self.token[obj_end:self.subj_start] + [self.subj_type.lower()]
        
        return new_token

    def get_tokens_and_wrap_entities(self, head_start: str = "[SUBJ-START]", head_end: str= "[SUBJ-END]", tail_start = "[OBJ-START]", tail_end = "[OBJ-END]"):
        subj_end   = self.subj_end + 1
        obj_end    = self.obj_end + 1
        if self.subj_start < self.obj_start:
            new_token = self.token[:self.subj_start] + [head_start] + self.token[self.subj_start:subj_end] + [head_end] + self.token[subj_end:self.obj_start] + [tail_start] + self.token[self.obj_start:obj_end] + [tail_end] + self.token[obj_end:]
        else:
            new_token = self.token[:self.obj_start] + [tail_start] + self.token[self.obj_start:obj_end] + [tail_end] + self.token[obj_end:self.subj_start] + [head_start] + self.token[self.subj_start:subj_end] + [head_end] + self.token[subj_end:]
        return new_token

@dataclass
class Episode:
    support_sentences: List[Sentence]
    test_sentence:     Sentence

def from_dict_to_s(s_dict: Dict, **kwargs) -> Sentence:
    return Sentence(
        relation   = kwargs.get('relation', s_dict['relation']),
        token      = kwargs.get('token', s_dict['token']),
        subj_start = kwargs.get('subj_start', s_dict['h'][2][0][0]),
        subj_end   = kwargs.get('subj_end', s_dict['h'][2][0][-1]),
        obj_start  = kwargs.get('obj_start', s_dict['t'][2][0][0]),
        obj_end    = kwargs.get('obj_end', s_dict['t'][2][0][-1]),
        subj_type  = kwargs.get('subj_type', s_dict['subj_type']),
        obj_type   = kwargs.get('obj_type', s_dict['obj_type']),
    )
