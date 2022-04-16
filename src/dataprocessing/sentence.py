from dataclasses import dataclass
from re import S
from typing import Dict, List

@dataclass
class Sentence:
    custom_id  : str
    original_id: str
    tokens     : List[str]
    e1_start   : int
    e1_end     : int
    e2_start   : int
    e2_end     : int
    e1         : List[str]
    e2         : List[str]
    relation   : str
    e1_type    : str
    e2_type    : str
    e1_function: str
    e2_function: str


    def get_tokens_with_entity_types(self) -> List[str]:
        return self.tokens[:self.e1_start] + [self.e1_type] + self.token[self.e1_end:self.e2_start] + [self.e2_type] + self.token[self.e2_end:]

    def get_tokens_and_wrap_entities(self, head_start: str = "[SUBJ-START]", head_end: str= "[SUBJ-END]", tail_start = "[OBJ-START]", tail_end = "[OBJ-END]"):
        # Defensive checks + error for failure
        if self.e1_function == 'head' and self.e2_function == "tail":
            e1_start_token = head_start
            e1_end_token   = head_end
            e2_start_token = tail_start
            e2_end_token   = tail_end
        elif self.e1_function == 'tail' and self.e2_function == "head":
            e1_start_token = tail_start
            e1_end_token   = tail_end
            e2_start_token = head_start
            e2_end_token   = head_end
        else:
            raise ValueError("Unknown entity function")

        return \
                self.tokens[:self.e1_start] + \
                [e1_start_token] + \
                self.tokens[self.e1_start:self.e1_end] + \
                [e1_end_token] + \
                self.tokens[self.e1_end:self.e2_start] + \
                [e2_start_token] + \
                self.tokens[self.e2_start:self.e2_end] + \
                [e2_end_token] + \
                self.tokens[self.e2_end:]

def from_dict(data_dict: Dict) -> Sentence:
    return Sentence(
        custom_id   = data_dict['custom_id'],
        original_id = data_dict['id'],
        tokens      = data_dict['tokens'],
        e1_start    = data_dict['e1_start'],
        e1_end      = data_dict['e1_end'],
        e2_start    = data_dict['e2_start'],
        e2_end      = data_dict['e2_end'],
        e1          = data_dict['e1'],
        e2          = data_dict['e2'],
        relation    = data_dict['relation'],
        e1_type     = data_dict['e1_type'],
        e2_type     = data_dict['e2_type'],
        e1_function = data_dict['e1_function'],
        e2_function = data_dict['e2_function'],
    )


if __name__ == "__main__":
    from src.dataprocessing.tacred.dataset_converter import convert_tacred_dict
    data1 = {'id': '61b3a5c8c9a882dcfcd2', 'docid': 'AFP_ENG_20070218.0019.LDC2009T13', 'relation': 'org:founded_by', 'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 'subj_start': 10, 'subj_end': 12, 'obj_start': 0, 'obj_end': 1, 'subj_type': 'ORGANIZATION', 'obj_type': 'PERSON', 'stanford_pos': ['NNP', 'NNP', 'VBD', 'IN', 'NNP', 'JJ', 'NN', 'TO', 'VB', 'DT', 'DT', 'NNP', 'NNP', '-LRB-', 'NNP', '-RRB-', ',', 'VBG', 'DT', 'NN', 'IN', 'CD', 'NNS', 'IN', 'NN', ',', 'VBG', 'JJ', 'NN', 'NNP', 'NNP', 'NNP', 'TO', 'VB', 'NN', 'CC', 'VB', 'DT', 'NN', 'NN', '.'], 'stanford_ner': ['PERSON', 'PERSON', 'O', 'O', 'DATE', 'DATE', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 'stanford_head': [2, 3, 0, 5, 3, 7, 3, 9, 3, 13, 13, 13, 9, 15, 13, 15, 3, 3, 20, 18, 23, 23, 18, 25, 23, 3, 3, 32, 32, 32, 32, 27, 34, 27, 34, 34, 34, 40, 40, 37, 3], 'stanford_deprel': ['compound', 'nsubj', 'ROOT', 'case', 'nmod', 'amod', 'nmod:tmod', 'mark', 'xcomp', 'det', 'compound', 'compound', 'dobj', 'punct', 'appos', 'punct', 'punct', 'xcomp', 'det', 'dobj', 'case', 'nummod', 'nmod', 'case', 'nmod', 'punct', 'xcomp', 'amod', 'compound', 'compound', 'compound', 'dobj', 'mark', 'xcomp', 'dobj', 'cc', 'conj', 'det', 'compound', 'dobj', 'punct']}
    data1 = convert_tacred_dict(data1)
    print(data1)
    print(from_dict(data1))
    s = from_dict(data1)
    print(s)

    