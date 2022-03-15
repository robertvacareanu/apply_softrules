import json
import hashlib
from typing import Dict
import datasets

def convert_tacred_dict(tacred_dict: Dict) -> Dict:

    tokens   = tacred_dict['token']
    e1_start = tacred_dict['subj_start']
    e1_end   = tacred_dict['subj_end'] + 1
    e2_start = tacred_dict['obj_start']
    e2_end   = tacred_dict['obj_end'] + 1

    if (e1_start < e2_start < e2_end < e1_end) or (e2_start < e1_start < e1_end < e2_end) or (e1_start < e2_start < e1_end < e2_end) or (e2_start < e1_start < e2_end < e1_end):
        raise ValueError(f"Overlapping entities in {tacred_dict}")


    new_e1_start = min(e1_start, e2_start)
    new_e1_end   = min(e1_end, e2_end)
    new_e2_start = max(e1_start, e2_start)
    new_e2_end   = max(e1_end, e2_end)

    # NOTE these are provided by TACRED
    if e1_start < e2_start:
        e1_function = 'head'
        e2_function = 'tail'
        e1_type = tacred_dict['subj_type'].lower()
        e2_type = tacred_dict['obj_type'].lower()
    else:
        e1_function = 'tail'
        e2_function = 'head'
        e1_type = tacred_dict['obj_type'].lower()
        e2_type = tacred_dict['subj_type'].lower()

    id_field  = ' '.join(tokens) + str(new_e1_start) + str(new_e1_end) + str(new_e2_start) + str(new_e2_end)
    custom_id = hashlib.md5(id_field.encode('utf-8')).hexdigest()

    return {
        "custom_id"  : custom_id,
        "id"         : tacred_dict['id'],
        "tokens"     : tokens,
        "e1_start"   : new_e1_start,
        "e1_end"     : new_e1_end,
        "e2_start"   : new_e2_start,
        "e2_end"     : new_e2_end,
        "e1"         : tokens[new_e1_start:new_e1_end],
        "e2"         : tokens[new_e2_start:new_e2_end],
        'relation'   : tacred_dict['relation'],
        'e1_type'    : e1_type,
        'e2_type'    : e2_type,
        'e1_function': e1_function,
        'e2_function': e2_function,
    }

def load_dataset_from_jsonl(path):
    d = datasets.load_dataset('text', data_files=path)
    d = d.map(lambda x: convert_tacred_dict(json.loads(x['text'])), batched=False)

    return d

if __name__ == '__main__':
    # data1 = {'id': '61b3a5c8c9a882dcfcd2', 'docid': 'AFP_ENG_20070218.0019.LDC2009T13', 'relation': 'org:founded_by', 'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 'subj_start': 10, 'subj_end': 12, 'obj_start': 0, 'obj_end': 1, 'subj_type': 'ORGANIZATION', 'obj_type': 'PERSON', 'stanford_pos': ['NNP', 'NNP', 'VBD', 'IN', 'NNP', 'JJ', 'NN', 'TO', 'VB', 'DT', 'DT', 'NNP', 'NNP', '-LRB-', 'NNP', '-RRB-', ',', 'VBG', 'DT', 'NN', 'IN', 'CD', 'NNS', 'IN', 'NN', ',', 'VBG', 'JJ', 'NN', 'NNP', 'NNP', 'NNP', 'TO', 'VB', 'NN', 'CC', 'VB', 'DT', 'NN', 'NN', '.'], 'stanford_ner': ['PERSON', 'PERSON', 'O', 'O', 'DATE', 'DATE', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 'stanford_head': [2, 3, 0, 5, 3, 7, 3, 9, 3, 13, 13, 13, 9, 15, 13, 15, 3, 3, 20, 18, 23, 23, 18, 25, 23, 3, 3, 32, 32, 32, 32, 27, 34, 27, 34, 34, 34, 40, 40, 37, 3], 'stanford_deprel': ['compound', 'nsubj', 'ROOT', 'case', 'nmod', 'amod', 'nmod:tmod', 'mark', 'xcomp', 'det', 'compound', 'compound', 'dobj', 'punct', 'appos', 'punct', 'punct', 'xcomp', 'det', 'dobj', 'case', 'nummod', 'nmod', 'case', 'nmod', 'punct', 'xcomp', 'amod', 'compound', 'compound', 'compound', 'dobj', 'mark', 'xcomp', 'dobj', 'cc', 'conj', 'det', 'compound', 'dobj', 'punct']}
    # data2 = {'id': '61b3a65fb9b7111c4ca4', 'docid': 'NYT_ENG_20071026.0056.LDC2009T13', 'relation': 'no_relation', 'token': ['In', '1983', ',', 'a', 'year', 'after', 'the', 'rally', ',', 'Forsberg', 'received', 'the', 'so-called', '``', 'genius', 'award', "''", 'from', 'the', 'John', 'D.', 'and', 'Catherine', 'T.', 'MacArthur', 'Foundation', '.'], 'subj_start': 9, 'subj_end': 9, 'obj_start': 19, 'obj_end': 20, 'subj_type': 'PERSON', 'obj_type': 'PERSON', 'stanford_pos': ['IN', 'CD', ',', 'DT', 'NN', 'IN', 'DT', 'NN', ',', 'NNP', 'VBD', 'DT', 'JJ', '``', 'NN', 'NN', "''", 'IN', 'DT', 'NNP', 'NNP', 'CC', 'NNP', 'NNP', 'NNP', 'NNP', '.'], 'stanford_ner': ['O', 'DATE', 'O', 'DURATION', 'DURATION', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'O'], 'stanford_head': [2, 11, 11, 5, 11, 8, 8, 5, 11, 11, 0, 16, 16, 16, 16, 11, 16, 21, 21, 21, 16, 21, 26, 26, 26, 21, 11], 'stanford_deprel': ['case', 'nmod', 'punct', 'det', 'nmod:tmod', 'case', 'det', 'nmod', 'punct', 'nsubj', 'ROOT', 'det', 'amod', 'punct', 'compound', 'dobj', 'punct', 'case', 'det', 'compound', 'nmod', 'cc', 'compound', 'compound', 'compound', 'conj', 'punct']}
    # data3 = {"id": "e7798379350da8869eba", "docid": "AFP_ENG_20100223.0018", "relation": "per:origin", "token": ["URGENT", "\\u00a5", "\\u00a5", "\\u00a5", "Leading", "Cuban", "dissident", "dies", "on", "hunger", "strike", ":", "hospital", "havana", ",", "Feb", "23", ",", "2010", "-LRB-", "AFP", "-RRB-", "Leading", "Cuban", "dissident", "Orlando", "Zapata", "died", "Tuesday", "in", "a", "Havana", "hospital", "on", "the", "85th", "day", "of", "a", "hunger", "strike", ",", "medical", "officials", "told", "AFP", "."], "subj_start": 25, "subj_end": 26, "obj_start": 23, "obj_end": 23, "subj_type": "PERSON", "obj_type": "NATIONALITY", "stanford_pos": ["JJ", "NN", "NN", "CD", "VBG", "JJ", "JJ", "VBZ", "IN", "NN", "NN", ":", "NN", "NN", ",", "NNP", "CD", ",", "CD", "-LRB-", "NN", "-RRB-", "VBG", "JJ", "JJ", "NNP", "NNP", "VBD", "NNP", "IN", "DT", "NNP", "NN", "IN", "DT", "JJ", "NN", "IN", "DT", "NN", "NN", ",", "JJ", "NNS", "VBD", "NN", "."], "stanford_ner": ["O", "O", "MONEY", "MONEY", "O", "MISC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "DATE", "DATE", "DATE", "DATE", "O", "ORGANIZATION", "O", "O", "MISC", "O", "PERSON", "PERSON", "O", "DATE", "O", "O", "LOCATION", "O", "O", "DATE", "DATE", "DATE", "O", "O", "O", "O", "O", "O", "O", "O", "ORGANIZATION", "O"], "stanford_head": [3, 3, 0, 3, 7, 7, 8, 3, 11, 11, 8, 3, 14, 28, 14, 27, 16, 16, 16, 21, 16, 21, 27, 27, 27, 27, 14, 45, 28, 33, 33, 33, 28, 37, 37, 37, 33, 41, 41, 41, 37, 45, 44, 45, 3, 45, 3], "stanford_deprel": ["amod", "compound", "ROOT", "nummod", "amod", "amod", "nsubj", "acl:relcl", "case", "compound", "nmod", "punct", "compound", "nsubj", "punct", "compound", "nummod", "punct", "nummod", "punct", "appos", "punct", "amod", "amod", "amod", "compound", "appos", "ccomp", "nmod:tmod", "case", "det", "compound", "nmod", "case", "det", "amod", "nmod", "case", "det", "compound", "nmod", "punct", "amod", "nsubj", "dep", "dobj", "punct"]}
    # data4 = {"id": "e7798379350da8869eba", "docid": "AFP_ENG_20100223.0018", "relation": "per:origin", "token": ["URGENT", "\\u00a5", "\\u00a5", "\\u00a5", "Leading", "Cuban", "dissident", "dies", "on", "hunger", "strike", ":", "hospital", "havana", ",", "Feb", "23", ",", "2010", "-LRB-", "AFP", "-RRB-", "Leading", "Cuban", "dissident", "Orlando", "Zapata", "died", "Tuesday", "in", "a", "Havana", "hospital", "on", "the", "85th", "day", "of", "a", "hunger", "strike", ",", "medical", "officials", "told", "AFP", "."], "subj_start": 25, "subj_end": 26, "obj_start": 23, "obj_end": 23, "subj_type": "PERSON", "obj_type": "NATIONALITY", "stanford_pos": ["JJ", "NN", "NN", "CD", "VBG", "JJ", "JJ", "VBZ", "IN", "NN", "NN", ":", "NN", "NN", ",", "NNP", "CD", ",", "CD", "-LRB-", "NN", "-RRB-", "VBG", "JJ", "JJ", "NNP", "NNP", "VBD", "NNP", "IN", "DT", "NNP", "NN", "IN", "DT", "JJ", "NN", "IN", "DT", "NN", "NN", ",", "JJ", "NNS", "VBD", "NN", "."], "stanford_ner": ["O", "O", "MONEY", "MONEY", "O", "MISC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "DATE", "DATE", "DATE", "DATE", "O", "ORGANIZATION", "O", "O", "MISC", "O", "PERSON", "PERSON", "O", "DATE", "O", "O", "LOCATION", "O", "O", "DATE", "DATE", "DATE", "O", "O", "O", "O", "O", "O", "O", "O", "ORGANIZATION", "O"], "stanford_head": [3, 3, 0, 3, 7, 7, 8, 3, 11, 11, 8, 3, 14, 28, 14, 27, 16, 16, 16, 21, 16, 21, 27, 27, 27, 27, 14, 45, 28, 33, 33, 33, 28, 37, 37, 37, 33, 41, 41, 41, 37, 45, 44, 45, 3, 45, 3], "stanford_deprel": ["amod", "compound", "ROOT", "nummod", "amod", "amod", "nsubj", "acl:relcl", "case", "compound", "nmod", "punct", "compound", "nsubj", "punct", "compound", "nummod", "punct", "nummod", "punct", "appos", "punct", "amod", "amod", "amod", "compound", "appos", "ccomp", "nmod:tmod", "case", "det", "compound", "nmod", "case", "det", "amod", "nmod", "case", "det", "compound", "nmod", "punct", "amod", "nsubj", "dep", "dobj", "punct"]}
    # # print(convert_tacred_dict(data1))
    # # print(convert_tacred_dict(data2))
    # # print(convert_tacred_dict(data3))
    # print(convert_tacred_dict(data4))
    # exit()
    import datasets
    from datasets import load_dataset
    # d = datasets.load_dataset('json', data_files='data_sample/tacred/sample.jsonl')
    d = load_dataset_from_jsonl('data_sample/tacred/sample.jsonl')
    print(d)
    print(d['train'][0])
    print(d['train'][1])
    print(d['train'][2])
    print(d['train'][3])

