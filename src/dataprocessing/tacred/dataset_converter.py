from typing import Dict

def convert_tacred_dict(tacred_dict: Dict) -> Dict:

    tokens   = tacred_dict['token']
    e1_start = tacred_dict['subj_start']
    e1_end   = tacred_dict['subj_end'] + 1
    e2_start = tacred_dict['obj_start']
    e2_end   = tacred_dict['obj_end'] + 1

    return {
        "id"      : tacred_dict['id'],
        "tokens"  : tokens,
        "e1_start": tacred_dict['subj_start'],
        "e1_end"  : tacred_dict['subj_end'] + 1,
        "e2_start": tacred_dict['obj_start'],
        "e2_end"  : tacred_dict['obj_end'] + 1,
        "e1"      : tokens[e1_start:e1_end],
        "e2"      : tokens[e2_start:e2_end],
        'relation': tacred_dict['relation'],
    }


if __name__ == '__main__':
    data1 = {'id': '61b3a5c8c9a882dcfcd2', 'docid': 'AFP_ENG_20070218.0019.LDC2009T13', 'relation': 'org:founded_by', 'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 'subj_start': 10, 'subj_end': 12, 'obj_start': 0, 'obj_end': 1, 'subj_type': 'ORGANIZATION', 'obj_type': 'PERSON', 'stanford_pos': ['NNP', 'NNP', 'VBD', 'IN', 'NNP', 'JJ', 'NN', 'TO', 'VB', 'DT', 'DT', 'NNP', 'NNP', '-LRB-', 'NNP', '-RRB-', ',', 'VBG', 'DT', 'NN', 'IN', 'CD', 'NNS', 'IN', 'NN', ',', 'VBG', 'JJ', 'NN', 'NNP', 'NNP', 'NNP', 'TO', 'VB', 'NN', 'CC', 'VB', 'DT', 'NN', 'NN', '.'], 'stanford_ner': ['PERSON', 'PERSON', 'O', 'O', 'DATE', 'DATE', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 'stanford_head': [2, 3, 0, 5, 3, 7, 3, 9, 3, 13, 13, 13, 9, 15, 13, 15, 3, 3, 20, 18, 23, 23, 18, 25, 23, 3, 3, 32, 32, 32, 32, 27, 34, 27, 34, 34, 34, 40, 40, 37, 3], 'stanford_deprel': ['compound', 'nsubj', 'ROOT', 'case', 'nmod', 'amod', 'nmod:tmod', 'mark', 'xcomp', 'det', 'compound', 'compound', 'dobj', 'punct', 'appos', 'punct', 'punct', 'xcomp', 'det', 'dobj', 'case', 'nummod', 'nmod', 'case', 'nmod', 'punct', 'xcomp', 'amod', 'compound', 'compound', 'compound', 'dobj', 'mark', 'xcomp', 'dobj', 'cc', 'conj', 'det', 'compound', 'dobj', 'punct']}
    data2 = {'id': '61b3a65fb9b7111c4ca4', 'docid': 'NYT_ENG_20071026.0056.LDC2009T13', 'relation': 'no_relation', 'token': ['In', '1983', ',', 'a', 'year', 'after', 'the', 'rally', ',', 'Forsberg', 'received', 'the', 'so-called', '``', 'genius', 'award', "''", 'from', 'the', 'John', 'D.', 'and', 'Catherine', 'T.', 'MacArthur', 'Foundation', '.'], 'subj_start': 9, 'subj_end': 9, 'obj_start': 19, 'obj_end': 20, 'subj_type': 'PERSON', 'obj_type': 'PERSON', 'stanford_pos': ['IN', 'CD', ',', 'DT', 'NN', 'IN', 'DT', 'NN', ',', 'NNP', 'VBD', 'DT', 'JJ', '``', 'NN', 'NN', "''", 'IN', 'DT', 'NNP', 'NNP', 'CC', 'NNP', 'NNP', 'NNP', 'NNP', '.'], 'stanford_ner': ['O', 'DATE', 'O', 'DURATION', 'DURATION', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'O'], 'stanford_head': [2, 11, 11, 5, 11, 8, 8, 5, 11, 11, 0, 16, 16, 16, 16, 11, 16, 21, 21, 21, 16, 21, 26, 26, 26, 21, 11], 'stanford_deprel': ['case', 'nmod', 'punct', 'det', 'nmod:tmod', 'case', 'det', 'nmod', 'punct', 'nsubj', 'ROOT', 'det', 'amod', 'punct', 'compound', 'dobj', 'punct', 'case', 'det', 'compound', 'nmod', 'cc', 'compound', 'compound', 'compound', 'conj', 'punct']}


    print(convert_tacred_dict(data1))
    print(convert_tacred_dict(data2))
