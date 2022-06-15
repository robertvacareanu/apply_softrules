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



def modify_episodes(episodes_path, rules_path):
    rules_list = pd.read_csv(rules_path, sep='\t').values.tolist()

    rules = defaultdict(list)
    for r in rules_list:
        rules[(r[3].lower(), int(r[4]), int(r[5]), int(r[6]), int(r[7]))].append(r[0])
    with open(episodes_path) as fin:
        data = json.load(fin)
    print(type(data))
    for episode in data[0]:
        meta_train = episode['meta_train']
        for examples_per_relation in meta_train:
            for example in examples_per_relation:
                tokens = ' '.join(example['token']).lower()
                subj_start = example['h'][2][0][0]
                subj_end   = example['h'][2][0][-1]
                obj_start  = example['t'][2][0][0]
                obj_end    = example['t'][2][0][-1]
                if (tokens, subj_start, subj_end, obj_start, obj_end) in rules:
                    example['rule'] = rules[(tokens, subj_start, subj_end, obj_start, obj_end)]
                else:
                    example['rule'] = None

                subj_end   = example['h'][2][0][-1] + 1
                obj_end    = example['t'][2][0][-1] + 1


                if subj_start < obj_start:
                    example['new_token'] = example['token'][:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:]
                    example['new_subj_start'] = len(example['token'][:subj_start])
                    example['new_subj_end']   = len(example['token'][:subj_start]) + 1
                    example['new_obj_start']  = len(example['token'][:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:obj_start])
                    example['new_obj_end']    = len(example['token'][:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:obj_start]) + 1
                else: 
                    example['new_token'] = example['token'][:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:]
                    example['new_subj_start'] = len(example['token'][:obj_start])
                    example['new_subj_end']   = len(example['token'][:obj_start]) + 1
                    example['new_obj_start']  = len(example['token'][:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:subj_start] + [example['subj_type'].lower()])
                    example['new_obj_end']    = len(example['token'][:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:subj_start] + [example['subj_type'].lower()]) + 1
            
        for example in episode['meta_test']:
            tokens = ' '.join(example['token']).lower()
            subj_start = example['h'][2][0][0]
            subj_end   = example['h'][2][0][-1] + 1
            obj_start  = example['t'][2][0][0]
            obj_end    = example['t'][2][0][-1] + 1

            if subj_start < obj_start:
                example['new_token'] = example['token'][:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:]
                example['new_subj_start'] = len(example['token'][:subj_start])
                example['new_subj_end']   = len(example['token'][:subj_start]) + 1
                example['new_obj_start']  = len(example['token'][:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:obj_start])
                example['new_obj_end']    = len(example['token'][:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:obj_start]) + 1

            else:
                example['new_token'] = example['token'][:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:subj_start] + [example['subj_type'].lower()] + example['token'][subj_end:]
                example['new_subj_start'] = len(example['token'][:obj_start])
                example['new_subj_end']   = len(example['token'][:obj_start]) + 1
                example['new_obj_start']  = len(example['token'][:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:subj_start])
                example['new_obj_end']    = len(example['token'][:obj_start] + [example['obj_type'].lower()] + example['token'][obj_end:subj_start]) + 1


    return data, rules

"""
Preparing the Few-Shot TACRED dataset
Format:
'rules', 'rules_relations', 'test_sentence', 'gold_relation'
"""
def prepare_dataset_episodes(data):
    output = []

    for episode, relations in zip(data[0], data[2]):
        meta_train = episode['meta_train']
        meta_test  = episode['meta_test']
        for test_sentence, gr in zip(meta_test, relations[1]):
            ts_sentence = from_dict_to_s(test_sentence)
            rules = []
            rules_relations = []
            if gr not in relations[0]:
                gold_relation = 'no_relation'
            else:
                gold_relation = gr
                # assert(gold_relation in relations[0]) # /SANITY checking that if it is not a no_relation, then the relation is in the support_sentences set
            for support_sentences_per_relation, relation in zip(meta_train, relations[0]):
                for ss in support_sentences_per_relation:
                    if ss['rule']:
                        ss_sentence = from_dict_to_s(ss)
                        for rule in ss['rule']:
                            rules.append(rule)
                            rules_relations.append(relation)
                            assert(relation == ss['relation']) # /SANITY just a sanity check
            output.append({
                'rules'                 : rules,
                'rules_relations'       : rules_relations,
                'test_sentence'         : ts_sentence.get_tokens_inbetween_entities(),
                # 'original_test_sentence': test_sentence['token'],
                # 'subj_start'            : test_sentence['new_subj_start'],
                # 'subj_end'              : test_sentence['new_subj_end'],
                # 'obj_start'             : test_sentence['new_obj_start'],
                # 'obj_end'               : test_sentence['new_obj_end'],
                "span_start"            : min(test_sentence['new_subj_end'], test_sentence['new_obj_end']),
                "span_end"              : max(test_sentence['new_subj_start'], test_sentence['new_obj_start']),
                'gold_relation'         : gold_relation
            })

    return output


# Read the data in the reldict format
# Example of such a file: '/data/nlp/corpora/fs-tacred/few-shot-dev/_train_data.json'
def read_reldict_data(path: str) -> List[Sentence]:
    with open(path) as fin:
        data = json.load(fin)

    output = []
    for key in data:
        for s_dict in data[key]:
            output.append(from_dict_to_s(s_dict))

    return output



def get_data_from_rule_path(
    rules_and_data_pd: str = '/data/nlp/corpora/softrules/rules/tacred_fewshot/train.tsv', 
    relations_to_consider = ['org:city_of_headquarters', 'per:stateorprovince_of_birth', 'org:website', 'per:cause_of_death', 'per:charges', 'org:subsidiaries', 'org:alternate_names', 'no_relation', 'per:title', 'org:number_of_employees/members', 'per:religion', 'per:parents', 'per:country_of_birth', 'org:stateorprovince_of_headquarters', 'org:members', 'per:cities_of_residence', 'org:dissolved', 'org:shareholders', 'per:countries_of_residence', 'per:spouse', 'org:political/religious_affiliation', 'per:city_of_birth', 'per:other_family', 'per:date_of_death', 'per:employee_of', 'per:country_of_death'],
):
    rtc  = set(relations_to_consider)

    data                  = pd.read_csv(rules_and_data_pd, sep='\t')
    rel_to_rules          = defaultdict(list)
    rel_to_sentences      = defaultdict(list)
    sentence_to_tokens    = defaultdict(set)
    entities_to_sentence  = defaultdict(list)
    entities_to_relations = defaultdict(list)

    sentence_to_rules     = defaultdict(list)

    for i, row in list(data.iterrows()):
        if row['relation'] in rtc:
            rel_to_sentences[row['relation']].append(row['inbetween_entities'])
            rel_to_rules[row['relation']].append(row['pattern'])
            entities_to_sentence[(row['first_type'],  row['second_type'])].append(row['inbetween_entities'])
            entities_to_relations[(row['first_type'], row['second_type'])].append(row['pattern'])

            sentence_to_rules[row['inbetween_entities']].append(row['pattern'])

            sentence_tokens = row['inbetween_entities'].split(' ')
            for word in row['pattern_lexicalized_words'].split(' '):
                if word in sentence_tokens:
                    sentence_to_tokens[row['inbetween_entities']].add(sentence_tokens.index(word))
                
    return (rel_to_rules, rel_to_sentences, sentence_to_tokens, entities_to_sentence, entities_to_relations, sentence_to_rules)


def prepare_dataset(relation_to_rules, relation_to_sentences, sentence_to_tokens, entities_to_sentence, entities_to_relations, all_sentences, how_many: 5):
    all_relations = list(relation_to_rules.keys())
    output = []
    sentences_to_relation = { vv:k for (k,v) in relation_to_sentences.items() for vv in v }
    already_set = set()
    for (s, tokens) in tqdm.tqdm(sentence_to_tokens.items()):
        for _ in range(how_many):
            s_relation      = sentences_to_relation[s]
            rule_for_s1      = random.choice(relation_to_rules[s_relation])
            rule_for_s2      = random.choice(relation_to_rules[s_relation])
            random_relation1 = random.choice(all_relations)
            while random_relation1 == s_relation:
                random_relation1 = random.choice(all_relations)

            random_relation2 = random.choice(all_relations)
            while random_relation2 == s_relation:
                random_relation2 = random.choice(all_relations)
            
            random_rule     = random.choice(relation_to_rules[random_relation1])
            random_sentence = random.choice(relation_to_sentences[random_relation2])
            if (rule_for_s1, s) not in already_set:
                good_sentence_tokens = np.zeros(len(s.split(' ')))
                good_sentence_tokens[sorted(list(tokens))] = 1

                random_sentence_tokens = np.zeros(len(random_sentence.split(' ')))
                random_sentence_tokens[sorted(list(sentence_to_tokens[random_sentence]))] = 1

                already_set.add((rule_for_s1, s))
                output.append({
                    'good_rule1'          : [rule_for_s1],
                    'good_sentence'       : s.split(' '),
                    'relation'            : s_relation,
                    'good_sentence_tokens': good_sentence_tokens.astype(int).tolist(),
                    'good_rule2'          : [rule_for_s2],
                    'second_good_sentence': random.choice(relation_to_sentences[s_relation]),

                    'random_rule'         : [random_rule],
                    'random_rule_relation': random_relation1,

                    'random_sentence'         : random_sentence.split(' '),
                    'random_sentence_relation': random_relation2,
                    'random_sentence_tokens'  : random_sentence_tokens.astype(int).tolist(),
                })
    
    return output

def prepare_dataset_maxmargin(relation_to_rules, relation_to_sentences, sentence_to_tokens, entities_to_sentence, entities_to_relations, all_sentences, how_many: 5):
    all_relations = list(relation_to_rules.keys())
    output = []
    sentences_to_relation = { vv:k for (k,v) in relation_to_sentences.items() for vv in v }
    already_set = set()
    for (s, tokens) in tqdm.tqdm(sentence_to_tokens.items()):
        for _ in range(how_many):
            s_relation      = sentences_to_relation[s]
            rule_for_s1      = random.choice(relation_to_rules[s_relation])
            rule_for_s2      = random.choice(relation_to_rules[s_relation])
            random_relation1 = random.choice(all_relations)
            while random_relation1 == s_relation:
                random_relation1 = random.choice(all_relations)

            random_relation2 = random.choice(all_relations)
            while random_relation2 == s_relation:
                random_relation2 = random.choice(all_relations)
            
            random_rule     = random.choice(relation_to_rules[random_relation1])
            random_sentence = random.choice(relation_to_sentences[random_relation2])
            if (rule_for_s1, s) not in already_set:
                good_sentence_tokens = np.zeros(len(s.split(' ')))
                good_sentence_tokens[sorted(list(tokens))] = 1

                random_sentence_tokens = np.zeros(len(random_sentence.split(' ')))
                random_sentence_tokens[sorted(list(sentence_to_tokens[random_sentence]))] = 1

                already_set.add((rule_for_s1, s))
                output.append({
                    'good_rule1'          : [rule_for_s1],
                    'good_sentence'       : s.split(' '),
                    'relation'            : s_relation,
                    'good_sentence_tokens': good_sentence_tokens.astype(int).tolist(),
                    'good_rule2'          : [rule_for_s2],
                    'second_good_sentence': random.choice(relation_to_sentences[s_relation]),

                    'random_rule'         : [random_rule],
                    'random_rule_relation': random_relation1,

                    'random_sentence'         : random_sentence.split(' '),
                    'random_sentence_relation': random_relation2,
                    'random_sentence_tokens'  : random_sentence_tokens.astype(int).tolist(),
                })
    
    return output

# src.dataprocessing.tacred_fewshot.rule_association
if __name__ == "__main__":
    from src.model.util import init_random
    init_random(1)

    # episode_path                  = '/data/nlp/corpora/fs-tacred/few-shot-dev/_train_data.json'
    # rules_path                    = '/data/nlp/corpora/softrules/rules/tacred_fewshot/train_between_entities.tsv'
    # relations_to_consider         = ['org:city_of_headquarters', 'per:stateorprovince_of_birth', 'org:website', 'per:cause_of_death', 'per:charges', 'org:subsidiaries', 'org:alternate_names', 'no_relation', 'per:title', 'org:number_of_employees/members', 'per:religion', 'per:parents', 'per:country_of_birth', 'org:stateorprovince_of_headquarters', 'org:members', 'per:cities_of_residence', 'org:dissolved', 'org:shareholders', 'per:countries_of_residence', 'per:spouse', 'org:political/religious_affiliation', 'per:city_of_birth', 'per:other_family', 'per:date_of_death', 'per:employee_of', 'per:country_of_death']
    # rtr, rts, stt, ets, etr, stor = get_data_from_rule_path(rules_path, relations_to_consider)
    # data                          = [x for x in read_reldict_data(episode_path) if x.relation in relations_to_consider]
    # cnt = 0
    # all_s = [' '.join(x.get_tokens_with_entity_types()) for x in data if x.relation != 'no_relation']
    # for x in all_s:
    #     if x in stt:
    #         cnt += 1
    # prepared_data = prepare_dataset(rtr, rts, stt, ets, etr, data, 50)
    # with open('/data/nlp/corpora/softrules/tacred_fewshot/train/hf_datasets/rules_sentences_pair/train_large2.jsonl', 'w+') as fout:
    #     for line in prepared_data:
    #         fout.write(f'{json.dumps(line)}\n')
    # exit()
    for (pth, sv_pth) in [
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160290.json', '5_way_1_shots_10K_episodes_3q_seed_160290'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160291.json', '5_way_1_shots_10K_episodes_3q_seed_160291'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160292.json', '5_way_1_shots_10K_episodes_3q_seed_160292'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160293.json', '5_way_1_shots_10K_episodes_3q_seed_160293'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160294.json', '5_way_1_shots_10K_episodes_3q_seed_160294'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160290.json', '5_way_5_shots_10K_episodes_3q_seed_160290'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160291.json', '5_way_5_shots_10K_episodes_3q_seed_160291'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160292.json', '5_way_5_shots_10K_episodes_3q_seed_160292'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160293.json', '5_way_5_shots_10K_episodes_3q_seed_160293'),
        ('/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160294.json', '5_way_5_shots_10K_episodes_3q_seed_160294'),
    ]:
        data, rules = modify_episodes(pth, '/data/nlp/corpora/softrules/rules/tacred_fewshot/test_between_entities.tsv')
        # data, rules = modify_episodes('/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160290.json', '/data/nlp/corpora/softrules/rules/tacred_fewshot/train.tsv')
        data = prepare_dataset_episodes(data)
        with open(f'/data/nlp/corpora/softrules/tacred_fewshot/test/hf_datasets/{sv_pth}_inbetweenentities.jsonl', 'w+') as fout:
            for line in data:
                _=fout.write(f'{json.dumps(line)}\n')
