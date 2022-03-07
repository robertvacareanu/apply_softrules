import json
from odinson.gateway.document import Document
from odinson.ruleutils.queryparser import parse_surface
import torch
import torch.nn as nn
import tqdm 
import hashlib
import multiprocessing

from src.model.baseline.word_embedding_baseline import get_word_embedding
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl
from src.model.baseline.word_rule_expander import WordRuleExpander
from src.config import Config
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from odinson.gateway import OdinsonGateway

def helper_function(tuple_l_c):
    line            = tuple_l_c[0]
    config          = tuple_l_c[1]
    tacred_rules    = tuple_l_c[2]
    md5_to_path_map = tuple_l_c[3]
    gw              = tuple_l_c[4]

    md5      = hashlib.md5(' '.join(line['tokens']).encode('utf-8')).hexdigest()
    doc_path = config.get('odinson_data_dir') + '/' + md5_to_path_map[md5]
    doc      = Document.from_file(doc_path)
    ee       = gw.open_memory_index([doc])
    result   = defaultdict(int)
    for rule in tacred_rules:
        rule_str = rule[1]
        # print(rule_str)
        if ee.search(rule_str).total_hits > 0:
            result[rule[0]] += 1  
    return result

def rule_expander_eval(config: Config):
    x = 0

    tacred_dataset = load_dataset_from_jsonl(config.get('dataset_path'))['train'].filter(lambda line: line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0)
    tacred_rules   = []
    wre            = WordRuleExpander(config.get("faiss_index")['index_path'], config.get("faiss_index")['vocab_path'], total_random_indices=config.get("faiss_index")["total_random_indices"])
    with open(config.get('rules_path')) as fin:
        lines = fin.readlines()
        for line in tqdm.tqdm(lines[:100]):
            split = line.split('\t')
            rule  = parse_surface(split[0].strip())
            if len(wre.extract_words(rule)) <= config.get('max_rule_length'):
                expansions = wre.rule_expander(rule=rule, similarity_threshold=0.9, k=3)
                for e in expansions:
                    tacred_rules.append((split[1].strip(), e))
            tacred_rules.append((split[1].strip(), split[0].strip()))

    with open(config.get('odinson_data_dir') + "/md5_to_doc_location.json") as fin:
        md5_to_path_map = json.load(fin)

    gw = OdinsonGateway.launch(javaopts=['-Xmx10g'])

    gold = []
    pred = []



    data = []
    for i, line in tqdm.tqdm(enumerate(tacred_dataset)):
        # FIXME memory inefficient; any way for closures?
        data.append((line, config, tacred_rules, md5_to_path_map, gw))
        # md5      = hashlib.md5(' '.join(line['tokens']).encode('utf-8')).hexdigest()
        # doc_path = config.get('odinson_data_dir') + '/' + md5_to_path_map[md5]
        # doc      = Document.from_file(doc_path)
        # ee       = gw.open_memory_index([doc])
        # result   = defaultdict(int)
        # for rule in tacred_rules:
            # rule_str = rule[1]
            # print(rule_str)
            # if ee.search(rule_str).total_hits > 0:
                # result[rule[0]] += 1

    
    pool = multiprocessing.Pool(40)
    parallel_result = pool.map(helper_function, data)

    for i, line in tqdm.tqdm(enumerate(tacred_dataset)):
        gold.append(line['relation'])
        if len(parallel_result[i]) == 0:
            pred.append('no_relation')
        else:
            pred.append(max(parallel_result[i].items(), key=lambda x: x[1])[0])

    print('accuracy: ', accuracy_score(gold, pred))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="micro"))
    print('precision: ', precision_score(gold, pred, average="micro"))
    print('recall: ', recall_score(gold, pred, average="micro"))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="macro"))
    print('precision: ', precision_score(gold, pred, average="macro"))
    print('recall: ', recall_score(gold, pred, average="macro"))

# python -m src.apps.eval.rule_expander_eval --path config/eval/rule_expander_baseline.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config().get('rule_expander_baseline')
    rule_expander_eval(config)


