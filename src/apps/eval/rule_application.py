from odinson.ruleutils.queryparser import parse_surface
import tqdm 
import json

from src.dataprocessing.general_dataprocessing import load_dataset_from_jsonl
from src.config import Config
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from odinson.gateway import OdinsonGateway

from src.utils import combine_result_dictionaries, get_ees_from_config, split_chunks
from src.model.baseline.word_rule_expander import Rule

def worker(params):
    gw = OdinsonGateway.launch(javaopts=['-Xmx4g'])
    index_path = params[0]
    ee = gw.open_index(index_path)
    tacred_rules = params[1]
    doc_to_rules_matched = {}

    # print("Apply each rule")

    for tr in tacred_rules:
        result = ee.search(str(tr.to_ast()))
        for doc in result.docs:
            docId = ee.extractor_engine.index().doc(doc.doc).getValues("docId")[0]
            if docId not in doc_to_rules_matched:
                doc_to_rules_matched[docId] = defaultdict(int)
            doc_to_rules_matched[docId][tr[0]] += 1
    return doc_to_rules_matched
"""
Read the generated rules and apply them on the dataset
"""
def rule_application(config):

    tacred_dataset = load_dataset_from_jsonl(config.get_path('dataset_path'), config.get('dataset_name'))['train']#.filter(lambda line: line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0)
    tacred_rules   = []
    with open(config.get_path('rules_path')) as fin:
        for line in tqdm.tqdm(fin):
            # split = line.split('\t')
            rule  = Rule.from_dict(json.loads(line))
            tacred_rules.append(rule)


    gw = OdinsonGateway.launch(javaopts=['-Xmx4g'])
    ees_paths = get_ees_from_config(None, config)
    # print(ees_paths)
    # exit()
    # gws_duplicated = [gw for x in range(len(ees_paths))]
    tacred_rules_split = split_chunks(tacred_rules, len(ees_paths))
    ees_rules = zip(ees_paths, tacred_rules_split)
    import multiprocessing
    pool = multiprocessing.Pool(len(ees_paths))
    result_dictionaries = pool.map(worker, ees_rules)
    merged_result = combine_result_dictionaries(result_dictionaries)
    # result = { k:max(v.items(), key=lambda x: x[1])[0]  for (k, v) in merged_result.items() }
    gold = []
    pred = []
    prefix = config.get('odinson_doc_file_prefix')
    for i, line in tqdm.tqdm(enumerate(tacred_dataset)):
        gold.append(line['relation'])
        prediction = list(merged_result.get(f'{prefix}_{i}', {}).items())
        if len(prediction) == 0:
            pred.append(config.get('no_relation_label'))
        else:
            pred.append(max(prediction, key=lambda x: x[1])[0])
    # print(gold[:10])
    # print(pred[:10])
    # print(config.get('no_relation_label'))
    # exit()
    # doc_to_rules_matched = {}
    # print("Apply each rule")

    # for tr in tqdm.tqdm(tacred_rules):
    #     result = ee.search(tr[1])
    #     for doc in result.docs:
    #         docId = ee.extractor_engine.index().doc(doc.doc).getValues("docId")[0]
    #         if docId not in doc_to_rules_matched:
    #             doc_to_rules_matched[docId] = defaultdict(int)
    #         doc_to_rules_matched[docId][tr[0]] += 1
    
    # gold = []
    # pred = []
    # prefix = config.get('odinson_doc_file_prefix')
    # for i, line in tqdm.tqdm(enumerate(tacred_dataset)):
    #     gold.append(line['relation'])
    #     prediction = doc_to_rules_matched.get(f'{prefix}_{i}', {})
    #     if len(prediction) == 0:
    #         pred.append(config.get('no_relation_label'))
    #     else:
    #         pred.append(max(prediction, key=lambda x: x[1])[0])

    print('accuracy: ', accuracy_score(gold, pred))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="micro"))
    print('precision: ', precision_score(gold, pred, average="micro"))
    print('recall: ', recall_score(gold, pred, average="micro"))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="macro"))
    print('precision: ', precision_score(gold, pred, average="macro"))
    print('recall: ', recall_score(gold, pred, average="macro"))



# python -m src.apps.eval.rule_application --path config/base_config.yaml config/odinson_index.yaml config/eval/rule_application_baseline.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()##.get('word_average_eval')
    rule_application(config)
