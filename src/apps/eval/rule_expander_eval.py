from odinson.ruleutils.queryparser import parse_surface
import tqdm 

from src.dataprocessing.general_dataprocessing import load_dataset_from_jsonl
from src.model.baseline.word_rule_expander import WordRuleExpander
from src.config import Config
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from odinson.gateway import OdinsonGateway

# def helper_function(tuple_l_c):
#     line            = tuple_l_c[0]
#     config          = tuple_l_c[1]
#     tacred_rules    = tuple_l_c[2]
#     md5_to_path_map = tuple_l_c[3]
#     gw              = tuple_l_c[4]

#     md5      = hashlib.md5(' '.join(line['tokens']).encode('utf-8')).hexdigest()
#     doc_path = config.get('odinson_data_dir') + '/' + md5_to_path_map[md5]
#     doc      = Document.from_file(doc_path)
#     ee       = gw.open_memory_index([doc])
#     result   = defaultdict(int)
#     for rule in tacred_rules:
#         rule_str = rule[1]
#         # print(rule_str)
#         if ee.search(rule_str).total_hits > 0:
#             result[rule[0]] += 1  
#     return result


# def helper_function2(ee_tacred_rules):
#     ee           = ee_tacred_rules[0]
#     tacred_rules = ee_tacred_rules[1]

#     result       = defaultdict(int)
#     for rule in tacred_rules:
#         rule_str = rule[1]
#         # print(rule_str)
#         if ee.search(rule_str).total_hits > 0:
#             result[rule[0]] += 1  
#     return result



def rule_expander_eval(config: Config):
    tacred_dataset = load_dataset_from_jsonl(config.get_path('dataset_path'), config.get('dataset_name'))['train'].filter(lambda line: line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0)
    tacred_rules   = []
    wre            = WordRuleExpander(config.get("faiss_index").get_path('index_path'), config.get("faiss_index").get_path('vocab_path'), total_random_indices=config.get("faiss_index").get("total_random_indices"))
    with open(config.get_path('rules_path')) as fin:
        lines = fin.readlines()
        for line in tqdm.tqdm(lines):
            split = line.split('\t')
            rule  = parse_surface(split[0].strip())
            if len(wre.extract_words(rule)) <= config.get('max_rule_length'):
                expansions = wre.rule_expander(rule=rule, similarity_threshold=0.9, k=3)
                for e in expansions:
                    tacred_rules.append((split[1].strip(), e))
            tacred_rules.append((split[1].strip(), split[0].strip()))


    gold = []
    pred = []

    gw = OdinsonGateway.launch(javaopts=['-Xmx10g'])
    ee = gw.open_index(config.get_path('odinson_index_dir'))

    doc_to_rules_matched = {}
    print("Apply each rule")
    for tr in tqdm.tqdm(tacred_rules):
        result = ee.search(tr[1])
        for doc in result.docs:
            docId = ee.extractor_engine.index().doc(doc.doc).getValues("docId")[0]
            if docId not in doc_to_rules_matched:
                doc_to_rules_matched[docId] = defaultdict(int)
            doc_to_rules_matched[docId][tr[0]] += 1
    
        
    prefix = config.get('odinson_doc_file_prefix')
    for i, line in tqdm.tqdm(enumerate(tacred_dataset)):
        gold.append(line['relation'])
        prediction = doc_to_rules_matched.get(f'{prefix}_{i}', {})
        if len(prediction) == 0:
            pred.append(config.get('no_relation_label'))
        else:
            pred.append(max(prediction, key=lambda x: x[1])[0])

    print('accuracy: ', accuracy_score(gold, pred))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="micro"))
    print('precision: ', precision_score(gold, pred, average="micro"))
    print('recall: ', recall_score(gold, pred, average="micro"))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="macro"))
    print('precision: ', precision_score(gold, pred, average="macro"))
    print('recall: ', recall_score(gold, pred, average="macro"))

# python -m src.apps.eval.rule_expander_eval --path config/base_config.yaml config/eval/rule_expander_baseline.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()
    rule_expander_eval(config)


