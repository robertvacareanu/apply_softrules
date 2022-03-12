from odinson.ruleutils.queryparser import parse_surface
import tqdm 

from src.dataprocessing.general_dataprocessing import load_dataset_from_jsonl
from src.config import Config
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from odinson.gateway import OdinsonGateway

"""
Read the generated rules and apply them on the dataset
"""
def rule_application(config):

    tacred_dataset = load_dataset_from_jsonl(config.get_path('dataset_path'), config.get('dataset_name'))['train']#.filter(lambda line: line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0)
    tacred_rules   = []
    with open(config.get_path('rules_path')) as fin:
        for line in tqdm.tqdm(fin):
            split = line.split('\t')
            tacred_rules.append((split[1].strip(), split[0].strip()))


    gw = OdinsonGateway.launch(javaopts=['-Xmx32g'])
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
    
    gold = []
    pred = []
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



# python -m src.apps.eval.word_average_eval --path config/base_config.yaml config/eval/rule_application_baseline.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()##.get('word_average_eval')
    rule_application(config)
