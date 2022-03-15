import json
from typing import Dict

import datasets

def convert_custom_conll04_dict(custom_conll04_dict: Dict) -> Dict:
    return None
    
def convert_to_custom_conll04_dict(input_path, save_path):
    with open(input_path) as fin:
        data = json.load(fin)

    output = []
    for line in data:
        relations_in_line = {}
        for r in line['relations']:
            relations_in_line[(r['head'], r['tail'] )] = r['type']
        for i, e1 in enumerate(line['entities']):
            for j, e2 in enumerate(line['entities']):
                if i != j:
                    resulting_dict = {
                        "id"         : 'a',
                        "tokens"     : line['tokens'],
                        "e1_start"   : e1['start'],
                        "e1_end"     : e1['end'],
                        "e2_start"   : e2['start'],
                        "e2_end"     : e2['end'],
                        "e1"         : line['tokens'][e1['start']:e1['end']],
                        "e2"         : line['tokens'][e2['start']:e2['end']],
                        'relation'   : relations_in_line.get((e1['start'], e2['start']), "no_relation"),
                        'e1_type'    : e1['type'],
                        'e2_type'    : e2['type'],
                        'e1_function': 'head',
                        'e2_function': 'tail',
                    }
                    output.append(resulting_dict)



def load_dataset_from_jsonl(path):
    d = datasets.load_dataset('text', data_files=path)

    return d

# python -m src.dataprocessing.conll04.dataset_converter.py
if __name__ == '__main__':
    # data1 = {'relation': 'P931', 'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], 'h': ['tjq', 'Q1331049', [[16]]], 't': ['tanjung pandan', 'Q3056359', [[13, 14]]]}
    # data2 = {'relation': 'P931', 'tokens': ['The', 'name', 'was', 'at', 'one', 'point', 'changed', 'to', 'Nottingham', 'East', 'Midlands', 'Airport', 'so', 'as', 'to', 'include', 'the', 'name', 'of', 'the', 'city', 'that', 'is', 'supposedly', 'most', 'internationally', 'recognisable', ',', 'mainly', 'due', 'to', 'the', 'Robin', 'Hood', 'legend', '.'], 'h': ['east midlands airport', 'Q8977', [[9, 10, 11]]], 't': ['nottingham', 'Q41262', [[8]]]}


    # print(convert_custom_conll04_dict(data1))
    # print(convert_custom_conll04_dict(data2))
    print(convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_train.json", "/data/nlp/corpora/softrules/conll04/conll04_train_custom.jsonl"))
    print(convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_dev.json", "/data/nlp/corpora/softrules/conll04/conll04_dev_custom.jsonl"))
    print(convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_test.json", "/data/nlp/corpora/softrules/conll04/conll04_test_custom.jsonl"))
    