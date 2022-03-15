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
    # data1 = {'tokens': ['The', 'self-propelled', 'rig', 'Avco', '5', 'was', 'headed', 'to', 'shore', 'with', '14', 'people', 'aboard', 'early', 'Monday', 'when', 'it', 'capsized', 'about', '20', 'miles', 'off', 'the', 'Louisiana', 'coast', ',', 'near', 'Morgan', 'City', ',', 'Lifa', 'said.'], 'entities': [{'type': 'Other', 'start': 19, 'end': 21}, {'type': 'Loc', 'start': 23, 'end': 24}, {'type': 'Loc', 'start': 27, 'end': 29}, {'type': 'Peop', 'start': 30, 'end': 31}], 'relations': [{'type': 'Located_In', 'head': 2, 'tail': 1}], 'orig_id': 2447}
    # data2 = {'tokens': ['Annie', 'Oakley', ',', 'also', 'known', 'as', 'Little', 'Miss', 'Sure', 'Shot', ',', 'was', 'born', 'Phoebe', 'Ann', 'Moses', 'in', 'Willowdell', ',', 'Darke', 'County', ',', 'in', '1860', '.'], 'entities': [{'type': 'Peop', 'start': 0, 'end': 2}, {'type': 'Peop', 'start': 6, 'end': 10}, {'type': 'Peop', 'start': 13, 'end': 16}, {'type': 'Loc', 'start': 17, 'end': 21}], 'relations': [{'type': 'Live_In', 'head': 0, 'tail': 3}, {'type': 'Live_In', 'head': 1, 'tail': 3}, {'type': 'Live_In', 'head': 2, 'tail': 3}], 'orig_id': 5284}


    # print(convert_custom_conll04_dict(data1))
    # print(convert_custom_conll04_dict(data2))
    print(convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_train.json", "/data/nlp/corpora/softrules/conll04/conll04_train_custom.jsonl"))
    print(convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_dev.json", "/data/nlp/corpora/softrules/conll04/conll04_dev_custom.jsonl"))
    print(convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_test.json", "/data/nlp/corpora/softrules/conll04/conll04_test_custom.jsonl"))
    