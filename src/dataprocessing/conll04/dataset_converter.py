import json
import hashlib
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
        for i, head in enumerate(line['entities']):
            for j, tail in enumerate(line['entities']):
                if i != j:
                    if head['start'] < tail['start']:
                        e1_start    = head['start']
                        e1_end      = head['end']
                        e2_start    = tail['start']
                        e2_end      = tail['end']
                        e1_type     = head['type']
                        e2_type     = tail['type']
                        e1_function = "head"
                        e2_function = "tail"
                    else:
                        e1_start    = tail['start']
                        e1_end      = tail['end']
                        e2_start    = head['start']
                        e2_end      = head['end']
                        e1_type     = tail['type']
                        e2_type     = head['type']
                        e1_function = "tail"
                        e2_function = "head"

                    id_field = ' '.join(line['tokens']) + str(e1_start) + str(e1_end) + str(e2_start) + str(e2_end)
                    cusotm_id = hashlib.md5(id_field.encode('utf-8')).hexdigest()

                    resulting_dict = {
                        "custom_id"  : cusotm_id,
                        "tokens"     : line['tokens'],
                        "e1_start"   : e1_start,
                        "e1_end"     : e1_end,
                        "e2_start"   : e2_start,
                        "e2_end"     : e2_end,
                        "e1"         : line['tokens'][e1_start:e1_end],
                        "e2"         : line['tokens'][e2_start:e2_end],
                        'relation'   : relations_in_line.get((i, j), "no_relation"),
                        'e1_type'    : e1_type,
                        'e2_type'    : e2_type,
                        'e1_function': e1_function,
                        'e2_function': e2_function,
                    }

                    output.append(resulting_dict)

    with open(save_path, 'w+') as fout:
        for line in output:
            fout.write(json.dumps(line))
            fout.write('\n')


def load_dataset_from_jsonl(path):
    d = datasets.load_dataset('text', data_files=path)
    d = d.map(lambda x: json.loads(x['text']), batched=False)

    return d

# python -m src.dataprocessing.conll04.dataset_converter
if __name__ == '__main__':
    # data1 = {'tokens': ['The', 'self-propelled', 'rig', 'Avco', '5', 'was', 'headed', 'to', 'shore', 'with', '14', 'people', 'aboard', 'early', 'Monday', 'when', 'it', 'capsized', 'about', '20', 'miles', 'off', 'the', 'Louisiana', 'coast', ',', 'near', 'Morgan', 'City', ',', 'Lifa', 'said.'], 'entities': [{'type': 'Other', 'start': 19, 'end': 21}, {'type': 'Loc', 'start': 23, 'end': 24}, {'type': 'Loc', 'start': 27, 'end': 29}, {'type': 'Peop', 'start': 30, 'end': 31}], 'relations': [{'type': 'Located_In', 'head': 2, 'tail': 1}], 'orig_id': 2447}
    # data2 = {'tokens': ['Annie', 'Oakley', ',', 'also', 'known', 'as', 'Little', 'Miss', 'Sure', 'Shot', ',', 'was', 'born', 'Phoebe', 'Ann', 'Moses', 'in', 'Willowdell', ',', 'Darke', 'County', ',', 'in', '1860', '.'], 'entities': [{'type': 'Peop', 'start': 0, 'end': 2}, {'type': 'Peop', 'start': 6, 'end': 10}, {'type': 'Peop', 'start': 13, 'end': 16}, {'type': 'Loc', 'start': 17, 'end': 21}], 'relations': [{'type': 'Live_In', 'head': 0, 'tail': 3}, {'type': 'Live_In', 'head': 1, 'tail': 3}, {'type': 'Live_In', 'head': 2, 'tail': 3}], 'orig_id': 5284}


    # print(convert_custom_conll04_dict(data1))
    # print(convert_custom_conll04_dict(data2))
    convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_train.json", "/data/nlp/corpora/softrules/conll04/conll04_train_custom.jsonl")
    convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_dev.json", "/data/nlp/corpora/softrules/conll04/conll04_dev_custom.jsonl")
    convert_to_custom_conll04_dict("/data/nlp/corpora/softrules/conll04/conll04_test.json", "/data/nlp/corpora/softrules/conll04/conll04_test_custom.jsonl")
    