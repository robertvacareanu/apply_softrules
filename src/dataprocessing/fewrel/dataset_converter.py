import json
from typing import Dict

import datasets

def convert_fewrel_dict(fewrel_dict: Dict) -> Dict:

    tokens   = fewrel_dict['tokens']
    e1_start = fewrel_dict['h'][-1][0][0]
    e1_end   = fewrel_dict['h'][-1][0][-1] + 1
    e2_start = fewrel_dict['t'][-1][0][0]
    e2_end   = fewrel_dict['t'][-1][0][-1] + 1

    return {
        "tokens"  : tokens,
        "e1_start": e1_start,
        "e1_end"  : e1_end,
        "e2_start": e2_start,
        "e2_end"  : e2_end,
        "e1"      : tokens[e1_start:e1_end],
        "e2"      : tokens[e2_start:e2_end],
        'relation': fewrel_dict['relation'],
    }
    
def load_dataset_from_jsonl(path):
    d = datasets.load_dataset('text', data_files=path)
    d = d.map(lambda x: convert_fewrel_dict(json.loads(x['text'])), batched=False)

    return d

if __name__ == '__main__':
    data1 = {'relation': 'P931', 'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], 'h': ['tjq', 'Q1331049', [[16]]], 't': ['tanjung pandan', 'Q3056359', [[13, 14]]]}
    data2 = {'relation': 'P931', 'tokens': ['The', 'name', 'was', 'at', 'one', 'point', 'changed', 'to', 'Nottingham', 'East', 'Midlands', 'Airport', 'so', 'as', 'to', 'include', 'the', 'name', 'of', 'the', 'city', 'that', 'is', 'supposedly', 'most', 'internationally', 'recognisable', ',', 'mainly', 'due', 'to', 'the', 'Robin', 'Hood', 'legend', '.'], 'h': ['east midlands airport', 'Q8977', [[9, 10, 11]]], 't': ['nottingham', 'Q41262', [[8]]]}


    print(convert_fewrel_dict(data1))
    print(convert_fewrel_dict(data2))
    print(load_dataset_from_jsonl("data_sample/fewrel/sample_unrolled.jsonl"))