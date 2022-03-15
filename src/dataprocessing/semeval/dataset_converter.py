
import json
import hashlib
from typing import Dict, List
import nltk
import numpy as np
import datasets

"""
Converts the data from SemEval format into our internal format
"""

def convert_dict(data: Dict) -> Dict:
    sentence = data['sentence']
    e1_start = sentence.index("<e1>") + 4 # offset <e1>, since we do not include it
    e1_end   = sentence.index("</e1>")
    e2_start = sentence.index("<e2>") + 4 # offset <e2>, since we do not include it
    e2_end   = sentence.index("</e2>")

    if (e1_start < e2_start < e2_end < e1_end) or (e2_start < e1_start < e1_end < e2_end) or (e1_start < e2_start < e1_end < e2_end) or (e2_start < e1_start < e2_end < e1_end):
        raise ValueError(f"Overlapping entities in {data}")

    # e1_start <..> e1_end <..> e2_start <..> e2_end
    if e1_start < e2_start:
        e1_start -= 4         # offset <e1>
        e1_end   -= 4         # offset <e1>
        e2_start -= 4 + 5 + 4 # offset <e1> </e1> <e2>
        e2_end   -= 4 + 5 + 4 # offset <e1> </e1> <e2>
    # e2_start <..> e2_end <..> e1_start <..> e1_end
    elif e2_start < e1_start:
        e2_end   -= 4         # offset <e2>
        e2_end   -= 4         # offset <e2>
        e1_start -= 4 + 5 + 4 # offset <e2> </e2> <e1>
        e1_end   -= 4 + 5 + 4 # offset <e2> </e2> <e1>
    else:
        raise ValueError("Entities should not be overlapped")

    new_sentence = sentence.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")      
    e1_str   = new_sentence[e1_start:e1_end]
    e2_str   = new_sentence[e2_start:e2_end]

    id_field  = ' '.join(new_sentence) + str(e1_start) + str(e1_end) + str(e2_start) + str(e2_end)
    custom_id = hashlib.md5(id_field.encode('utf-8')).hexdigest()


    # We know that SemEval specifies them in order
    return {
        "custom_id"        : custom_id,
        "sentence"         : new_sentence,
        "original_sentence": sentence,
        "e1_start"         : e1_start,
        "e1_end"           : e1_end,
        "e2_start"         : e2_start,
        "e2_end"           : e2_end,
        "e1"               : e1_str,
        "e2"               : e2_str,
        'relation'         : data['relation'],
        'e1_type'          : 'entity',  # Not provided by the SemEval dataset
        'e2_type'          : 'entity',  # Not provided by the SemEval dataset
        'e1_function'      : 'unknown', # Not provided by the SemEval dataset
        'e2_function'      : 'unknown', # Not provided by the SemEval dataset

    }


def convert_semeval_dict(semeval_dict: Dict) -> Dict: 
    converted_dict = convert_dict(semeval_dict)
    tokens         = nltk.word_tokenize(converted_dict['sentence'])
    lengths       = np.array([len(x) for x in tokens]).cumsum() + np.arange(len(tokens))
    e1_start      = len(lengths[lengths < converted_dict['e1_start']])
    e1_end        = len(lengths[lengths < converted_dict['e1_end']]) + 1
    e2_start      = len(lengths[lengths < converted_dict['e2_start']])
    e2_end        = len(lengths[lengths < converted_dict['e2_end']]) + 1

    return {
        "tokens"  : tokens,
        "e1_start": e1_start,
        "e1_end"  : e1_end,
        "e2_start": e2_start,
        "e2_end"  : e2_end,
        "e1"      : tokens[e1_start:e1_end],
        "e2"      : tokens[e2_start:e2_end],
        'relation': semeval_dict['relation'],
        'e1_type' : 'entity',
        'e2_type' : 'entity',
        'e1_function': 'unknown',
        'e2_function': 'unknown',

    }

def load_dataset_from_jsonl(path):
    d = datasets.load_dataset('text', data_files=path)
    d = d.map(lambda x: convert_semeval_dict(json.loads(x['text'])), batched=False)

    return d
    

if __name__ == "__main__":
    data1 = {'sentence': 'The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.', 'relation': 3}
    data2 = {'sentence': 'The system as described above has its greatest application in an <e2>arrayed configuration</e2> of <e1>antenna elements</e1>.', 'relation': 3}


    print(convert_semeval_dict(data1))
    print(convert_semeval_dict(data2))

    # import datasets
    # d  = datasets.load_dataset("sem_eval_2010_task_8")
    # tv = d['train'].train_test_split(test_size=0.1, seed=1)
    # import json
    # with open('/data/nlp/corpora/softrules/semeval/train/train.jsonl', 'w+') as fout:
    #     for line in tv['train']:
    #         fout.write(json.dumps(line))
    #         fout.write('\n')
    # with open('/data/nlp/corpora/softrules/semeval/dev/dev.jsonl', 'w+') as fout:
    #     for line in tv['test']:
    #         fout.write(json.dumps(line))
    #         fout.write('\n')
    # with open('/data/nlp/corpora/softrules/semeval/test/test.jsonl', 'w+') as fout:
    #     for line in d['test']:
    #         fout.write(json.dumps(line))
    #         fout.write('\n')
    # print(d)
    # print(tv)