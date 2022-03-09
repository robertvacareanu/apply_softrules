
from typing import Dict, List
import nltk
import numpy as np

"""
Converts the data from SemEval format into our internal format
"""

def convert_dict(data: Dict) -> Dict:
    sentence = data['sentence']
    e1_start = sentence.index("<e1>") + 4 # offset <e1>, since we do not include it
    e1_end   = sentence.index("</e1>")
    e2_start = sentence.index("<e2>") + 4 # offset <e2>, since we do not include it
    e2_end   = sentence.index("</e2>")

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

    return {
        "sentence"         : new_sentence,
        "original_sentence": sentence,
        "e1_start"         : e1_start,
        "e1_end"           : e1_end,
        "e2_start"         : e2_start,
        "e2_end"           : e2_end,
        "e1_str"           : e1_str,
        "e2_str"           : e2_str,
        'relation'         : data['relation'],
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
    }
    

if __name__ == "__main__":
    data1 = {'sentence': 'The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.', 'relation': 3}
    data2 = {'sentence': 'The system as described above has its greatest application in an <e2>arrayed configuration</e2> of <e1>antenna elements</e1>.', 'relation': 3}


    print(convert_semeval_dict(data1))
    print(convert_semeval_dict(data2))
