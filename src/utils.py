from collections import Counter, defaultdict
import math

import sys
from typing import Dict, List

from src.config import Config

NO_RELATION = "no_relation"

def tacred_score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    if verbose:
        print( "Precision (micro): {:.2%}".format(prec_micro) ) 
        print( "   Recall (micro): {:.2%}".format(recall_micro) )
        print( "       F1 (micro): {:.2%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro


"""
The idea is to be able to open the index or multiple views of the same index (but multiplicated on disk)
The benefit is to be able to run things in parallel, but at the expense of having to duplicate the index
"""
def get_ees_from_config(gw, config: Config):
    if config.contains('odinson_index_multiplicated') and config.get('odinson_index_multiplicated'):
        basepath                 = config.get_path('odinson_index_basepath')
        index_muliplication_name = config.get('odinson_index_muliplication_name')
        start                    = config.get('odinson_index_start')
        end                      = config.get('odinson_index_end')
        # ees = [gw.open_index(f"{basepath}/{index_muliplication_name}{i}") for i in range(start, end)]
        ees_paths = [f"{basepath}/{index_muliplication_name}{i}" for i in range(start, end)]
    else:
        # ees = [gw.open_index(config.get_path('odinson_index_dir'))]
        ees_paths = [config.get_path('odinson_index_dir')]

    return ees_paths


def split_chunks(l, chunk_size):
    list_size = math.ceil(len(l)/chunk_size)
    for i in range(0, len(l), list_size):
        yield l[i:i + list_size]




def combine_result_dictionaries(dictionaries: List[Dict]) -> Dict:
    result = {}
    for d in dictionaries:
        for (doc_name, dict_rel_to_count) in d.items():
            if doc_name not in result:
                result[doc_name] = defaultdict(int)
            
            for (relation, count) in dict_rel_to_count.items():
                result[doc_name][relation] += count
        
    return result

"""
Takes as input a path to a file containing sentences separated by new line
Then, it creates a new file for each sentence and saves it in output_path
"""
def split_lines_to_files(input_file: str, output_path, file_basename):
    import tqdm
    data = []
    with open(input_file) as fin:
        for line in fin:
            data.append(line)
    
    for i, line in tqdm.tqdm(enumerate(data)):
        with open(f'{output_path}/{file_basename}_{i}.txt') as fout:
            fout.write(line)


# TODO alternative ways to store this
POS_TAGS = [
    'RBR', 
    'CC', 
    'SYM', 
    'WRB', 
    'VBZ', 
    'RBS', 
    'LS', 
    'VBG', 
    'NNP', 
    'PRP$', 
    'JJ', 
    'JJS', 
    'VBD', 
    '-LRB-', 
    'JJR', 
    'PRP', 
    'NN', 
    '#', 
    'WP$', 
    'VBP', 
    'RP', 
    'POS', 
    'IN', 
    'RB', 
    'CD', 
    'NNS', 
    'VBN', 
    'NNPS', 
    'MD', 
    'WDT', 
    'VB', 
    'DT', 
    '-RRB-', 
    'WP', 
    'FW', 
    'UH', 
    'TO', 
    'PDT', 
    'EX', 
]


