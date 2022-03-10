import json
from typing import List
import torch
import torch.nn as nn
import tqdm 

from src.model.baseline.word_embedding_baseline import WordEmbeddingAverager, get_word_embedding
from src.dataprocessing.general_dataprocessing import load_dataset_from_jsonl
from src.config import Config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.rulegeneration.simple_rule_generation import Rule

from gensim.models import KeyedVectors

def apply_rules_with_threshold(dataset, rules, no_relation_threshold = 0.8):
    rules_embeddings = torch.cat([x[1] for x in rules], dim=0)

    # Calculate embeddings
    predAndGold = []
    for (sentence_embedding, relation) in dataset:
        cosine_similarities = nn.functional.cosine_similarity(sentence_embedding, rules_embeddings)
        if cosine_similarities.max() < no_relation_threshold:
            predAndGold.append(("no_relation", relation))
        else:
            predAndGold.append((rules[cosine_similarities.argmax()][0], relation))    

    gold = [x[1] for x in predAndGold]
    pred = [x[0] for x in predAndGold]

    return (gold, pred)

def soft_apply_rules(dataset, rules, no_relation_threshold = 0.8):
    rules_embeddings = torch.cat([x[1] for x in rules], dim=0)
    from collections import defaultdict
    # Calculate embeddings
    gold = []
    pred = []

    # predAndGold = []
    for (sentence_embedding, relation) in tqdm.tqdm(dataset):
        cosine_similarities = nn.functional.cosine_similarity(sentence_embedding, rules_embeddings)
        gold.append(relation)

        prediction = defaultdict(list)

        # if cosine_similarities.max() < no_relation_threshold:
            # pred["no_relation"].append(1-cosine_similarities.max())
        # else:
        for i, cs in enumerate(cosine_similarities.cpu().detach().numpy().tolist()):
            prediction[rules[i][0]].append(cs)

        # gold = [x[1] for x in predAndGold]
        # pred = [x[0] for x in predAndGold]
        prediction = {k:sum(v)/len(v) for (k, v) in prediction.items()}
        prediction = sorted(list(prediction.items()), key=lambda x: -x[1])
        if len(prediction) > 0 and prediction[0][1] > no_relation_threshold:
            pred.append(prediction[0][0])
        else:
            pred.append("no_relation")

        # print(pred)
        # print(relation)
        # exit()
        
    return (gold, pred)
    

def word_averager(config: Config):


    glove_dict = {
        'fname'    : config.get('gensim_model').get_path('fname'),
        'binary'   : config.get('gensim_model').get('binary'),
        'no_header': config.get('gensim_model').get('no_header'),
    }

    tacred_dataset = load_dataset_from_jsonl(config.get_path('dataset_path'), config.get('dataset_name'))['train']#.filter(lambda line: line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0)
    tacred_rules   = []
    gensim_model   = KeyedVectors.load_word2vec_format(**glove_dict)
    wea            = WordEmbeddingAverager(gensim_model, aggregation_operator = lambda x: torch.max(x, dim=0)[0], skip_unknown_words=config.get('skip_unknown_words'))
    with open(config.get_path('rules_path')) as fin:
        lines = fin.readlines()
        for line in tqdm.tqdm(lines):
            rule = Rule.from_dict(json.loads(line))
            tacred_rules.append((rule.relation, wea.forward_rule(str(rule.to_ast()))))
    
    sentence_embeddings = []
    for line in tqdm.tqdm(tacred_dataset):
        tokens = line['tokens']
        if config.get('use_full_sentence'):
            before_e1  = tokens[:min(line['e1_start'], line['e2_start'])]
            in_between = tokens[min(line['e1_end'], line['e2_end']):max(line['e1_start'], line['e2_start'])]
            after_e2   = tokens[max(line['e1_end'], line['e2_end']):]
        else:
            left   = min(line['e1_start'], line['e1_start']) - config.get('number_of_words_left_right')
            left   = max(left, 0)
            right  = max(line['e1_end'], line['e2_end']) + config.get('number_of_words_left_right')
            right  = min(right, len(tokens))
            before_e1  = tokens[left:min(line['e1_start'], line['e2_start'])]
            in_between = tokens[min(line['e1_end'], line['e2_end']):max(line['e1_start'], line['e2_start'])]
            after_e2   = tokens[max(line['e1_end'], line['e2_end']):right]

        tokens = before_e1 + [line['e1_type']] + in_between + [line['e2_type']] + after_e2
        # print(tokens)
        # exit()
        sentence_embedding = wea.forward_sentence(tokens)#[min(line['e1_end'], line['e2_end']):max(line['e1_start'], line['e2_start'])])
        sentence_embeddings.append((sentence_embedding, line['relation']))

    for threshold in config.get('thresholds'):
        (gold, pred) = apply_rules_with_threshold(sentence_embeddings, tacred_rules, threshold)
        scores       = [accuracy_score(gold, pred), f1_score(gold, pred, average="micro"), precision_score(gold, pred, average="micro"), recall_score(gold, pred, average="micro"), f1_score(gold, pred, average="macro"), precision_score(gold, pred, average="macro"), recall_score(gold, pred, average="macro")]
        print('--------------')
        print(f"Threshold: {threshold}")
        print('accuracy: ', scores[0])
        print('---------------------------------------')
        print('f1: ', scores[1])
        print('precision: ', scores[2])
        print('recall: ', scores[3])
        print('---------------------------------------')
        print('f1: ', scores[4])
        print('precision: ', scores[5])
        print('recall: ', scores[6])
        print('--------------\n\n')

# python -m src.apps.eval.word_average_eval --path config/base_config.yaml config/eval/word_average_baseline.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()##.get('word_average_eval')
    word_averager(config)#config.get_path('dataset_path'), config.get_path('rules_path'), config.get('thresholds'), config.get('dataset_name'))
