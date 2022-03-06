import torch
import torch.nn as nn
import tqdm 

from src.model.baseline.word_embedding_baseline import get_word_embedding
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def word_averager(dataset, rules, no_relation_threshold = 0.8):
    wea              = get_word_embedding('glove-50d')
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

    print('accuracy: ', accuracy_score(gold, pred))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="micro"))
    print('precision: ', precision_score(gold, pred, average="micro"))
    print('recall: ', recall_score(gold, pred, average="micro"))
    print('---------------------------------------')
    print('f1: ', f1_score(gold, pred, average="macro"))
    print('precision: ', precision_score(gold, pred, average="macro"))
    print('recall: ', recall_score(gold, pred, average="macro"))


if __name__ == "__main__":
    tacred_dataset = load_dataset_from_jsonl("/data/nlp/corpora/softrules/tacred/processed/dev.jsonl")['train'].filter(lambda line: line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0)
    tacred_rules   = []
    wea            = get_word_embedding('glove-50d', aggregation_operator = lambda x: torch.max(x, dim=0)[0])
    with open("/data/nlp/corpora/softrules/tacred/processed/train_rules") as fin:
        for line in tqdm.tqdm(fin):
            split = line.split('\t')
            tacred_rules.append((split[1].strip(), wea.forward_rule(split[0].strip())))

    sentence_embeddings = []
    for line in tqdm.tqdm(tacred_dataset):
        sentence_embedding = wea.forward_sentence(line['tokens'][min(line['e1_end'], line['e2_end']):max(line['e1_start'], line['e2_start'])])
        sentence_embeddings.append((sentence_embedding, line['relation']))

    
    word_averager(sentence_embeddings, tacred_rules, 0.5)
    print("\n\n")
    word_averager(sentence_embeddings, tacred_rules, 0.6)
    print("\n\n")
    word_averager(sentence_embeddings, tacred_rules, 0.7)
    print("\n\n")
    word_averager(sentence_embeddings, tacred_rules, 0.8)
    print("\n\n")
    word_averager(sentence_embeddings, tacred_rules, 0.9)
    print("\n\n")
    word_averager(sentence_embeddings, tacred_rules, 0.95)
    print("\n\n")
    word_averager(sentence_embeddings, tacred_rules, 0.99)

