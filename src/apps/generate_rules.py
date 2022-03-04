import tqdm

from src.rulegeneration.simple_rule_generation import word_rule
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl


def generate_rules(jsonl_file: str, save_path: str, split_name = 'train'):
    dataset = load_dataset_from_jsonl(jsonl_file)

    with open(save_path, 'w+') as fout:
        for line in tqdm.tqdm(dataset[split_name]):
            # print(line)
            if line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0 and line['relation'] != 'no_relation':
                relation = line['relation']
                fout.write(f'{word_rule(line)}\t{relation}\n')
    

if __name__ == "__main__":
    # generate_rules("/data/nlp/corpora/softrules/tacred/processed/train.jsonl", "/data/nlp/corpora/softrules/tacred/processed/train_rules")
    d = load_dataset_from_jsonl("/data/nlp/corpora/softrules/tacred/processed/train.jsonl").filter(lambda x: x['relation'] != 'no_relation')
    print(d['train'][0])
    print(d['train'][1])
    print(d['train'][2])