from typing import Dict
from src.config import Config
import tqdm

from src.rulegeneration.simple_rule_generation import WordRuleGenerator, word_rule
from src.dataprocessing.general_dataprocessing import load_dataset_from_jsonl


def generate_rules(config: Config):
    wrg     = WordRuleGenerator(use_entities=config.get('use_entities'))
    dataset = load_dataset_from_jsonl(config.get_path('dataset_path'), config.get('dataset_name'))
    d = []
    for l in dataset[config.get('split_name')]:
        d.append(l)

    with open(config.get_path('save_path'), 'w+') as fout:
        for line in tqdm.tqdm(d[1805:]):
            # print(line)
            if line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0 and line['relation'] != 'no_relation':
                relation = line['relation']
                fout.write(f'{wrg.word_rule(line).to_ast()}\t{relation}\n')
    
# python -m src.apps.generate_rules --path config/base_config.yaml config/generate_rules.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()
    generate_rules(config)
    # generate_rules("/data/nlp/corpora/softrules/tacred/processed/train.jsonl", "/data/nlp/corpora/softrules/tacred/processed/train_rules")
    # d = load_dataset_from_jsonl("/data/nlp/corpora/softrules/tacred/processed/train.jsonl").filter(lambda x: x['relation'] != 'no_relation')
    # print(d['train'][0])
    # print(d['train'][1])
    # print(d['train'][2])