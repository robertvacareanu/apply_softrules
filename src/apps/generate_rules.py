from typing import Dict
from src.config import Config
import tqdm

from src.rulegeneration.simple_rule_generation import word_rule
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl


def generate_rules(config: Dict):
    dataset = load_dataset_from_jsonl(config.get_path('dataset_path'))

    with open(config.get_path('save_path'), 'w+') as fout:
        for line in tqdm.tqdm(dataset[config.get('split_name')]):
            # print(line)
            if line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0 and line['relation'] != 'no_relation':
                relation = line['relation']
                fout.write(f'{word_rule(line)}\t{relation}\n')
    
# python -m src.apps.generate_rules --path config/base_config.yaml config/generate_rules.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()
    generate_rules(config)
    # generate_rules("/data/nlp/corpora/softrules/tacred/processed/train.jsonl", "/data/nlp/corpora/softrules/tacred/processed/train_rules")
    # d = load_dataset_from_jsonl("/data/nlp/corpora/softrules/tacred/processed/train.jsonl").filter(lambda x: x['relation'] != 'no_relation')
    # print(d['train'][0])
    # print(d['train'][1])
    # print(d['train'][2])