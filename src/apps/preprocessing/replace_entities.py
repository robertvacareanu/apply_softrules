import json
from typing import Dict
from src.config import Config
import tqdm

from src.dataprocessing.general_dataprocessing import load_dataset_from_jsonl, replace_entity_words_with_type


def replace_entities_with_their_type(config: Config):
    dataset = load_dataset_from_jsonl(config.get_path('dataset_path'), config.get('dataset_name'))
    result = []
    for l in dataset[config.get('split_name')]:
        result.append(replace_entity_words_with_type(l))

    with open(config.get_path('save_path'), 'w+') as fout:
        for line in tqdm.tqdm(result):
            fout.write(' '.join(line))
            fout.write('\n')
    
"""
Take a file in the .jsonl format and replace the entities with their type
If their type is not present, replace them with "entity" (see our internal format in README.md)
"""
# python -m src.apps.preprocessing.replace_entities --path config/base_config.yaml config/replace_entities_config.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()
    replace_entities_with_their_type(config)
