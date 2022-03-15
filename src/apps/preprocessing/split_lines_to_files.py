from src.config import Config
from src.dataprocessing.general_dataprocessing import load_dataset_from_jsonl



# python -m src.apps.preprocessing.split_lines_to_files --path config/base_config.yaml config/eval/word_average_baseline.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()##.get('word_average_eval')
    dataset = load_dataset_from_jsonl(config.get_path('dataset_path'), config.get('dataset_name'))['train']#.filter(lambda line: line['e1_start'] - line['e2_end'] != 0 and line['e2_start'] - line['e1_end'] != 0)
    print(dataset)