import json
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl as tacred_loader
from src.dataprocessing.fewrel.dataset_converter import load_dataset_from_jsonl as fewrel_loader
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl as semeval_loader
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl as tacred_fewshot_loader

def from_json_to_jsonl(from_path: str, to_path: str):
    with open(from_path) as fin:
        data = json.load(fin)
    with open(to_path, 'w+') as fout:
        for line in data:
            s = json.dumps(line)
            fout.write(s)
            fout.write('\n')

dataset_name_to_reader = {
    
    'tacred'        : tacred_loader,
    'fewrel'        : fewrel_loader,
    'semeval'       : semeval_loader,
    'tacred_fewshot': tacred_fewshot_loader,
}

def load_dataset_from_jsonl(path: str, dataset_name: str):
    if dataset_name not in dataset_name_to_reader:
        raise ValueError(f"The dataset name ({dataset_name}) is not find in the map, which contains the following keys: {list(dataset_name_to_reader.keys())}")
    return dataset_name_to_reader[dataset_name](path)

if __name__ == "__main__":
    # from_json_to_jsonl("data_sample/tacred/sample.json", 'data_sample/tacred/sample.jsonl')
    # from_json_to_jsonl("data_sample/fewrel/sample_unrolled.json", 'data_sample/fewrel/sample_unrolled.jsonl')
    from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/train/train.json", "/data/nlp/corpora/softrules/tacred/processed/train.jsonl")
    from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/dev/dev.json", "/data/nlp/corpora/softrules/tacred/processed/dev.jsonl")
    from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/test/test.json", "/data/nlp/corpora/softrules/tacred/processed/test.jsonl")