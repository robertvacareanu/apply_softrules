import json
from src.dataprocessing.tacred.dataset_converter import load_dataset_from_jsonl as tacred_loader
from src.dataprocessing.fewrel.dataset_converter import load_dataset_from_jsonl as fewrel_loader
from src.dataprocessing.semeval.dataset_converter import load_dataset_from_jsonl as semeval_loader
from src.dataprocessing.conll04.dataset_converter import load_dataset_from_jsonl as conll04_loader
# from src.dataprocessing.tacred_fewshot.dataset_converter import load_dataset_from_jsonl as tacred_fewshot_loader

# Some datasets look like this: {"relation1": [<every item with relation = relation1>], <..>}
def from_reldict_to_list_of_json(from_path: str, to_path: str):
    with open(from_path) as fin:
        data = json.load(fin)

    output_data = []
    for (relation, data_list) in data.items():
        for data_point in data_list:
            output_data.append({'relation': relation, **data_point})

    with open(to_path, 'w+') as fout:
        json.dump(output_data, fout)

# Go from a list of dictionaries saved with json dump
# to a file where each line is a json object
def from_json_to_jsonl(from_path: str, to_path: str):
    with open(from_path) as fin:
        data = json.load(fin)
    with open(to_path, 'w+') as fout:
        for line in data:
            s = json.dumps(line)
            fout.write(s)
            fout.write('\n')


"""
For example, for a sentence like: "John visited Tucson", where "John" and "Tucson"
are the entities (given by every dataset), we replace it to:
- person visited location (if the entity types are given)
- entity visited entity   (if the entity types are not given)
"""
def replace_entity_words_with_type(line) -> str:
    tokens     = line['tokens']
    before_e1  = tokens[:min(line['e1_start'], line['e2_start'])]
    in_between = tokens[min(line['e1_end'], line['e2_end']):max(line['e1_start'], line['e2_start'])]
    after_e2   = tokens[max(line['e1_end'], line['e2_end']):]
    tokens     = before_e1 + [line['e1_type']] + in_between + [line['e2_type']] + after_e2

    return tokens

dataset_name_to_reader = {
    'tacred'        : tacred_loader,
    'fewrel'        : fewrel_loader,
    'semeval'       : semeval_loader,
    'conll04'       : conll04_loader,
    'tacred_fewshot': None,
}

def load_dataset_from_jsonl(path: str, dataset_name: str):
    if dataset_name not in dataset_name_to_reader:
        raise ValueError(f"The dataset name ({dataset_name}) is not find in the map, which contains the following keys: {list(dataset_name_to_reader.keys())}")
    return dataset_name_to_reader[dataset_name](path)

# python -m src.dataprocessing.general_dataprocessing
if __name__ == "__main__":
    print("main")
    # from_json_to_jsonl("data_sample/tacred/sample.json", 'data_sample/tacred/sample.jsonl')
    # from_json_to_jsonl("data_sample/fewrel/sample_unrolled.json", 'data_sample/fewrel/sample_unrolled.jsonl')
    # from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/train/train.json", "/data/nlp/corpora/softrules/tacred/processed/train.jsonl")
    # from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/dev/dev.json", "/data/nlp/corpora/softrules/tacred/processed/dev.jsonl")
    # from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/test/test.json", "/data/nlp/corpora/softrules/tacred/processed/test.jsonl")
    # from_reldict_to_list_of_json('/data/nlp/corpora/softrules/fewrel/train/train_wiki.json', '/data/nlp/corpora/softrules/fewrel/processed/train_wiki_processed.json')
    # from_json_to_jsonl('/data/nlp/corpora/softrules/fewrel/processed/train_wiki_processed.json', '/data/nlp/corpora/softrules/fewrel/processed/train_wiki_processed.jsonl')