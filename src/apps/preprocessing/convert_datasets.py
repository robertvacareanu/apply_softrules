from src.dataprocessing.general_dataprocessing import from_json_to_jsonl

# python -m src.apps.preprocessing.convert_datasets
def convert_datasets_to_jsonl_format():
    # from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/train/train.json", "/data/nlp/corpora/softrules/tacred/processed/train.jsonl")
    # from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/dev/dev.json", "/data/nlp/corpora/softrules/tacred/processed/dev.jsonl")
    # from_json_to_jsonl("/data/nlp/corpora/softrules/tacred/test/test.json", "/data/nlp/corpora/softrules/tacred/processed/test.jsonl")
    from_json_to_jsonl("/data/nlp/corpora/softrules/fewrel/train/train_wiki.json", "/data/nlp/corpora/softrules/fewrel/processed/train_wiki.jsonl")