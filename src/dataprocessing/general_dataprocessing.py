import json

def from_json_to_jsonl(from_path: str, to_path: str):
    with open(from_path) as fin:
        data = json.load(fin)
    with open(to_path, 'w+') as fout:
        for line in data:
            s = json.dumps(line)
            fout.write(s)
            fout.write('\n')

if __name__ == "__main__":
    from_json_to_jsonl("data_sample/tacred/sample.json", 'data_sample/tacred/sample.jsonl')
    from_json_to_jsonl("data_sample/fewrel/sample_unrolled.json", 'data_sample/fewrel/sample_unrolled.jsonl')
