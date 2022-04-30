import json
from src.model.baseline.word_rule_expander import WordRuleExpander
from src.rulegeneration.simple_rule_generation import Rule
from src.config import Config
from odinson.ruleutils.queryparser import parse_surface
import tqdm 


def expand_rules(config: Config):
    wre           = WordRuleExpander(config.get("faiss_index").get_path('index_path'), config.get("faiss_index").get_path('vocab_path'), total_random_indices=config.get("faiss_index").get("total_random_indices"))
    rules         = set()

    with open(config.get_path('rules_path')) as fin:
        lines = fin.readlines()
        for line in tqdm.tqdm(lines):
            rule  = Rule.from_dict(json.loads(line))

            rule_length = 0 if rule.rule == '' else len(wre.extract_words(parse_surface(rule.rule)))
            if 0 < rule_length <= config.get('max_rule_length'):
                expansions = wre.rule_expander(rule=rule, similarity_threshold=config.get('similarity_threshold'), k=config.get('expansion_number'))
                for e in expansions:
                    rules.add(json.dumps(e.to_dict()))
            rules.add(json.dumps(rule.to_dict()))

    rules = list(rules)
    with open(config.get_path('rules_save_path'), 'w+') as fout:
        for rule in rules:
            fout.write(json.dumps(rule.to_dict()))
            fout.write('\n')




# python -m src.apps.rule_expander --path config/base_config.yaml config/expand_rules.yaml
if __name__ == "__main__":
    config = Config.parse_args_and_get_config()
    expand_rules(config)
    