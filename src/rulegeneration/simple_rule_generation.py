from dataclasses import dataclass
from typing import Dict, Optional
from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_surface

@dataclass
class Rule:
    entity1: Optional[str]
    rule:    str
    entity2: Optional[str]

    def to_ast(self) -> AstNode:
        if self.entity1 and self.entity2:
            return parse_surface(f"[word={self.entity1}] {self.rule} [word={self.entity2}]")
        else:
            return parse_surface(self.rule)

    """
    Escape word if needed.
    The characters that are considered are: {', "}. Otherwise return the word as is
    """
    @staticmethod
    def escape_if_needed(word):
        # FIXME Maybe better ways to ensure proper escaping
        word_copy = word
        chars = [
            "'",
            '"',
        ]
        replaced = False
        for char in chars:
            if char in word:
                word_copy = word_copy.replace(char, f'\{char}')
                replaced = True
        if not replaced:
            word_copy = word_copy.encode("unicode_escape").decode("utf-8")
        # for char in ['\\']:
            # if char in word:
                # word_copy = word_copy.replace(char, f'\\\\')

        return word_copy

class WordRuleGenerator:

    """
    :param use_full_sentence -> whether to use the complete 
    """
    def __init__(self, use_entities: bool):
        self.use_entities = use_entities

    def word_rule(self, data: Dict) -> Rule:
        if data['e1_start'] > data['e2_start']:
            tokens = data['tokens'][data['e2_end']:data['e1_start']]
            tokens = ' '.join([f"""[word="{Rule.escape_if_needed(x)}"]""" for x in tokens])
            if self.use_entities:
                rule = Rule(data['e2_type'], tokens, data['e1_type'])# [data['e2_type']] + tokens + [data['e1_type']]
            else:
                rule = Rule(None, tokens, None)
        else:
            tokens = data['tokens'][data['e1_end']:data['e2_start']]
            tokens = ' '.join([f"""[word="{Rule.escape_if_needed(x)}"]""" for x in tokens])
            if self.use_entities:
                rule = Rule(data['e1_type'], tokens, data['e2_type'])# [data['e1_type']] + tokens + [data['e2_type']]
            else:
                rule = Rule(None, tokens, None)

        return rule




"""
Escape word if needed.
The characters that are considered are: {', "}. Otherwise return the word as is
"""
def escape_if_needed(word):
    # FIXME Maybe better ways to ensure proper escaping
    word_copy = word
    chars = [
        "'",
        '"',
    ]
    replaced = False
    for char in chars:
        if char in word:
            word_copy = word_copy.replace(char, f'\{char}')
            replaced = True
    if not replaced:
        word_copy = word_copy.encode("unicode_escape").decode("utf-8")
    # for char in ['\\']:
        # if char in word:
            # word_copy = word_copy.replace(char, f'\\\\')

"""
Create a rule using the words in-between the two entities
"""
def word_rule(data: Dict) -> AstNode:
    if data['e1_start'] > data['e2_start']:
        tokens = data['tokens'][data['e2_end']:data['e1_start']]
        return parse_surface(' '.join([f"""[word="{escape_if_needed(x)}"]""" for x in tokens]))
    else:
        tokens = data['tokens'][data['e1_end']:data['e2_start']]
        return parse_surface(' '.join([f"""[word="{escape_if_needed(x)}"]""" for x in tokens]))

if __name__ == "__main__":
    data1 = {'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], 'e1_start': 16, 'e1_end': 17, 'e2_start': 13, 'e2_end': 15, 'e1': ['TJQ'], 'e2': ['Tanjung', 'Pandan'], 'relation': 'P931'}
    data2 = {'tokens': ['The', 'system', 'as', 'described', 'above', 'has', 'its', 'greatest', 'application', 'in', 'an', 'arrayed', 'configuration', 'of', 'antenna', 'elements', '.'], 'e1_start': 12, 'e1_end': 13, 'e2_start': 15, 'e2_end': 16, 'e1': ['configuration'], 'e2': ['elements'], 'relation': 3}
    data3 = {'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '"', 'TJQ', ')', '.'], 'e1_start': 16, 'e1_end': 17, 'e2_start': 13, 'e2_end': 15, 'e1': ['TJQ'], 'e2': ['Tanjung', 'Pandan'], 'relation': 'P931'}
    data4 = {'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', "'", 'TJQ', ')', '.'], 'e1_start': 16, 'e1_end': 17, 'e2_start': 13, 'e2_end': 15, 'e1': ['TJQ'], 'e2': ['Tanjung', 'Pandan'], 'relation': 'P931'}
    print(word_rule(data1))
    print(word_rule(data2))
    print(word_rule(data3))
    print(word_rule(data4))
