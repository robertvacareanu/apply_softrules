from typing import List, Union


from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_surface


"""
Use FAISS index to return a list of similar words, then
keep only those above :param similarity_threshold

The indented work case is something like:
- given a rule such as: "[word=was] [word=founded] [word=by]
- return a list like ["[word=is] [word=founded] [word=by]", "[word=was] [word=created] [word=by]", "[word=is] [word=created] [word=by]", etc]

Be careful of drifting too far away from the initial rule. If similarity threshold is too low, the resulting rule might be too far away


:param rule: the rule to be expanded
:param similarity_threshold: used to filter out the words returned by similarity search.
:returns a list of tuples, where the first element is the new rule and the second element is the similarity score

The similarity score is computed by multiplying each similarity score
e.g.
similarity(was, is) = 0.95
similarity(founded, created) = 0.92
The similarity score of the rule "[word=is] [word=created] [word=by]" will be 0.95 * 0.92 = 0.874
"""
def rule_expander(rule: AstNode, similarity_threshold = 0.9) -> List[Union[AstNode, float]]:
    return []


class RuleExpander:
    def __init__(self, faiss_index):
        self.faiss_index = faiss_index
        
    def rule_expander(rule: AstNode, similarity_threshold = 0.9) -> List[Union[AstNode, float]]:
        return []

    