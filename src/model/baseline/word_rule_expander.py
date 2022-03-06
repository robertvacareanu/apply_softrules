from typing import List, Union

import faiss
import pickle
import random
import numpy as np

from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_surface

from collections import defaultdict


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
class WordRuleExpander:
    def __init__(self, faiss_index_path, vocab_path, **kwargs):
        self.faiss_index = faiss.read_index(faiss_index_path)

        with open(vocab_path, 'rb') as fin:
            self.index_to_key = pickle.load(fin)
        self.key_to_index = {v:k for (k,v) in enumerate(self.index_to_key)}

        # In case there are OOV words
        if 'average_vector' in kwargs:
            self.average_vector = kwargs['average_vector']
        else:
            random_indices = list(set([random.randint(0, self.faiss_index.ntotal) for i in range(kwargs.get('total_random_indices', 1000))]))
            self.average_vector = np.array([self.faiss_index.reconstruct(ri) for ri in random_indices]).mean(axis=0)

        self.expand_unknown_words = kwargs.get("expand_unknown_words", False)
        
    def rule_expander(self, rule: AstNode, similarity_threshold = 0.9, k=10) -> List[Union[AstNode, float]]:
        words = self.__extract_words(rule)
        expansions = defaultdict(list)
        vectors_to_search = []
        for w in words:
            if w in self.key_to_index:
                vector_to_search = self.faiss_index.reconstruct(self.key_to_index[w])
            else:
                vector_to_search = self.average_vector
            vectors_to_search.append(vector_to_search)
        vectors_to_search = np.array(vectors_to_search)

        (similarities, indices) = self.faiss_index.search(vectors_to_search, k + 1) # + 1 because it includes itself as well (if it finds it)
        cosines = 1 - similarities/2
        for i, c in enumerate(cosines):
            if words[i] in self.key_to_index:
                for j, similarity in enumerate(c):
                    if similarity > similarity_threshold:
                        expansions[words[i]].append((self.index_to_key[indices[i][j]], similarity))
            else:
                if self.expand_unknown_words:
                    for j, similarity in enumerate(c):
                        if similarity > similarity_threshold:
                            expansions[words[i]].append((self.index_to_key[indices[i][j]], similarity))
        
        rule_expansions = [(str(rule), 1.0)]
        for (word, expansion) in expansions.items():
            result = [(r[0].replace(f"word={word}", f"word={e[0]}"), r[1] * e[1]) for r in rule_expansions for e in expansion]
            rule_expansions += result
        rule_expansions = list(set(sorted(rule_expansions, key=lambda x: -x[1])))

        rule_str = str(rule)
        # print([re for re in rule_expansions if re != rule_str])
        return [re[0] for re in rule_expansions if re != rule_str][:k]

    def __extract_words(self, node: AstNode) -> List[str]:
        if type(node) == FieldConstraint:
            # If field constraint return the value
            return [node.value.string]
        else:
            # Flatten the result
            result = [self.__extract_words(x) for x in node.children()]
            return [y for x in result for y in x]



wre = WordRuleExpander("/data/nlp/corpora/softrules/faiss_index/glove.6B.50d_index", "/data/nlp/corpora/softrules/faiss_index/glove.6B.50d_vocab", total_random_indices=100000)
wre.rule_expander(parse_surface("[word=city] [word=of] [word=Tucson]"), 0.9)