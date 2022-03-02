"""
    This baseline works only for surface rules containing only simple word constraints.
"""

from typing import Callable, List, Union
from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_surface
import torch
import torch.nn as nn

"""
    A simple baseline to rule and sentence embeddings: just use the constituent words
    and average their word embedding
"""
class WordEmbeddingAverager(nn.Module):
    def __init__(self, gensim_model, device = torch.device('cpu')):
        super().__init__()
        weights   = torch.FloatTensor(gensim_model.vectors)
        embedding = nn.Embedding.from_pretrained(torch.cat([weights, weights.mean(dim=0).unsqueeze(dim=0)], dim=0))
        embedding.requires_grad = False

        self.embedder   = embedding
        self.vocabulary = gensim_model.key_to_index
        self.unk_id     = weights.shape[0]
        self.device     = device

    def forward(self, batch):
        return 0

    """
        Embed one or a list of rules into a tensor
    """
    def forward_rule(self, rule: Union[List[AstNode], AstNode]):
        if isinstance(rule, AstNode):
            return self.__embed_rule(rule).unsqueeze(dim=0)
        elif isinstance(rule, List):
            return self.__embed_rules(rule)
        else:
            raise ValueError("Unrecognized Type")

    """
        Embed one or a list of sentences into a tensor
    """
    def forward_sentence(self, words: Union[List[List[str]], List[str]]):
        if len(words) == 0:
            raise ValueError("Empty sentence")

        if isinstance(words, List) and isinstance(words[0], str):
            return self.__embed_sentence(words).unsqueeze(dim=0)
        elif isinstance(words, List) and isinstance(words[0], List):
            return self.__embed_sentences(words)
        else:
            raise ValueError("Unrecognized Type")

    def __embed_rule(self, rule: AstNode):
        if not self.__check_simple_constraints_rule(rule):
            raise ValueError("This type of embedder works only with simple word constraints. No ORs, ANDs, or WILDCARDS. Only Concatenations")

        words = self.__extract_words(rule)
        return self.embedder(torch.tensor([self.vocabulary.get(word, self.unk_id) for word in words]).to(self.device)).mean(dim=0)
    
    def __embed_rules(self, rules: List[AstNode]):
        return torch.stack([self.__embed_rule(r) for r in rules])

    def __embed_sentence(self, words: List[str]):
        return self.embedder(torch.tensor([self.vocabulary.get(word, self.unk_id) for word in words]).to(self.device)).mean(dim=0)

    def __embed_sentences(self, sentences: List[List[str]]):
        return torch.stack([self.__embed_sentence(s) for s in sentences])

    """
        :param node -> the AstNode on which to apply the extraction
        :return     -> return a list of strings, corresponding to the constraints
    """
    def __extract_words(self, node: AstNode) -> List[str]:
        if type(node) == FieldConstraint:
            # If field constraint return the value
            return [node.value.string]
        else:
            # Flatten the result
            result = [self.__extract_words(x) for x in node.children()]
            return [y for x in result for y in x]


    def __check_simple_constraints_rule(self, node: AstNode) -> bool:
        node_type = type(node)
        if any([node_type == TokenSurface, node_type == ConcatSurface, node_type == ExactMatcher, node_type == FieldConstraint]) == False:
            return False
        else:
            return all([self.__check_simple_constraints_rule(x) for x in node.children()])


def get_word_embedding(what_type: str) -> WordEmbeddingAverager:
    from gensim.models import KeyedVectors

    # Map what_type name to parameters
    # Mapping is done this way because some files contain no header (otherwise just mapping to path would suffice)
    types = {
        'glove'    : {'fname': '/data/nlp/models/glove/glove.840B.300d.txt', 'binary': False, 'no_header': True}, #'glove-wiki-gigaword-300',
        'glove-50d': {'fname': '/data/nlp/models/glove/glove.6B.50d.txt',    'binary': False, 'no_header': True}, #'glove-wiki-gigaword-50',
    }

    if what_type not in types:
        raise ValueError("Only {'word2vec', 'fasttext', 'glove'} are supported.")

    model = KeyedVectors.load_word2vec_format(**types[what_type])

    return WordEmbeddingAverager(model)


m = get_word_embedding('glove-50d')
print(torch.nn.functional.cosine_similarity(m.forward_sentence("Bill Gates founded Microsoft".split(" ")), m.forward_rule(parse_surface("[word=founded]"))))
print(torch.nn.functional.cosine_similarity(m.forward_sentence("Bill Gates founded Microsoft".split(" ")), m.forward_rule(parse_surface("[word=created]"))))



