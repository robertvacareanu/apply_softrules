"""
Adapted from run_qa.py
It handles situations wherethe model has to predict a START and an END,
and the data 
"""
from collections import defaultdict
import numpy as np 
from typing import Dict, List, Optional, Union
from torch import tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

def prepare_train_features_with_start_end(examples,
    tokenizer,
    first_sentence_column_name  = 'question',
    second_sentence_column_name = 'context',
    answer_column_name          = 'answers',
    pad_on_right                = True,
    max_seq_length              = 500,
    stride                      = 100,
):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[first_sentence_column_name] = [q.lstrip() for q in examples[first_sentence_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[first_sentence_column_name if pad_on_right else second_sentence_column_name],
        examples[second_sentence_column_name if pad_on_right else first_sentence_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    tokenized_examples['noisy_positions'] = np.zeros((len(tokenized_examples['input_ids']), len(tokenized_examples['input_ids'][0])))
    for i, (sp, se) in enumerate(zip(tokenized_examples['start_positions'], tokenized_examples['end_positions'])):
        if sp == 0 and se == 0:
            # print("NO ANSWER")
            pass
        else:
            tokenized_examples['noisy_positions'][i][sp:(se + 1)] = 1
    
    tokenized_examples['noisy_positions'] = tokenized_examples['noisy_positions'].astype(bool).tolist()
    
    # for te in tokenized_examples['input_ids']:
        # te['noisy'] = [0 for _ in range(te)]
        # te['noisy'][te['start_pos']]
    # exit()

    return tokenized_examples


def pair_tokenization(examples, tokenizer, max_seq_length=500):
    gr_gs = tokenizer(examples['good_rule'],   [' '.join(x) for x in examples['good_sentence']], max_length=max_seq_length, padding='max_length', truncation=True)
    rr_gs = tokenizer(examples['random_rule'], [' '.join(x) for x in examples['good_sentence']], max_length=max_seq_length, padding='max_length', truncation=True)
    gr_rs = tokenizer(examples['good_rule'],   [' '.join(x) for x in examples['random_sentence']], max_length=max_seq_length, padding='max_length', truncation=True)

    return {
        'input_ids_gr_gs'     : gr_gs['input_ids'],
        'token_type_ids_gr_gs': gr_gs['token_type_ids'],
        'attention_mask_gr_gs': gr_gs['attention_mask'],
        'input_ids_rr_gs'     : rr_gs['input_ids'],
        'token_type_ids_rr_gs': rr_gs['token_type_ids'],
        'attention_mask_rr_gs': rr_gs['attention_mask'],
        'input_ids_gr_rs'     : gr_rs['input_ids'],
        'token_type_ids_gr_rs': gr_rs['token_type_ids'],
        'attention_mask_gr_rs': gr_rs['attention_mask'],
    }

def pair_tokenization_sentence_baseline(examples, tokenizer, max_seq_length=500):
    gr_gs = tokenizer([' '.join(x) for x in examples['good_rule_sentence']],   [' '.join(x) for x in examples['good_sentence']],      max_length=max_seq_length, padding='max_length', truncation=True)
    gs_gr = tokenizer([' '.join(x) for x in examples['good_sentence']],        [' '.join(x) for x in examples['good_rule_sentence']], max_length=max_seq_length, padding='max_length', truncation=True)
    rr_gs = tokenizer([' '.join(x) for x in examples['random_rule_sentence']], [' '.join(x) for x in examples['good_sentence']],      max_length=max_seq_length, padding='max_length', truncation=True)
    gs_rs = tokenizer([' '.join(x) for x in examples['good_sentence']],        [' '.join(x) for x in examples['random_sentence']],    max_length=max_seq_length, padding='max_length', truncation=True)

    return {
        'input_ids_gr_gs'     : gr_gs['input_ids'],
        'token_type_ids_gr_gs': gr_gs['token_type_ids'],
        'attention_mask_gr_gs': gr_gs['attention_mask'],

        'input_ids_gs_gr'     : gs_gr['input_ids'],
        'token_type_ids_gs_gr': gs_gr['token_type_ids'],
        'attention_mask_gs_gr': gs_gr['attention_mask'],

        'input_ids_rr_gs'     : rr_gs['input_ids'],
        'token_type_ids_rr_gs': rr_gs['token_type_ids'],
        'attention_mask_rr_gs': rr_gs['attention_mask'],

        'input_ids_gs_rs'     : gs_rs['input_ids'],
        'token_type_ids_gs_rs': gs_rs['token_type_ids'],
        'attention_mask_gs_rs': gs_rs['attention_mask'],
    }

def tokenize_pair(tokenizer, s1, s2, s1_subj_pos, s1_obj_pos, s2_subj_pos, s2_obj_pos, max_seq_length=500):
    s1s2       = tokenizer(s1, s2, is_split_into_words=True, max_length=max_seq_length, padding='max_length', truncation=True, return_offsets_mapping=True, return_tensors='pt')
    s1s2_subj1 = highlighted_indices_tokenization_space([x+1 for x in s1_subj_pos], s1s2['offset_mapping'])
    s1s2_obj1  = highlighted_indices_tokenization_space([x+1 for x in s1_obj_pos],  s1s2['offset_mapping'])
    s1s2_subj2 = highlighted_indices_tokenization_space([x+2+len(s1) for x in s2_subj_pos], s1s2['offset_mapping'])
    s1s2_obj2  = highlighted_indices_tokenization_space([x+2+len(s1) for x in s2_obj_pos],  s1s2['offset_mapping'])
    return (s1s2, s1s2_subj1, s1s2_obj1, s1s2_subj2, s1s2_obj2)
    

def pair_tokenization_sentence_baseline_exp(e, tokenizer, max_seq_length=500):
    output = {}

    s1r1_s2r1, s1r1_s2r1_subj1, s1r1_s2r1_obj1, s1r1_s2r1_subj2, s1r1_s2r1_obj2 = tokenize_pair(tokenizer, e['s1r1'], e['s2r1'], e['s1r1_subj_pos'], e['s1r1_obj_pos'], e['s2r1_subj_pos'], e['s2r1_obj_pos'], max_seq_length=max_seq_length)
    s1r1_s3r1, s1r1_s3r1_subj1, s1r1_s3r1_obj1, s1r1_s3r1_subj2, s1r1_s3r1_obj2 = tokenize_pair(tokenizer, e['s1r1'], e['s3r1'], e['s1r1_subj_pos'], e['s1r1_obj_pos'], e['s3r1_subj_pos'], e['s3r1_obj_pos'], max_seq_length=max_seq_length)
    s1r1_s4r2, s1r1_s4r2_subj1, s1r1_s4r2_obj1, s1r1_s4r2_subj2, s1r1_s4r2_obj2 = tokenize_pair(tokenizer, e['s1r1'], e['s4r2'], e['s1r1_subj_pos'], e['s1r1_obj_pos'], e['s4r2_subj_pos'], e['s4r2_obj_pos'], max_seq_length=max_seq_length)

    # s1r1_s2r1['token_type_ids'][0][s1r1_s2r1_subj1] = 2
    # s1r1_s2r1['token_type_ids'][0][s1r1_s2r1_obj1]  = 3
    # s1r1_s2r1['token_type_ids'][0][s1r1_s2r1_subj2] = 4
    # s1r1_s2r1['token_type_ids'][0][s1r1_s2r1_obj2]  = 5

    # s1r1_s3r1['token_type_ids'][0][s1r1_s3r1_subj1] = 2
    # s1r1_s3r1['token_type_ids'][0][s1r1_s3r1_obj1]  = 3
    # s1r1_s3r1['token_type_ids'][0][s1r1_s3r1_subj2] = 4
    # s1r1_s3r1['token_type_ids'][0][s1r1_s3r1_obj2]  = 5

    # s1r1_s4r2['token_type_ids'][0][s1r1_s4r2_subj1] = 2
    # s1r1_s4r2['token_type_ids'][0][s1r1_s4r2_obj1]  = 3
    # s1r1_s4r2['token_type_ids'][0][s1r1_s4r2_subj2] = 4
    # s1r1_s4r2['token_type_ids'][0][s1r1_s4r2_obj2]  = 5

    output['input_ids_s1r1_s2r1'] = s1r1_s2r1['input_ids'][0]
    output['token_type_ids_s1r1_s2r1'] = s1r1_s2r1['token_type_ids'][0]
    output['attention_mask_s1r1_s2r1'] = s1r1_s2r1['attention_mask'][0]
    output['subj1_s1r1_s2r1'] = s1r1_s2r1_subj1
    output['subj2_s1r1_s2r1'] = s1r1_s2r1_subj2
    output['obj1_s1r1_s2r1'] = s1r1_s2r1_obj1
    output['obj2_s1r1_s2r1'] = s1r1_s2r1_obj2

    output['input_ids_s1r1_s3r1'] = s1r1_s3r1['input_ids'][0]
    output['token_type_ids_s1r1_s3r1'] = s1r1_s3r1['token_type_ids'][0]
    output['attention_mask_s1r1_s3r1'] = s1r1_s3r1['attention_mask'][0]
    output['subj1_s1r1_s3r1'] = s1r1_s3r1_subj1
    output['subj2_s1r1_s3r1'] = s1r1_s3r1_subj2
    output['obj1_s1r1_s3r1'] = s1r1_s3r1_obj1
    output['obj2_s1r1_s3r1'] = s1r1_s3r1_obj2

    output['input_ids_s1r1_s4r2'] = s1r1_s4r2['input_ids'][0]
    output['token_type_ids_s1r1_s4r2'] = s1r1_s4r2['token_type_ids'][0]
    output['attention_mask_s1r1_s4r2'] = s1r1_s4r2['attention_mask'][0]
    output['subj1_s1r1_s4r2'] = s1r1_s4r2_subj1
    output['subj2_s1r1_s4r2'] = s1r1_s4r2_subj2
    output['obj1_s1r1_s4r2'] = s1r1_s4r2_obj1
    output['obj2_s1r1_s4r2'] = s1r1_s4r2_obj2


    return output

def highlighted_indices_tokenization_space(words_of_interest: List[int], offset_mapping: tensor) -> List[int]:
    sentence_idx = torch.arange(offset_mapping.shape[1]).unsqueeze(dim=1)  # prepare an idx to make operation easier
    sentence_offsets_with_idx = torch.cat([sentence_idx, offset_mapping[0]], dim=1) # append idx
    word_start = sentence_offsets_with_idx[sentence_offsets_with_idx[:,1] == 0] # keep only the ones which represents the start of a new word
    last_word_start = word_start.shape[0]
    indices = []
    for woi in words_of_interest:
        if woi < last_word_start - 1:
            indices += range(word_start[woi][0], word_start[woi+1][0])
        elif woi == last_word_start - 1:
            indices += range(word_start[woi][0], sentence_offsets_with_idx.shape[0])
        else:
            raise ValueError("Outside sentence boundaries")
    return indices

# def pair_tokenization_sentence_baseline_with_entities(examples, tokenizer, max_seq_length=500):
#     for e in examples:
#         gr = tokenizer(e['good_rule_sentence'],   truncation=False, padding='do_not_pad', is_split_into_words=True, add_special_tokens=False, return_offsets_mapping=True)
#         gs = tokenizer(e['good_sentence'],        truncation=False, padding='do_not_pad', is_split_into_words=True, add_special_tokens=False, return_offsets_mapping=True)
#         rr = tokenizer(e['random_rule_sentence'], truncation=False, padding='do_not_pad', is_split_into_words=True, add_special_tokens=False, return_offsets_mapping=True)


#     gr    = tokenizer(examples['good_rule_sentence'], truncation=False, padding='do_not_pad', is_split_into_words=True, add_special_tokens=False, return_offsets_mapping=True)
#     gr_gs = tokenizer([' '.join(x) for x in examples['good_rule_sentence']],   [' '.join(x) for x in examples['good_sentence']],   max_length=max_seq_length, padding='max_length', truncation=True)
#     rr_gs = tokenizer([' '.join(x) for x in examples['random_rule_sentence']], [' '.join(x) for x in examples['good_sentence']],   max_length=max_seq_length, padding='max_length', truncation=True)
#     gr_rs = tokenizer([' '.join(x) for x in examples['good_rule_sentence']],   [' '.join(x) for x in examples['random_sentence']], max_length=max_seq_length, padding='max_length', truncation=True)

#     return {
#         'input_ids_gr_gs'     : gr_gs['input_ids'],
#         'token_type_ids_gr_gs': gr_gs['token_type_ids'],
#         'attention_mask_gr_gs': gr_gs['attention_mask'],
#         'input_ids_rr_gs'     : rr_gs['input_ids'],
#         'token_type_ids_rr_gs': rr_gs['token_type_ids'],
#         'attention_mask_rr_gs': rr_gs['attention_mask'],
#         'input_ids_gr_rs'     : gr_rs['input_ids'],
#         'token_type_ids_gr_rs': gr_rs['token_type_ids'],
#         'attention_mask_gr_rs': gr_rs['attention_mask'],
#     }


"""
alignment_strategy: START_END -> 
"""
def align_char_to_subtokens(input_tensor, start_offset, end_offset, alignment_strategy="START_END"):
    pass


from typing import Dict, List
import torch
import random
import numpy as np  
import json
from torch import tensor
import tqdm
from pathlib import Path

def init_random(seed):
    """
    Init torch, torch.cuda and numpy with same seed. For reproducibility.
    :param seed: Random number generator seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# from transformers import AutoModel, AutoTokenizer
# model     = AutoModel.from_pretrained('bert-base-cased')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# import datasets
# squad = datasets.load_dataset('json', data_files='/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/merged_train_split_train_expanded.jsonl', split="train[:32]")
# from torch.utils.data import DataLoader
# dataset = squad.map(lambda examples: prepare_train_features_with_start_end(examples, tokenizer), batched=True)
# # prepare_train_features(b1, tokenizer)

"""
    :param tokenizer: a tokenizer object (typically HuggingFace, or respecting its API/functionality)
    :param examples : Dict[List]
    :param text_column_name : which column from @see examples contains the text to tokenize. Importantly, it should be of type List[str]
    :param label_column_name: which column from @see examples contains the label for each word
    :param padding_strategy : a tokenizer object (typically HuggingFace, or respecting its API/functionality)
    :param max_seq_length   : the maximum length of the sequence; We need it to know when to truncate/pad
    :param label_all_tokens : by default, we assign the label for that word to its first sub token; If this flat
                              is set, then we will assign it to every subtoken resulting from that word

    Copy-Pasteable example:
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    >>> text_column_name  = 'tokens'
    >>> label_column_name = 'labels'
    >>> examples  = {
    >>>     text_column_name : [['this', 'is', 'a', 'test'], ['this', 'is', 'a', 'longer', 'testtesttest', 'with', 'subwords']],
    >>>     label_column_name: [[0,1,0,0], [0,0,0,0,1,0,1]]
    >>> }
    >>> tokenized = tokenize_words_and_align_labels(tokenizer, examples, text_column_name, label_column_name, max_seq_length=12)
    >>> list(zip([tokenizer.decode([x]) for x in tokenized['input_ids'][0]], tokenized[label_column_name][0]))
    [('[CLS]', -100), ('this', 0), ('is', 1), ('a', 0), ('test', 0), ('[SEP]', -100), ('[PAD]', -100), ('[PAD]', -100), ('[PAD]', -100), ('[PAD]', -100), ('[PAD]', -100), ('[PAD]', -100)]
    >>> list(zip([tokenizer.decode([x]) for x in tokenized['input_ids'][1]], tokenized[label_column_name][1]))
    [('[CLS]', -100), ('this', 0), ('is', 0), ('a', 0), ('longer', 0), ('test', 1), ('##test', -100), ('##test', -100), ('with', 0), ('sub', 1), ('##words', -100), ('[SEP]', -100)]
"""
def tokenize_words_and_align_labels(
    tokenizer        : Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    examples         : Dict[str, List],
    first_column_name : str,
    second_column_name: Optional[str],
    # text_column_name : str,
    label_column_name: str,
    label_for_sequence: int = 1,
    padding_strategy : str  = 'max_length',
    max_seq_length   : int  = 500,
    label_all_tokens : bool =  False,
    ):
    if not (isinstance(examples[first_column_name][0], List) and examples[first_column_name][0][0], str):
        raise ValueError(f"This works only when the text is already split into words. But the type is: {type(examples[first_column_name][0])}")
    tokenized_inputs = tokenizer(
        examples[first_column_name], 
        None if second_column_name is None else examples[second_column_name],
        padding=padding_strategy,
        truncation=True,
        max_length=max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for token_index, word_idx in enumerate(word_ids):
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None or tokenized_inputs.token_to_sequence(i, token_index) != label_for_sequence:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    first_column_name  = 'rule'
    second_column_name  = 'sentence'
    label_column_name = 'labels'
    examples  = {
        first_column_name : [["this is a rule"], ["this is another rule"]],
        second_column_name : [['this', 'is', 'a', 'test'], ['this', 'is', 'a', 'longer', 'testtesttest', 'with', 'subwords']],
        label_column_name: [[0,1,0,0], [0,0,0,0,1,0,1]]
    }
    tokenized = tokenize_words_and_align_labels(tokenizer, examples, first_column_name, second_column_name, label_column_name, max_seq_length=20)
    print(list(zip([tokenizer.decode([x]) for x in tokenized['input_ids'][0]], tokenized[label_column_name][0])))
    print(list(zip([tokenizer.decode([x]) for x in tokenized['input_ids'][1]], tokenized[label_column_name][1])))


