# import pytorch_lightning as pl
import argparse
from turtle import forward
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datasets
import numpy as np
from pytorch_lightning import Trainer
from torch import nn
from transformers import BertModel, BertConfig, AdamW, AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict, namedtuple
from dataclasses import asdict, dataclass, make_dataclass
from src.model.util import tokenize_words_and_align_labels
from src.utils import tacred_score
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ProgressBar, GradientAccumulationScheduler, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from torch.functional import F

@dataclass
class TransformerBasedScorerOutput:
    """
    Output type of [`TransformerBasedScorer`].

    Args:
        tag_logits            (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Prediction scores of the model for start token (scores for each vocabulary token before SoftMax).
        match_prediction_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    tag_logits             : Optional[torch.FloatTensor] = None
    match_prediction_logits: Optional[torch.FloatTensor] = None
    cls_embedding          : Optional[torch.FloatTensor] = None



"""
Input:  (rule, sentence)
Output: 
"""
class TransformerBasedScorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model           = model

        self.tag_predictor   = nn.Linear(self.model.config.hidden_size, 2)
        
        self.match_predictor = nn.Linear(self.model.config.hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        return_tag_logits: bool = True,
        return_match_logits: bool = True,
        return_cls_embedding: bool = False,

    ) -> TransformerBasedScorerOutput:
        embeddings = self.model(input_ids, attention_mask, token_type_ids)

        cls_embedding = embeddings.last_hidden_state[:,0,:]

        output_tag_logits              = None
        output_match_prediction_logits = None
        output_cls_embedding           = None

        if return_tag_logits:
            output_tag_logits   = self.tag_predictor(embeddings.last_hidden_state)
        if return_match_logits:
            output_match_prediction_logits = self.match_predictor(cls_embedding)
        if return_cls_embedding:
            output_cls_embedding = cls_embedding
        
        return TransformerBasedScorerOutput(output_tag_logits, output_match_prediction_logits, output_cls_embedding)


class PLWrapper(pl.LightningModule):
    def __init__(self, hyperparameters) -> None:
        super().__init__()
        self.hyperparameters   = hyperparameters
        (model, tokenizer)     = get_bertlike_model_with_customs(hyperparameters.get('model_name'), hyperparameters.get('extra_tokens', []))
        self.model             = TransformerBasedScorer(model)
        self.tokenizer         = tokenizer
        self.sigmoid           = nn.Sigmoid()
        self.cel               = nn.CrossEntropyLoss()
        self.bce               = nn.BCEWithLogitsLoss()
        self.threshold         = hyperparameters.get('threshold')
        self.no_relation_label = hyperparameters.get('no_relation_label')
        self.loss_calculation_strategy_map = {
            'loss1': self.aggregate_loss_1,
            'loss2': self.aggregate_loss_2,
            'loss3': self.aggregate_loss_3,
            'loss4': self.aggregate_loss_4,
            'loss5': self.aggregate_loss_5,
        }
        self.loss_aggregator = self.loss_calculation_strategy_map[hyperparameters.get('loss_fn', 'loss1')]

        self.tag_loss_multiplier = hyperparameters.get('tag_loss_multiplier', 1)
        self.match_loss_multiplier = hyperparameters.get('match_loss_multiplier', 1)
        self.embedding_loss_multiplier = hyperparameters.get('embedding_loss_multiplier', 1)

        print(self.loss_aggregator)
        self.save_hyperparameters()

    def forward(self, batch) -> TransformerBasedScorerOutput:
        return self.model(**batch)
    
    def forward_rules_sentences(self, rules, sentences) -> TransformerBasedScorerOutput:
        tokenized = self.tokenizer(rules, sentences, return_tensors='pt', is_split_into_words=True, truncation=True, max_length=512, padding='longest')
        output = self.model.forward(**{k:v.to(self.device) for (k,v) in tokenized.items()})
        return output

    def predict_rule_sentence_pair(self, rule: Union[str, List[str]], sentence: Union[List[str], str]) -> Tuple[bool, List[str]]:
        if isinstance(rule, str):
            rule_input = [[rule]]
        elif isinstance(rule, List) and len(rule) == 1:
            rule_input = [rule]
        else:
            raise ValueError(f"Not supported type: {type(rule)}")
        if isinstance(sentence, str):
            sentence_input = [sentence.split(' ')]
        elif isinstance(sentence, List) and isinstance(sentence[0], str):
            sentence_input = [sentence]
        tokenized   = self.tokenizer(rule_input, sentence_input, return_tensors='pt', is_split_into_words=True, truncation=True, max_length=512, padding='longest')
        output      = self.model.forward(**{k:v.to(self.device) for (k, v) in tokenized.items()})
        matched     = (output.match_prediction_logits[0][0] > 0).item()
        tagged_word = self.tokenizer.decode([x[1] for x in zip(output.tag_logits[0][:,1].tolist(), tokenized['input_ids'][0].tolist()) if x[0] > 0])
        return (matched, tagged_word)

    def predict_episode(self, episode: Dict[str, Any]):
        rules     = [[x] for x in episode['rules']]
        sent      = [episode['test_sentence'] for r in rules]
        tokenized = self.tokenizer(rules, sent, return_tensors='pt', is_split_into_words=True, truncation=True, max_length=512, padding='longest')
        output      = self.model.forward(**{k:v.to(self.device) for (k, v) in tokenized.items()})
        matched     = F.sigmoid(output.match_prediction_logits)
        result = []
        for batch_idx, rule in enumerate(rules):
            tagged_words = self.tokenizer.decode([x[1] for x in zip(output.tag_logits[batch_idx][:,1].tolist(), tokenized['input_ids'][0].tolist()) if x[0] > 0])
            result.append([rule[0], matched[batch_idx].detach().cpu().item(), tagged_words])
        return result



    def calculate_loss_gr1_gs(self, batch, return_match_loss: bool = True, return_tag_loss: bool = True, return_cls_embedding: bool = False):
        output_gr1_gs = self.model.forward(input_ids=batch['input_ids_gr1_gs'], attention_mask=batch['attention_mask_gr1_gs'], token_type_ids=batch['token_type_ids_gr1_gs'], return_tag_logits=return_tag_loss, return_match_logits=return_match_loss, return_cls_embedding=return_cls_embedding)

        match_loss    = None
        tag_loss      = None
        
        if return_match_loss:
            match_loss   = self.bce(output_gr1_gs.match_prediction_logits.squeeze(1), torch.ones_like(output_gr1_gs.match_prediction_logits.squeeze(1)))
        if return_tag_loss:
            tag_loss     = self.cel(output_gr1_gs.tag_logits.view(-1, 2), batch['tags_gr1_gs'].view(-1))

        # output_gr1_gs.cls_embedding will be None if return_cls_embedding is False
        return (match_loss, tag_loss, output_gr1_gs.cls_embedding)

    def calculate_loss_gr2_gs(self, batch, return_match_loss: bool = True, return_tag_loss: bool = True, return_cls_embedding: bool = False):
        output_gr2_gs = self.model.forward(input_ids=batch['input_ids_gr2_gs'], attention_mask=batch['attention_mask_gr2_gs'], token_type_ids=batch['token_type_ids_gr2_gs'], return_tag_logits=return_tag_loss, return_match_logits=return_match_loss, return_cls_embedding=return_cls_embedding)

        match_loss    = None
        tag_loss      = None
        
        if return_match_loss:
            match_loss   = self.bce(output_gr2_gs.match_prediction_logits.squeeze(1), torch.ones_like(output_gr2_gs.match_prediction_logits.squeeze(1)))
        if return_tag_loss:
            tag_loss     = self.cel(output_gr2_gs.tag_logits.view(-1, 2), batch['tags_gr2_gs'].view(-1))

        # output_gr2_gs.cls_embedding will be None if return_cls_embedding is False
        return (match_loss, tag_loss, output_gr2_gs.cls_embedding)

    def calculate_loss_rr_gs(self, batch, return_match_loss: bool = True, return_tag_loss: bool = True, return_cls_embedding: bool = False):
        output_rr_gs = self.model.forward(input_ids=batch['input_ids_rr_gs'], attention_mask=batch['attention_mask_rr_gs'], token_type_ids=batch['token_type_ids_rr_gs'], return_tag_logits=return_tag_loss, return_match_logits=return_match_loss, return_cls_embedding=return_cls_embedding)

        match_loss    = None
        tag_loss      = None
        
        if return_match_loss:
            match_loss   = self.bce(output_rr_gs.match_prediction_logits.squeeze(1), torch.zeros_like(output_rr_gs.match_prediction_logits.squeeze(1)))
        if return_tag_loss:
            tag_loss     = self.cel(output_rr_gs.tag_logits.view(-1, 2), torch.zeros(output_rr_gs.tag_logits.shape[0] * output_rr_gs.tag_logits.shape[1], dtype=torch.long).to(self.device))

        # output_gr1_gs.cls_embedding will be None if return_cls_embedding is False
        return (match_loss, tag_loss, output_rr_gs.cls_embedding)

    def calculate_loss_rr_rs(self, batch, return_match_loss: bool = True, return_tag_loss: bool = True, return_cls_embedding: bool = False):
        output_rr_rs = self.model.forward(input_ids=batch['input_ids_rr_rs'], attention_mask=batch['attention_mask_rr_rs'], token_type_ids=batch['token_type_ids_rr_rs'], return_tag_logits=return_tag_loss, return_match_logits=return_match_loss, return_cls_embedding=return_cls_embedding)

        match_loss    = None
        tag_loss      = None
        
        if return_match_loss:
            match_loss   = self.bce(output_rr_rs.match_prediction_logits.squeeze(1), torch.zeros_like(output_rr_rs.match_prediction_logits.squeeze(1)))
        if return_tag_loss:
            tag_loss     = self.cel(output_rr_rs.tag_logits.view(-1, 2), torch.zeros(output_rr_rs.tag_logits.shape[0] * output_rr_rs.tag_logits.shape[1], dtype=torch.long).to(self.device))

        # output_gr1_gs.cls_embedding will be None if return_cls_embedding is False
        return (match_loss, tag_loss, output_rr_rs.cls_embedding)

    # Consider the loss only on match_loss and tag_loss for:
    # (i)  (good_rule, good_sentence)
    # (ii) (random_rule, good_sentence)
    def aggregate_loss_1(self, batch):
        gr1_gs = self.calculate_loss_gr1_gs(batch, True, True, False)
        rr_gs  = self.calculate_loss_rr_gs(batch,  True, True, False)

        match_loss = self.match_loss_multiplier * ((gr1_gs[0] + rr_gs[0])/2)
        tag_loss   = self.tag_loss_multiplier   * ((gr1_gs[1] + rr_gs[1])/2)

        self.log('match_loss', match_loss)
        self.log('tag_loss',   tag_loss)

        return (match_loss + tag_loss)/2

    # Consider the loss only on match_loss
    def aggregate_loss_2(self, batch):
        gr1_gs = self.calculate_loss_gr1_gs(batch, True, False, False)
        rr_gs  = self.calculate_loss_rr_gs(batch,  True, False, False)

        match_loss = self.match_loss_multiplier * ((gr1_gs[0] + rr_gs[0])/2)

        self.log('match_loss', match_loss)

        return match_loss
        
    # Consider the loss on match_loss and embedding loss
    def aggregate_loss_3(self, batch):
        gr1_gs = self.calculate_loss_gr1_gs(batch, True, False, True)
        gr2_gs = self.calculate_loss_gr2_gs(batch, True, False, True)
        rr_gs  = self.calculate_loss_rr_gs(batch,  True, False, True)

        match_loss         = self.match_loss_multiplier * ((gr1_gs[0] + gr2_gs[0] + rr_gs[0])/3)

        eps                = self.hyperparameters.get('triplet_loss', 1)
        embedding_loss_pos = F.pairwise_distance(gr1_gs[2], gr2_gs[2], p=2)
        embedding_loss_neg = F.pairwise_distance(gr1_gs[2], rr_gs[2],  p=2)
        embedding_loss     = self.embedding_loss_multiplier * (F.relu(embedding_loss_pos - embedding_loss_neg + eps).mean())

        self.log('match_loss',     match_loss)
        self.log('embedding_loss', embedding_loss)


        return (match_loss + embedding_loss)/2

    # Consider the loss on match_loss, tag_loss and embedding loss
    def aggregate_loss_4(self, batch):
        gr1_gs = self.calculate_loss_gr1_gs(batch, True, True, True)
        gr2_gs = self.calculate_loss_gr2_gs(batch, True, True, True)
        rr_gs  = self.calculate_loss_rr_gs(batch,  True, True, True)

        match_loss         = self.match_loss_multiplier * (gr1_gs[0] + gr2_gs[0] + rr_gs[0])/3
        tag_loss           = self.tag_loss_multiplier   * (gr1_gs[1] + gr2_gs[1] + rr_gs[1])/3

        eps                = self.hyperparameters.get('triplet_loss', 1)
        embedding_loss_pos = F.pairwise_distance(gr1_gs[2], gr2_gs[2], p=2)
        embedding_loss_neg = F.pairwise_distance(gr1_gs[2], rr_gs[2],  p=2)
        embedding_loss     = self.embedding_loss_multiplier * (F.relu(embedding_loss_pos - embedding_loss_neg + eps).mean())

        self.log('match_loss', match_loss)
        self.log('tag_loss', tag_loss)
        self.log('embedding_loss', embedding_loss)


        return (match_loss + tag_loss + embedding_loss)/3

    # Consider the loss on match_loss, tag_loss and embedding loss
    # but do not consider the tag_loss for the one where the prediction should be negative
    def aggregate_loss_5(self, batch):
        gr1_gs = self.calculate_loss_gr1_gs(batch, True, True, True)
        gr2_gs = self.calculate_loss_gr2_gs(batch, True, True, True)
        rr_gs  = self.calculate_loss_rr_gs(batch,  True, False, True)

        match_loss         = self.match_loss_multiplier * (gr1_gs[0] + gr2_gs[0] + rr_gs[0])/3
        tag_loss           = self.tag_loss_multiplier   * (gr1_gs[1] + gr2_gs[1])/2

        eps                = self.hyperparameters.get('triplet_loss', 1)
        embedding_loss_pos = F.pairwise_distance(gr1_gs[2], gr2_gs[2], p=2)
        embedding_loss_neg = F.pairwise_distance(gr1_gs[2], rr_gs[2],  p=2)
        embedding_loss     = self.embedding_loss_multiplier * (F.relu(embedding_loss_pos - embedding_loss_neg + eps).mean())
        
        self.log('match_loss', match_loss)
        self.log('tag_loss', tag_loss)
        self.log('embedding_loss', embedding_loss)


        return (match_loss + tag_loss + embedding_loss)/3

    def training_step(self, batch, batch_idx):
        loss = self.loss_aggregator(batch)

        self.log(f'train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hyperparameters['validation_style'] == 'training_style':
            return self.run_training_style_validation_step(batch, batch_idx)
        elif self.hyperparameters['validation_style'] == 'episode_style':
            return self.run_fewshot_episode_style_validation_step(batch, batch_idx)
        else:
            raise ValueError("Unknown validation style. It should be {'training_style', 'episode_style'}")

    def validation_epoch_end(self, outputs: List):
        if self.hyperparameters['validation_style'] == 'episode_style':
            complete_results = self.compute_results(outputs)#, thresholds=np.linspace(0.1, 1.0, 901).tolist())
            # import pickle
            # with open('5way1shot.pkl', 'wb') as fout:
                # pickle.dump(complete_results, fout, protocol=pickle.HIGHEST_PROTOCOL)
            # print(complete_results)
            # exit()
            for (threshold, (p, r, f1, f1_macro)) in complete_results.items():
                self.log(f'f1_{threshold}', f1, sync_dist=True)
                self.log(f'p_{threshold}',  p , sync_dist=True)
                self.log(f'r_{threshold}',  r , sync_dist=True)

            (p, r, f1, f1_macro) = complete_results[0.5]
            self.log(f'f1',        f1      , prog_bar=True, sync_dist=True)
            self.log(f'p',         p       , prog_bar=True, sync_dist=True)
            self.log(f'r',         r       , prog_bar=True, sync_dist=True)
            self.log(f'f1_macro',  f1_macro, prog_bar=True, sync_dist=True)
            self.log(f'best_f1',   max(complete_results.items(), key=lambda x: x[1][2])[1][2], sync_dist=True, prog_bar=True)
            self.log(f'best_thr',  max(complete_results.items(), key=lambda x: x[1][2])[0], sync_dist=True, prog_bar=True)

            return {'f1': f1 * 100, 'p': p * 100, 'r': r * 100}
        elif self.hyperparameters['validation_style'] == 'training_style':
            pred = [y for x in outputs for y in x['pred']]
            gold = [y for x in outputs for y in x['gold']]

            p, r, f1  = f1_score(gold, pred), precision_score(gold, pred), recall_score(gold, pred)
            
            self.log(f'f1', f1, prog_bar=True)
            self.log(f'p',  p , prog_bar=True)
            self.log(f'r',  r , prog_bar=True)

            return {'f1': f1, 'p': p, 'r': r}
        else:
            raise ValueError("Unknown validation style. It should be {'training_style', 'episode_style'}")

    def compute_results(self, outputs, thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999], verbose=False) -> Dict[float, List[float]]:
        relations      = [y for x in outputs for y in x['relations']]
        predictions    = [y for x in outputs for y in x['predictions']]
        gold_relations = [y for x in outputs for y in x['gold']]

        complete_results = {}

        for threshold in thresholds:
            pred = []
            gold = []
            for (i, gold_relation) in enumerate(gold_relations):
                rel_to_score = defaultdict(list)
                for (pred_score, rel) in zip(predictions[i], relations[i]):
                    rel_to_score[rel].append(pred_score)
                # rel_to_score = [(rel, np.mean(torch.topk(torch.tensor(scores), 3).values.tolist())) for (rel, scores) in rel_to_score.items()]
                rel_to_score = [(rel, np.mean(scores)) for (rel, scores) in rel_to_score.items()]
                max_score    = max(rel_to_score, key=lambda x: x[1])
                if max_score[1] > threshold:
                    pred.append(max_score[0])
                else:
                    pred.append('no_relation')
                gold.append(gold_relation)
            p, r, f1 = tacred_score(gold, pred, verbose=verbose)
            complete_results[threshold] = [p * 100, r * 100, f1 * 100, f1_score(gold, pred, average='macro', labels=list(set(gold).difference(["no_relation"]))) * 100]

        return complete_results

    def compute_pred_for_threshold(self, predictions, relations, threshold):
        pred = []
        for i in range(len(predictions)):
            rel_to_score = defaultdict(list)
            for (pred_score, rel) in zip(predictions[i], relations[i]):
                rel_to_score[rel].append(pred_score)
            # rel_to_score = [(rel, np.mean(torch.topk(torch.tensor(scores), 3).values.tolist())) for (rel, scores) in rel_to_score.items()]
            rel_to_score = [(rel, np.mean(scores)) for (rel, scores) in rel_to_score.items()]
            max_score    = max(rel_to_score, key=lambda x: x[1])
            if max_score[1] > threshold:
                pred.append(max_score[0])
            else:
                pred.append('no_relation')

        return pred

    def run_training_style_validation_step(self, batch, batch_idx):
        output      = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
        predictions = self.sigmoid(output.match_prediction_logits.squeeze(1)) > self.threshold

        return {
            'pred': predictions.detach().cpu().numpy().tolist(),
            'gold': batch['match'].detach().cpu().numpy().tolist(),
        }

    def run_fewshot_episode_style_validation_step(self, batch, batch_idx):
        rules     = []
        sentences = []
        lengths   = []
        relations = []
        for line in batch: 
            lengths.append(len(line['rules']))
            relations.append(line['rules_relations'])
            for (rule, relation) in zip(line['rules'], line['rules_relations']):
                if isinstance(rule, str):
                    rules.append(rule)
                else:
                    raise ValueError("Rule is a list")
                sentences.append(' '.join(line['test_sentence']))
        tokenized = self.tokenizer(rules, sentences, return_tensors='pt', padding='longest', truncation=True, max_length=512)
        output = self.forward({k:v.to(self.device) for (k,v) in tokenized.items()})
        output_split = self.sigmoid(output.match_prediction_logits).squeeze(1).split(lengths)
        gold = [b['gold_relation'] for b in batch]
        
        # We do not take max because we want to investigate multiple thresholds
        return {
            'gold': gold,
            'predictions': [x.detach().cpu().numpy().tolist() for x in list(output_split)],
            'relations': relations,
        }

    def configure_optimizers(self):
        lr             = self.hyperparameters.get('learning_rate', 3e-5)
        base_lr        = self.hyperparameters.get('base_lr', self.hyperparameters.get('learning_rate', 3e-5)/5)
        max_lr         = self.hyperparameters.get('max_lr', self.hyperparameters.get('learning_rate', 3e-5)*5)

        optimizer      = torch.optim.AdamW(self.parameters(), lr=lr)
        if not self.hyperparameters['use_scheduler']:
            return optimizer
        else:
            return (
                [optimizer],
                [{
                    # 'scheduler'        : get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hyperparameters.get('num_warmup_steps', 1000), num_training_steps=self.hyperparameters.get('num_training_steps', 10000)),
                    'scheduler'        : torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = base_lr, max_lr = max_lr, mode='triangular2', cycle_momentum=False, step_size_up=self.hyperparameters.get('step_size_up', 10000)),
                    'interval'         : 'step',
                    'frequency'        : 1,
                    'strict'           : True,
                    'reduce_on_plateau': False,
                }]
            )


def get_bertlike_model_with_customs(name: str, special_tokens: List[str]):
    model     = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if len(special_tokens) > 0:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        model.resize_token_embeddings(len(tokenizer)) 
    # model.embeddings.token_type_embeddings = torch.nn.modules.sparse.Embedding(4, model.config.hidden_size)
    # torch.nn.init.uniform_(model.embeddings.token_type_embeddings.weight, -0.01, 0.01)
    return (model, tokenizer)

def prepare_train(examples, tokenizer, max_seq_length=500, padding_strategy='max_length'):
    gr1_gs = tokenize_words_and_align_labels(tokenizer, examples, "good_rule1", "good_sentence", "good_sentence_tokens", label_for_sequence=1, padding_strategy=padding_strategy, max_seq_length=max_seq_length, label_all_tokens=False)
    gr2_gs = tokenize_words_and_align_labels(tokenizer, examples, "good_rule2", "good_sentence", "good_sentence_tokens", label_for_sequence=1, padding_strategy=padding_strategy, max_seq_length=max_seq_length, label_all_tokens=False)
    rr_gs  = tokenizer(
        examples['random_rule'], 
        examples['good_sentence'],
        padding=padding_strategy,
        truncation=True,
        max_length=max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    # rr_gs  = tokenizer(
    #     examples['random_rule'], 
    #     examples['good_sentence'],
    #     padding=padding_strategy,
    #     truncation=True,
    #     max_length=max_seq_length,
    #     # We use this argument because the texts in our dataset are lists of words (with a label for each word).
    #     is_split_into_words=True,
    # )
    return {
        'input_ids_gr1_gs'     : gr1_gs['input_ids'],
        'attention_mask_gr1_gs': gr1_gs['attention_mask'],
        'token_type_ids_gr1_gs': gr1_gs['token_type_ids'],
        'tags_gr1_gs'          : gr1_gs['labels'],

        'input_ids_gr2_gs'     : gr2_gs['input_ids'],
        'attention_mask_gr2_gs': gr2_gs['attention_mask'],
        'token_type_ids_gr2_gs': gr2_gs['token_type_ids'],
        'tags_gr2_gs'          : gr2_gs['labels'],
        
        'input_ids_rr_gs'     : rr_gs['input_ids'],
        'attention_mask_rr_gs': rr_gs['attention_mask'],
        'token_type_ids_rr_gs': rr_gs['token_type_ids'],
        
    }


def get_callbacks(accumulate_grad, params):
    callbacks = []

    pb = ProgressBar(refresh_rate=1)
    callbacks.append(pb)
    
    accumulator = GradientAccumulationScheduler(scheduling={0: accumulate_grad, 100: 4})
    callbacks.append(accumulator)
    
    cp = ModelCheckpoint(
        monitor    = 'best_f1',
        save_top_k = 7,
        mode       = 'max',
        save_last=True,
        filename='{epoch}-{step}-{best_f1:.3f}-{best_thr:.5f}-{f1:.3f}-{p:.3f}-{r:.3f}-{f1_macro:.3f}'
    )
    callbacks.append(cp)
    
    # cp = kwargs.get('split_dataset_training', {}).get('dataset_modelcheckpoint', base_cp)
    lrm = LearningRateMonitor(logging_interval='step')
    callbacks.append(lrm)
    
    es = EarlyStopping(
        monitor  = 'best_f1',
        patience = 3,
        mode     = 'max'
    )
    callbacks.append(es)
    if params['use_swa']:
        swa = StochasticWeightAveraging(swa_lrs=params['learning_rate'])
        callbacks.append(swa)
    
    return callbacks

def get_arg_parser():
    from distutils.util import strtobool

    extra_tokens_default  = ['misc', 'criminal_charge', 'cause_of_death', 'url', 'state_or_province']
    train_dataset_default = '/data/nlp/corpora/softrules/tacred_fewshot/train/hf_datasets/rules_sentences_pair/train_large.jsonl'
    eval_dataset1_default = '/data/nlp/corpora/softrules/tacred_fewshot/dev/hf_datasets/5_way_1_shots_10K_episodes_3q_seed_160290.jsonl'
    eval_dataset2_default = '/data/nlp/corpora/softrules/tacred_fewshot/dev/hf_datasets/5_way_5_shots_10K_episodes_3q_seed_160290.jsonl'

    parser = argparse.ArgumentParser(description='CLI Parameters for the (Rule, Sentence) scorer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate',             type=float,                         default=3e-5,                                help='learning rate')
    parser.add_argument('--model_name',                type=str,                           default='google/bert_uncased_L-4_H-256_A-4', help="model_name")
    parser.add_argument('--threshold',                 type=float,                         default=0.5,                                 help="extra_tokens")
    parser.add_argument('--extra_tokens', nargs='+',   type=str,                           default=extra_tokens_default,                help="threshold")
    parser.add_argument('--no_relation_label',         type=str,                           default='no_relation',                       help="no_relation_label")
    parser.add_argument('--num_warmup_steps_factor',   type=float,                         default=0.2,                                 help="num_warmup_steps")
    parser.add_argument('--step_size_up_factor',       type=float,                         default=0.5,                                 help="step_size_up")
    parser.add_argument('--use_scheduler',             type=lambda x: bool(strtobool(x)),  default=True,                                help="use_scheduler")
    parser.add_argument('--scheduler_type',            type=str,                           default='cycliclr',                          help="scheduler_type")
    parser.add_argument('--gradient_clip_val',         type=float,                         default=1.0,                                 help="gradient_clip_val")
    parser.add_argument('--gradient_clip_algorithm',   type=str,                           default="value",                             help="gradient_clip_algorithm")
    parser.add_argument('--description',               type=str,                           default=None,                                help="description of the model")
    parser.add_argument('--validation_style',          type=str,                           default='episode_style',                     help="validation_style")
    parser.add_argument('--loss_fn',                   type=str,                           default='loss1',                             help="loss_fn")
    parser.add_argument('--accumulate_grad_batches',   type=int,                           default=1,                                   help="accumulate_grad_batches")
    parser.add_argument('--train_batch_size',          type=int,                           default=32,                                  help="train_batch_size")
    parser.add_argument('--eval_batch_size',           type=int,                           default=4,                                   help="eval_batch_size")
    parser.add_argument('--max_epochs',                type=int,                           default=5,                                   help="max_epochs")
    parser.add_argument('--train_dataset',             type=str,                           default=train_dataset_default,               help="train_dataset")
    parser.add_argument('--eval_dataset1',             type=str,                           default=eval_dataset1_default,               help="eval_dataset1")
    parser.add_argument('--eval_dataset2',             type=str,                           default=eval_dataset2_default,               help="eval_dataset2")
    parser.add_argument('--tag_loss_multiplier',       type=float,                         default=1.0,                                 help="tag_loss_multiplier")
    parser.add_argument('--match_loss_multiplier',     type=float,                         default=1.0,                                 help="match_loss_multiplier")
    parser.add_argument('--embedding_loss_multiplier', type=float,                         default=1.0,                                 help="embedding_loss_multiplier")
    parser.add_argument('--log_save_name',             type=str,                           default='rule-sentence',                     help="log_save_name")
    parser.add_argument('--use_swa',                   type=lambda x: bool(strtobool(x)),  default=True,                                help="use StochasticWeightAveraging")

    return parser

if __name__ == '__main__':
    from src.model.util import init_random
    from src.model.util import prepare_train_features_with_start_end
    init_random(1)
    # exit()
    args = vars(get_arg_parser().parse_args())

    # pl_model = PLWrapper.load_from_checkpoint('/data/logs/rule-sentence-fm/version_3/checkpoints/epoch=2-step=3890-best_f1=18.763-best_thr=0.99000-f1=9.396-p=5.047-r=68.248-f1=f1_macro=15.106.ckpt')
    # pl_model.to(torch.device('cuda'))
    # pl_model.eval()
    # pl_model.half()
    # eval_dataset1  = datasets.load_dataset('json', data_files='/data/datasets/sr/5_way_1_shots_10K_episodes_3q_seed_160290.jsonl', split="train")
    # eval_dataset2  = datasets.load_dataset('json', data_files='/data/datasets/sr/5_way_5_shots_10K_episodes_3q_seed_160290.jsonl', split="train")
    # dl_eval11 = DataLoader(eval_dataset1, batch_size=8, collate_fn = lambda x: x, shuffle=False, num_workers=16)
    # dl_eval12 = DataLoader(eval_dataset2, batch_size=1, collate_fn = lambda x: x, shuffle=False, num_workers=16)
    # outputs1 = []
    # outputs2 = []
    # import tqdm
    # for batch in tqdm.tqdm(dl_eval11):
    #     outputs1.append(pl_model.validation_step(batch, None))
    # complete_results1 = pl_model.compute_results(outputs1, thresholds=np.linspace(0.5, 1.0, 501).tolist())
    # for batch in tqdm.tqdm(dl_eval12):
    #     outputs2.append(pl_model.validation_step(batch, None))
    # complete_results2 = pl_model.compute_results(outputs2, thresholds=np.linspace(0.5, 1.0, 501).tolist())
    # for batch in tqdm.tqdm(dl_eval12):
        # outputs2.append(pl_model.validation_step(batch, None))
    # exit()
    # dl_eval11 = DataLoader(eval_dataset1, batch_size=2, collate_fn = lambda x: x, shuffle=False, num_workers=16)
    # from collections import defaultdict
    # o = []
    # import tqdm
    # for batch in tqdm.tqdm(dl_eval11):
    #     o.append(pl_model.run_fewshot_episode_style_validation_step(batch, None, 0.8, True))

    # ed1 = eval_dataset1.filter(lambda x: x['gold_relation'] != 'no_relation')
    # ed = ed1[2]
    # for rule, rule_relation in zip(ed['rules'], ed['rules_relations']):
        # print('-------')
        # print(' '.join(ed['test_sentence']))
        # print(rule_relation, rule)
        # print(pl_model.predict_rule_sentence_pair(rule, ed['test_sentence']))
        # print(ed['gold_relation'])
        # print('-------')
        # print('\n\n')
    # exit()

    
    # (model, tokenizer) = get_bertlike_model_with_customs('google/bert_uncased_L-2_H-128_A-2', [])
    # (model, tokenizer) = get_bertlike_model_with_customs('bert-base-cased', [])
    # ntsb               = NoisyTransformerBasedScorer(model)
    # pl_model           = PLWrapper('bert-base-cased')
    # model_name              = 'google/bert_uncased_L-2_H-128_A-2'
    # model_name              = 'bert-base-cased'
    model_name              = args['model_name']#'google/bert_uncased_L-4_H-256_A-4'
    extra_tokens            = args['extra_tokens']#['misc', 'criminal_charge', 'cause_of_death', 'url', 'state_or_province']
    accumulate_grad_batches = args['accumulate_grad_batches']
    max_epochs              = args['max_epochs']

    (_, tokenizer) = get_bertlike_model_with_customs(model_name, extra_tokens)

    train_dataset = datasets.load_dataset('json', data_files=args['train_dataset']).map(lambda examples: prepare_train(examples, tokenizer, max_seq_length=300), batched=True)['train']
    train_dataset.set_format(type='torch', columns=[
        'input_ids_gr1_gs', 'attention_mask_gr1_gs', 'token_type_ids_gr1_gs', 'tags_gr1_gs',
        'input_ids_gr2_gs', 'attention_mask_gr2_gs', 'token_type_ids_gr2_gs', 'tags_gr2_gs',
        'input_ids_rr_gs',  'attention_mask_rr_gs',  'token_type_ids_rr_gs', 
    ])

    eval_dataset1  = datasets.load_dataset('json', data_files=args['eval_dataset1'], split="train")
    eval_dataset2  = datasets.load_dataset('json', data_files=args['eval_dataset2'], split="train")
    
    dl_train  = DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True, num_workers=32)
    dl_eval11 = DataLoader(eval_dataset1, batch_size=args['eval_batch_size'], collate_fn = lambda x: x, shuffle=False, num_workers=16)
    dl_eval12 = DataLoader(eval_dataset2, batch_size=1, collate_fn = lambda x: x, shuffle=False, num_workers=2)

    num_training_steps      = len(dl_train) / accumulate_grad_batches
    step_size_up            = int(num_training_steps * args['step_size_up_factor'])
    num_warmup_steps_factor = int(num_training_steps * args['num_warmup_steps_factor'])
    print(len(dl_train))
    print(len(dl_eval11))

    params = {
        **args,
        'num_training_steps'     : num_training_steps,
        'step_size_up'           : step_size_up,
        'num_warmup_steps_factor': num_warmup_steps_factor,
    }
    print(params)
    pl_model           = PLWrapper(params)
    # o                  = ntsb(**tokenizer("This is a test", return_tensors='pt'))
    logger = TensorBoardLogger('/data/logs/', name=args['log_save_name'])
    trainer_params = {
        'max_epochs'             : max_epochs,
        'accelerator'            : 'gpu',
        'devices'                : 1,
        'precision'              : 16,
        'num_sanity_val_steps'   : 50,
        'gradient_clip_val'      : 1.0,
        'gradient_clip_algorithm': "value",
        'logger'                 : logger,
        # 'check_val_every_n_epoch': 1,
        # 'val_check_interval'     : 1.0,
        # 'accumulate_grad_batches': 1,#accumulate_grad_batches,
        # 'log_every_n_steps'      : 1000,
    }

    trainer = Trainer(**trainer_params, callbacks = get_callbacks(accumulate_grad_batches, args),)
    # pl_model = PLWrapper.load_from_checkpoint('/home/rvacareanu/projects/temp/clean_repos/rules_softmatch/logs/span-prediction/version_29/checkpoints/epoch=0-step=7046-val_loss=0.000-f1=0.009-p=0.005-r=0.085.ckpt')
    # trainer.test(model=pl_model, dataloaders=dl_eval11)
    # trainer.test(model=pl_model, dataloaders=dl_eval12)
    trainer.fit(pl_model, train_dataloaders = dl_train, val_dataloaders = dl_eval11)
    # trainer.test(model=pl_model, dataloaders=dl_eval11)
    # trainer.test(model=pl_model, dataloaders=dl_eval12)
