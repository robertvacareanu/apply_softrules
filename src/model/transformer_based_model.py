# import pytorch_lightning as pl
from turtle import forward
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datasets
from pytorch_lightning import Trainer
from torch import nn
from transformers import BertModel, BertConfig, AdamW, AutoModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from typing import Dict, List, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import namedtuple
from dataclasses import asdict, dataclass, make_dataclass
from src.model.noise_layer import NoiseLayer
from src.utils import tacred_score
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ProgressBar, GradientAccumulationScheduler
from pytorch_lightning.loggers import TensorBoardLogger

@dataclass
class TransformerBasedScorerOutput:
    """
    Output type of [`TransformerBasedScorer`].

    Args:
        start_logits            (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Prediction scores of the model for start token (scores for each vocabulary token before SoftMax).
        end_logits              (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Prediction scores of the model for end token (scores for each vocabulary token before SoftMax).
        match_prediction_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    start_logits           : torch.FloatTensor = None
    end_logits             : torch.FloatTensor = None
    match_prediction_logits: torch.FloatTensor = None



"""
Input:  (rule, sentence)
Output: 
"""
class TransformerBasedScorer(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model           = model
        self.tokenizer       = tokenizer

        # Yes, they can be merged into nn.Linear(self.model.config.hidden_size, 2)
        self.start_predictor = nn.Linear(self.model.config.hidden_size, 1)
        self.end_predictor   = nn.Linear(self.model.config.hidden_size, 1)
        
        self.match_predictor = nn.Linear(self.model.config.hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ) -> TransformerBasedScorerOutput:
        embeddings = self.model(input_ids, attention_mask, token_type_ids)
        match_logits = self.match_predictor(embeddings.pooler_output)
        start_logits = self.start_predictor(embeddings.last_hidden_state)
        end_logits = self.end_predictor(embeddings.last_hidden_state)
        return TransformerBasedScorerOutput(start_logits, end_logits, match_logits)


"""
We want to add some noise, at some given positions
But Huggingface's transformers are handling everything between the tensor with the
token ids and the final hidden layers (i.e. mapping ids to embeddings, etc)
But the forward method allows to give `inputs_embeds`. But, for example, BERT
constructs the final embedding for token t1 as:
emb(t1) = token_emb(t1) + pos_emb(t1) + token_type_emb(t1)
If you pass `input_embeds`, then the embedding for token t1 is calcualted as:
emb(t1) = input_embeds + pos_emb(t1) + token_type_emb(t1)

Therefore, the point is:
    1. you need to call `model.get_input_embeddings()` to obtain the embedding layer for the tokens
    2. embed
    3. do the changes you want on those
    4. call forward with your `input_embeds`

"""
class NoisyTransformerBasedScorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model       = model

        self.noisy_layer = NoiseLayer()
        self.wte         = self.model.get_input_embeddings()

        # Yes, they can be merged into nn.Linear(self.model.config.hidden_size, 2)
        self.start_predictor = nn.Linear(self.model.config.hidden_size, 1)
        self.end_predictor   = nn.Linear(self.model.config.hidden_size, 1)

        self.match_predictor = nn.Linear(self.model.config.hidden_size, 1)

        self.cel = nn.BCEWithLogitsLoss()


    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        where_to_add_noise: Optional[torch.tensor] = None
    ) -> TransformerBasedScorerOutput:

        inputs_embeds      = self.wte(input_ids)
        noisy_input_embeds = self.noisy_layer(inputs_embeds, where_to_add_noise)
        # embeddings         = self.model(inputs_embeds=noisy_input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embeddings         = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        match_logits       = self.match_predictor(embeddings.pooler_output)
        start_logits       = self.start_predictor(embeddings.last_hidden_state)
        end_logits         = self.end_predictor(embeddings.last_hidden_state)
        
        return TransformerBasedScorerOutput(start_logits, end_logits, match_logits)

    # """
    # Similar to forward, just that it does not require the parameters to be tokenized
    # """
    # def forward_rules_sentences(self, rules: str, sentences: str) -> TransformerBasedScorerOutput:
    #     tokenized = tokenizer(rules, sentences, return_tensors='pt', padding=True)
    #     output = self.forward(**{k:v.to(self.device) for (k,v) in tokenized.items()})

    #     return output



class PLWrapper(pl.LightningModule):
    def __init__(self, model_name='google/bert_uncased_L-2_H-128_A-2', threshold = 0.5, no_relation_label = 'no_relation') -> None:
        super().__init__()
        (model, tokenizer)     = get_bertlike_model_with_customs(model_name, [])
        self.model             = NoisyTransformerBasedScorer(model)
        self.tokenizer         = tokenizer
        self.sigmoid           = nn.Sigmoid()
        self.cel               = nn.BCEWithLogitsLoss()
        self.threshold         = threshold
        self.no_relation_label = no_relation_label
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, token_type_ids, where_to_add) -> TransformerBasedScorerOutput:
        return self.model(input_ids, attention_mask, token_type_ids, where_to_add)
    
    def forward_rules_sentences(self, rules, sentences) -> TransformerBasedScorerOutput:
        tokenized = self.tokenizer(rules, sentences, return_tensors='pt', padding=True)
        output = self.forward(**{k:v.to(self.device) for (k,v) in tokenized.items()}, where_to_add=None)
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'], where_to_add=batch['noisy_positions'])
        start_positions = batch['start_positions'].to(self.device)
        end_positions   = batch['end_positions'].to(self.device)
        gold_match      = batch['match'].to(self.device).float()

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = output.start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index) # shape is: (batch_size,)
        end_positions = end_positions.clamp(0, ignored_index) # shape is: (batch_size,)

        # start_logits = output.start_logits.squeeze(-1) # shape is (batch_size, max_seq_length)
        # end_logits   = output.end_logits.squeeze(-1) # shape is (batch_size, max_seq_length)
        

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        # start_loss = loss_fct(start_logits, start_positions)
        # end_loss = loss_fct(end_logits, end_positions)
        match_loss = self.cel(output.match_prediction_logits.squeeze(1), gold_match)

        # loss = (start_loss + end_loss + match_loss)/3
        loss = match_loss


        self.log(f'train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output      = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'], where_to_add=None)
        predictions = self.sigmoid(output.match_prediction_logits.squeeze(1)) > self.threshold

        return {
            'pred': predictions.detach().cpu().numpy().tolist(),
            'gold': batch['match'].detach().cpu().numpy().tolist(),
        }

    def validation_epoch_end(self, outputs: List):
        pred = [y for x in outputs for y in x['pred']]
        gold = [y for x in outputs for y in x['gold']]
        p, r, f1  = f1_score(gold, pred), precision_score(gold, pred), recall_score(gold, pred)
        
        self.log(f'f1', f1, prog_bar=True)
        self.log(f'p',  p , prog_bar=True)
        self.log(f'r',  r , prog_bar=True)

        return {'f1': f1, 'p': p, 'r': r}

    def validation_step2(self, batch, batch_idx):
        # print(batch[0])
        # exit()
        rules     = []
        sentences = []
        lengths   = []
        relations = []
        for line in batch: 
            lengths.append(len(line['rules']))
            relations.append(line['rules_relations'])
            for (rule, relation) in zip(line['rules'], line['rules_relations']):
                rules.append(' '.join(rule))
                sentences.append(' '.join(line['test_sentence']))
        tokenized = self.tokenizer(rules, sentences, return_tensors='pt', padding=True)
        output = self.forward(**{k:v.to(self.device) for (k,v) in tokenized.items()}, where_to_add=None)

        output_split = self.sigmoid(output.match_prediction_logits).squeeze(1).split(lengths)

        pred = []
        # prediction is a tensor of shape <number_of_rules>
        # relation is a list of strings, each string being associated with a 
        # value in the prediction tensor (representing the "probability" of that
        # relation)
        for prediction, relation in zip(output_split, relations): 
            if prediction.max() > self.threshold:
                pred.append(relation[prediction.argmax()])
            else:
                pred.append(self.no_relation_label)        

        gold = [b['gold_relation'] for b in batch]
        # print(output.match_prediction_logits.squeeze(1))
        # print(self.sigmoid(output.match_prediction_logits).squeeze(1))
        # print(output_split)
        # print(pred)
        # print(gold)
        # print(output.start_logits[output.match_prediction_logits.argmax()].argmax())
        # print(output.end_logits[output.match_prediction_logits.argmax()].argmax())
        # a()
        # p, r, f1  = tacred_score(gold, pred)

        # self.log(f'f1', f1, prog_bar=True)
        # self.log(f'p',  p , prog_bar=True)
        # self.log(f'r',  r , prog_bar=True)

        return {
            'pred': pred,
            'gold': gold,
        }    

    def validation_epoch_end2(self, outputs: List):
        pred = [y for x in outputs for y in x['pred']]
        gold = [y for x in outputs for y in x['gold']]
        p, r, f1  = tacred_score(gold, pred)
        
        self.log(f'f1', f1, prog_bar=True)
        self.log(f'p',  p , prog_bar=True)
        self.log(f'r',  r , prog_bar=True)

        return {'f1': f1, 'p': p, 'r': r}

    def configure_optimizers(self):
        from transformers import get_linear_schedule_with_warmup

        lr             = 3e-4
        base_lr        = 3e-6
        max_lr         = 3e-4
        mode           = 'triangular2'
        cycle_momentum = False
        step_size_up   = 2500
        optimizer      = AdamW(self.parameters(), lr=3e-5)
        # return optimizer
        return (
            [optimizer],
            [{
                'scheduler'        : get_linear_schedule_with_warmup(optimizer, 500, 2500),
                'interval'         : 'step',
                'frequency'        : 1,
                'strict'           : True,
                'reduce_on_plateau': False,
            }]
        )


def get_bertlike_model_with_customs(name: str, special_tokens: List[str]):
    model     = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer)) 
    model.embeddings.token_type_embeddings = torch.nn.modules.sparse.Embedding(4, model.config.hidden_size)
    torch.nn.init.uniform_(model.embeddings.token_type_embeddings.weight, -0.01, 0.01)
    return (model, tokenizer)




if __name__ == '__main__':
    from src.model.util import init_random
    from src.model.util import prepare_train_features_with_start_end
    init_random(1)
    # (model, tokenizer) = get_bertlike_model_with_customs('google/bert_uncased_L-2_H-128_A-2', [])
    # (model, tokenizer) = get_bertlike_model_with_customs('bert-base-cased', [])
    # ntsb               = NoisyTransformerBasedScorer(model)
    # pl_model           = PLWrapper('bert-base-cased')
    pl_model           = PLWrapper('google/bert_uncased_L-2_H-128_A-2')
    # o                  = ntsb(**tokenizer("This is a test", return_tensors='pt'))
    accumulate_grad_batches = 8
    logger = TensorBoardLogger('logs', name='span-prediction')
    pb = ProgressBar(refresh_rate=1)
    accumulator = GradientAccumulationScheduler(scheduling={0: 8, 100: 4})
    cp = ModelCheckpoint(
        monitor    = 'f1',
        save_top_k = 7,
        mode       = 'max',
        save_last=True,
        filename='{epoch}-{step}-{val_loss:.3f}-{f1:.3f}-{p:.3f}-{r:.3f}'
    )
    # cp = kwargs.get('split_dataset_training', {}).get('dataset_modelcheckpoint', base_cp)
    lrm = LearningRateMonitor(logging_interval='step')

    es = EarlyStopping(
        monitor  = 'f1',
        patience = 3,
        mode     = 'max'
    )
    trainer_params = {
        'max_epochs'             : 5,
        'accelerator'            : 'gpu',
        'devices'                : 1,
        'precision'              : 16,
        'num_sanity_val_steps'   : 1000,
        'gradient_clip_val'      : 1,
        'logger'                 : logger,
        'check_val_every_n_epoch': 1,
        # 'accumulate_grad_batches': 1,#accumulate_grad_batches,
        'log_every_n_steps'      : 1000,
    }
    trainer = Trainer(**trainer_params, callbacks = [lrm, cp, es, pb, accumulator,],)

    # train_dataset = datasets.load_dataset('json', data_files='/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/merged_train_split_train.jsonl', split="train").map(lambda examples: prepare_train_features_with_start_end(examples, tokenizer), batched=True)
    # train_dataset = datasets.load_dataset('json', data_files='/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/merged_train_split_train_expanded.jsonl', split="train[:10000]").map(lambda examples: prepare_train_features_with_start_end(examples, tokenizer), batched=True)
    # train_dataset = datasets.load_dataset('json', data_files='/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/merged_train_split_train.jsonl', split="train").filter(lambda x: len(x['context'].split(' ')) < 200).map(lambda examples: prepare_train_features_with_start_end(examples, tokenizer), batched=True)
    # train_dataset.save_to_disk('/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/arrow/merged_train_split_train_bert_uncased_L-2_H-128_A-2')
    # train_dataset = datasets.load_from_disk('/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/arrow/merged_train_split_train_bertbaseuncased')
    train_dataset = datasets.load_from_disk('/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/arrow/merged_train_split_train_bert_uncased_L-2_H-128_A-2')#.select(range(1000))
    dataset = datasets.load_from_disk('/data/nlp/corpora/odinsynth2/pretraining/random_rules_extractions/arrow/merged_train_split_train_bert_uncased_L-2_H-128_A-2').select(range(500000))
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'noisy_positions', 'match'])
    dataset = dataset.train_test_split(0.2)
    # exit()
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'noisy_positions', 'match'])
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'noisy_positions', 'match'])
    eval_dataset  = datasets.load_dataset('json', data_files='/data/nlp/corpora/softrules/tacred_fewshot/dev/hf_datasets/5_way_1_shots_10K_episodes_3q_seed_160290.jsonl', split="train")
    
    dl_train = DataLoader(dataset['train'], batch_size=32, shuffle=True, num_workers=32)
    num_training_steps = len(dl_train) / accumulate_grad_batches
    # print(len(dl_train))
    print(num_training_steps)
    # exit()
    dl_eval  = DataLoader(dataset['test'], batch_size=32, num_workers=32)#ollate_fn = lambda x: x, shuffle=False, num_workers=32)
    dl_eval2  = DataLoader(eval_dataset, batch_size=32, num_workers=32)#ollate_fn = lambda x: x, shuffle=False, num_workers=32)
    # exit()
    # pl_model = PLWrapper.load_from_checkpoint('/home/rvacareanu/projects/temp/clean_repos/rules_softmatch/logs/span-prediction/version_29/checkpoints/epoch=0-step=7046-val_loss=0.000-f1=0.009-p=0.005-r=0.085.ckpt')
    trainer.validate(model=pl_model, dataloaders=dl_eval)
    trainer.fit(pl_model, train_dataloaders = dl_train, val_dataloaders = dl_eval)
    trainer.validate(model=pl_model, dataloaders=dl_eval)


