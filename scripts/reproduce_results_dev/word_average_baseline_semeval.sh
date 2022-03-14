#!/bin/bash

python -m src.apps.eval.word_average_eval --path config/base_config.yaml config/dataset_specifics/semeval_specifics.yaml config/eval/word_average_baseline.yaml --dataset-path softrules/semeval/dev/dev.jsonl --dataset-name semeval --rules-path softrules/semeval/train_rules
