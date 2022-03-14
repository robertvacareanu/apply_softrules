#!/bin/bash

python -m src.apps.eval.word_average_eval --path config/base_config.yaml config/dataset_specifics/tacred_specifics.yaml config/eval/word_average_baseline.yaml --dataset-path softrules/tacred/processed/dev.jsonl --dataset-name tacred --rules-path softrules/tacred/processed/train_rules2