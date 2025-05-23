---
library_name: peft
license: other
base_model: huihui-ai/Llama-3.3-70B-Instruct-abliterated
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: llama70b
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama70b

This model is a fine-tuned version of [huihui-ai/Llama-3.3-70B-Instruct-abliterated](https://huggingface.co/huihui-ai/Llama-3.3-70B-Instruct-abliterated) on the alpaca_zh_demo dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- total_eval_batch_size: 64
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.15.1
- Transformers 4.51.3
- Pytorch 2.7.0+cu126
- Datasets 3.5.0
- Tokenizers 0.21.1