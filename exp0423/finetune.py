import numpy as np
import os
import sys
import json
import copy
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.opt_utils import load_model_and_tokenizer
from utils.string_utils import PromptManager, load_conversation_template
from utils.generate import generate
from utils.model import GPT

def get_args():
    parser = argparse.ArgumentParser(description="Finetune manager.")
    # Experiment Settings
    parser.add_argument("--model_name", type=str, default="llama")

    # Finetune (Generation) Parameters
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)

    # Finetune (LoRa) Parameters
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    # parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--learning_rate", type=float, default=5e-5) 
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_seq_length", type=int, default=2048)
   
    # System Settings
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--FP16", type=bool, default=True)
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--GPT_API", type=str, default=None)

    return parser.parse_args()

args = get_args()


# Set the random seed for NumPy
np.random.seed(args.seed)
# Set the random seed for PyTorch
torch.manual_seed(args.seed)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(args.seed)

# Load model and template
if args.model_name == "vicuna":
    model_name = "lmsys/vicuna-7b-v1.5"
elif args.model_name == "llama":
    model_name = "cognitivecomputations/Dolphin3.0-Llama3.2-1B"
elif args.model_name == "qwen":
    model_name = "Qwen/Qwen2-1.5B-Instruct"
elif args.model_name == "llama2":
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

else:
    raise ValueError("Invalid model name.")

# Logging Settings
output_dir = "../lora_modules/" + args.model_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_name = "finetune_"+args.model_name+".log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, log_name)),
        logging.StreamHandler()
    ]
)
logging.info(f"Args: {args}")



# Load Model, Tokenizer and Template
device = f'cuda:{args.device}'
model, tokenizer = load_model_and_tokenizer(model_name, 
                       FP16=args.FP16,
                       low_cpu_mem_usage=args.low_cpu_mem_usage,
                       use_cache=args.use_cache,
                       do_sample=False,
                       device=device)




ft_datasets = []
# save_path = "ft_data_llama.json"
# save_path = "ft_data_qwen1.json"
save_path = f"/root/SafeDecoding/datasets/ft_data/ft_data_{args.model_name}.json"


# LoRa Training
# Load Dataset
# dataset = load_dataset('json', data_files=save_path, split="train[:100]")
dataset = load_dataset('json', data_files=save_path, split="train")


# Define LoRA parameters
peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
    bias=args.bias,
    task_type="CAUSAL_LM",
    # target_modules=['c_attn','c_proj','w1','w2']
    # target_modules=['q_proj','k_proj','v_proj','o_proj']
    target_modules=['q_proj','v_proj']
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim=args.optim,
    num_train_epochs=args.num_train_epochs,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    fp16=False,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=args.lr_scheduler_type,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

# Debug: Check if LoRa B Matrix is 0
lora_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
if next(iter(lora_params.values())).any():
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    logging.info(f"Model is saved to {output_dir}. All done!")
else:
    logging.info("LoRA B Matrix is 0. Please Debug. Model not saved.")