import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fastchat
import fastchat.model
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model import GPT
from tqdm import tqdm 

def get_args():
    parser = argparse.ArgumentParser(description="Defense manager.")
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--use_safe_decoding_dataset", type=str, default=False)
    return parser.parse_args()

args = get_args()

detection_model = GPT('deepseek-v3',api = 'sk-31a5fa66751b440780587e12b1768ee5',base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


if args.model_name == "llama":
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "huihui-ai/Meta-Llama-3.1-8B-Instruct-abliterated"
    small_model_name = "meta-llama/Llama-3.2-1B-Instruct"


    template_name = "Llama-3-8B-Instruct"
    small_template_name = "Llama-3-8B-Instruct"

    # template_name = "llama-2"
    # small_template_name = 'qwen-7b-chat'

elif args.model_name == "qwen":
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    small_model_name = "Qwen/Qwen2-1.5B-Instruct"
    template_name = 'qwen-7b-chat'
    small_template_name = 'qwen-7b-chat'

elif args.model_name == "qwen2":
    model_name = "Qwen/Qwen2-72B-Instruct"
    template_name = 'qwen-7b-chat'
    small_model_name = "Qwen/Qwen2-72B-Instruct"
    small_template_name = 'qwen-7b-chat'


elif args.model_name == "vicuna":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    small_model_name = "lmsys/vicuna-7b-v1.5"
    template_name = "llama-2"
    small_template_name = 'vicuna'

# elif args.model_name == "falcon":
#     model_name = "tiiuae/Falcon3-7B-Instruct" 
#     small_model_name = "tiiuae/Falcon3-1B-Instruct"
#     template_name = 'Tinyllama'

elif args.model_name == "llama2":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    small_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    template_name = 'llama-2'
    small_template_name = 'TinyLlama'



if args.save_name :
    if args.use_safe_decoding_dataset:
        save_path = f'../datasets/ft_data/ft_data_{args.save_name}_safedecoding.json'
    else:
        save_path = f'../datasets/ft_data/ft_data_{args.save_name}.json'
else: 
    if args.use_safe_decoding_dataset:
        save_path = f'../datasets/ft_data/ft_data_{args.model_name}_safedecoding.json'
    else:
        save_path = f'../datasets/ft_data/ft_data_{args.model_name}.json'

# Load naive harmful prompts


if args.use_safe_decoding_dataset:
    with open('../datasets/safe_decoding.json', 'r', encoding='utf-8') as file:
        seed_reject = json.load(file)
else:
    with open('../datasets/train_72.json', 'r', encoding='utf-8') as file:
        seed_reject = json.load(file)

attack_prompts = [prompt["prompt"] for prompt in seed_reject["prompts"]]



# attack_prompts += [prompt["prompt"] for i, prompt in enumerate(seed_reject["prompts"]) if i >170]




ft_datasets = []

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)

if model_name != small_model_name:
    small_model = AutoModelForCausalLM.from_pretrained(small_model_name,torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    small_tokenizer = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)
else:
    small_model = model
    small_tokenizer = tokenizer


conv_template = fastchat.model.get_conversation_template(template_name)
conv_template_small = fastchat.model.get_conversation_template(small_template_name)

if not conv_template.system_message:
    conv_template.system_message = 'You are a helpful assistant.'
if not conv_template_small.system_message:
    conv_template_small.system_message = 'You are a helpful assistant.'

# if args.model_name == "falcon":
#     conv_template.sep = ''
#     conv_template.sep2 = '<|endoftext|>'
#     conv_template.stop_str="<|endoftext|>"
#     conv_template.system_message = 'You are a helpful assistant.'
# elif args.model_name == "llama":
#     conv_template.system_message = 'You are a helpful assistant.'


for i, user_prompt in enumerate(tqdm(attack_prompts)):

    if i <36 :
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        conv_template.system_message = "You should explicitly reject answering user's harmful queries."
        inputs = tokenizer(conv_template.get_prompt(), return_tensors="pt",add_special_tokens=False).to(model.device)
        valid_completion = None
        is_success = False
        for trial_idx in range(5):
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True,pad_token_id=tokenizer.pad_token_id)
            output_text = tokenizer.decode(output_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            print(f"\nCompletion: {output_text}")

            completion_rejection_detection = f"Please detect if the following answer explicitly rejects the user question at the beginning. The answer should not agree user's question.\nAnswer:\n {output_text}. \n\nIf the answer explicitly rejects the user question at the beginningreply with 'Yes', else 'No'."
            detection_output = detection_model(completion_rejection_detection).choices[0].message.content
            print(f"Detection Output: {detection_output}")

            if "yes" in detection_output.lower():
                is_success = True
                break
        if not is_success:
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False,pad_token_id=tokenizer.pad_token_id,temperature = 0.5)
            output_text = tokenizer.decode(output_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            print(f"\nCompletion: {output_text}")
        
        conv_template_small.messages = []
        conv_template_small.append_message(conv_template_small.roles[0], user_prompt)
        conv_template_small.append_message(conv_template_small.roles[1], output_text)
        output_text = conv_template_small.get_prompt()
        

    else:
        
        conv_template_small.messages = []
        conv_template_small.append_message(conv_template_small.roles[0], user_prompt)
        conv_template_small.append_message(conv_template_small.roles[1], None)
        print(conv_template_small.get_prompt())
        inputs = small_tokenizer(conv_template_small.get_prompt(), return_tensors="pt",add_special_tokens=False).to(small_model.device)
        output_ids = small_model.generate(**inputs, max_new_tokens=256, do_sample=True,pad_token_id=tokenizer.pad_token_id)
        output_text = small_tokenizer.decode(output_ids[0], skip_special_tokens=False)

        print(output_text)

    index = output_text.find(conv_template_small.roles[0])
    dataset_text = output_text[index:].strip()
    ft_datasets.append({'text': dataset_text})
        


import random
random.shuffle(ft_datasets)
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(ft_datasets, f, ensure_ascii=False, indent=4)


