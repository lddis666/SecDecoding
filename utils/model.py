from typing import Any
from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed
import google.generativeai as genai
from transformers import GenerationConfig
import boto3
import json
# "https://dashscope.aliyuncs.com/compatible-mode/v1"
class GPT:
    def __init__(self, model_name, api=None, temperature=0, seed=0, base_url ="https://api.openai.com/v1" ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api,
            base_url= base_url
        )
        self.T = temperature
        self.seed=seed
        self.generation_config = GenerationConfig.from_pretrained("gpt2")
        

    def __call__(self, prompt,  n:int=1, debug=False, **kwargs: Any) -> Any:
        prompt = [{'role':'user', 'content':prompt}]
        if debug:
            return self.client.chat.completions.create(messages=prompt, n=n, model=self.model_name, temperature=self.T, seed=self.seed, **kwargs)
            
        else:
            return self.call_wrapper(messages=prompt, n=n, model=self.model_name,temperature=self.T, seed=self.seed,  **kwargs)
    

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                [wait_fixed(5) for i in range(2)] +
                [wait_fixed(10)])) # not use for debug
    def call_wrapper(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]


    def generate(self, prompt, max_new_tokens):
        completion = self.client.completions.create(
            model=self.model_name,
            prompt= prompt,
            max_tokens=max_new_tokens,
            temperature=0,
            logprobs = 5,)

        return completion.choices[0].text, completion.choices[0].logprobs.tokens, completion.choices[0].logprobs.top_logprobs

    

def load_model(model_name, api_idx, **kwargs):
    if "gpt" in model_name and "gpt2" not in model_name:
        return GPT(model_name, **kwargs)

    else:
        raise ValueError(f"model_name invalid")


