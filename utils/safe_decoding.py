import torch
import numpy as np
import copy
import logging
from peft import PeftModel, PeftModelForCausalLM
from transformers import AutoModelForCausalLM
from utils.string_utils import pad_and_merge
from math import exp
import torch.nn.functional as F

from utils.model import GPT

from typing import Tuple, Set

class SafeDecoding:
    def __init__(self, model, tokenizer, adapter_names, alpha=1, first_m=5, top_k = 10, num_common_tokens = 3, verbose=False, small_model = None):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_names = adapter_names
        self.alpha = alpha
        self.first_m = first_m 
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose
        self.small_model = small_model

        logging.info("SafeDecoding initialized.")

    @torch.no_grad()
    def safedecoding_lora(self, inputs, gen_config=None,MMLU = None):
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens
        do_sample = gen_config.do_sample

        # Override the generation config for our decoding
        gen_config.max_new_tokens = 1  # We generate one token at a time
        gen_config.do_sample = False  # We use greedy decoding

        generated_sequence = []
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        step = 1  # Keep track of generation steps
        while step <= min(max_token_len, self.first_m):  # Loop until we reach the first m tokens
            # Generate the next token
            # duplicate inputs for two original and expert model
            inputs_duplicated = {k:v.repeat(2,1) for k,v in inputs.items()}

            outputs = self.model.generate(**inputs_duplicated,
                                    adapter_names=self.adapter_names,
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id = self.tokenizer.eos_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,)
            
            output_base = copy.deepcopy(outputs)
            output_expert = copy.deepcopy(outputs)
            output_base.sequences = output_base.sequences[0].unsqueeze(0)
            output_base.scores = output_base.scores[0][0].unsqueeze(0)
            output_expert.sequences = output_expert.sequences[1].unsqueeze(0)
            output_expert.scores = output_expert.scores[0][1].unsqueeze(0)

            # Process the scores to get the top tokens
            k = self.top_k  # Change this to display more or less tokens
            scores_base = output_base.scores[-1].squeeze()  # Get the scores of the last token
            scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
            topk_scores_base, topk_indices_base = scores_base.topk(k) 
            
            scores_expert = output_expert.scores[-1].squeeze()  # Get the scores of the last token
            scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
            topk_scores_expert, topk_indices_expert = scores_expert.topk(k) 

            sorted_indices_base = torch.argsort(scores_base, descending=True)
            sorted_indices_expert = torch.argsort(scores_expert, descending=True)

            # Step 1: Define Sample Space
            common_tokens = set()
            iter_range = self.num_common_tokens
            while len(common_tokens) < self.num_common_tokens:
                current_indices_base = sorted_indices_base[:iter_range]
                current_indices_expert = sorted_indices_expert[:iter_range]

                common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
                common_tokens.update(common_in_iteration)

                iter_range += 1

                if iter_range > min(len(sorted_indices_base), len(sorted_indices_expert)):
                    break

            # Display the top tokens
            if self.verbose and step <=3:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("Original Model")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                logging.info("Expert Model")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (score, token_id) in enumerate(zip(topk_scores_expert, topk_indices_expert)):
                    token = self.tokenizer.decode(token_id.item())
                    prob = torch.exp(score)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

            intersection_indices = torch.tensor(list(common_tokens), device=self.model.device)
            
            # Step 2: New Probability Calculation
            updated_scores = []
            for token_id in intersection_indices:
                # Steer scores
                # new_score = (1-self.alpha) * scores_base[token_id] + self.alpha * scores_expert[token_id]
                # updated_scores.append(new_score)

                # Steer probabilities
                prob_diff = torch.exp(scores_expert[token_id]) - torch.exp(scores_base[token_id])
                updated_prob = torch.exp(scores_base[token_id]) + self.alpha * prob_diff
                # Floor the probability to 1e-8 to avoid log(0)
                updated_prob = updated_prob if updated_prob > 0 else torch.tensor(1e-8, device=self.model.device)
                updated_score = torch.log(updated_prob)
                updated_scores.append(updated_score)

                if self.verbose:
                    logging.info(f"----------------token id: {token_id}-----------------")
                    logging.info(f"Prob Base: {torch.exp(scores_base[token_id])}")
                    logging.info(f"Prob Expert: {torch.exp(scores_expert[token_id])}")
                    logging.info(f"Base score: {scores_base[token_id]}")
                    logging.info(f"Expert score: {scores_expert[token_id]}")
                    logging.info(f"Updated Probability: {updated_prob}")
                    logging.info(f"Updated Score: {updated_score}")

            # Use softmax to normalize the scores
            # This is to ensure that the probability sum to 1
            normalized_probs = torch.nn.functional.softmax(torch.tensor(updated_scores).float(), dim=0)

            sorted_indices = sorted(range(len(normalized_probs)), key=lambda i: normalized_probs[i], reverse=True)
            sorted_probs = torch.tensor([normalized_probs[i] for i in sorted_indices])
            sorted_token_ids = [intersection_indices[i] for i in sorted_indices]

            if self.verbose:
                logging.info("\n-----------------------------------------------")
                logging.info(f"Generation Step {step}")
                logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                logging.info("|----|----------|---------|----------|---------|")
                for idx, (prob, token_id) in enumerate(zip(sorted_probs, sorted_token_ids)):
                    token = self.tokenizer.decode(token_id.item())
                    score = torch.log(prob)
                    logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")


            if MMLU:
                if MMLU == 1:
                    letters = ["A","B","C","D"]
                else:
                    letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
                letter_token_ids = [self.tokenizer.encode(letter,add_special_tokens=False)[0] for letter in letters]

                id_to_rank = {token_id.item(): rank for rank, token_id in enumerate(sorted_token_ids)}
                best_token_id = min(letter_token_ids, key=lambda tid: id_to_rank.get(tid, float('inf')))           
                if id_to_rank.get(best_token_id, float('inf')) == float('inf'):
                    best_token_id = sorted_token_ids[0]

                selected_token_id = torch.tensor(best_token_id,device = self.model.device).unsqueeze(0)

            ### Sample the next token
            elif do_sample == False:
                # Greedy decoding
                # Append the selected token to the sequence
                selected_token_id = sorted_token_ids[0].unsqueeze(0)
            elif gen_config.top_p != None and do_sample == True:
                # Top-p sampling, sample from the top-p tokens
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                p_index = torch.where(cumulative_probs >= gen_config.top_p)[0][0]
                sorted_top_p_token_ids = sorted_token_ids[:p_index + 1]
                sorted_top_p_probs = sorted_probs[:p_index + 1]
                sorted_top_p_scores = torch.log(sorted_top_p_probs)
                if self.verbose:
                    logging.info(f"Top-p token ids: {sorted_top_p_token_ids}")
                    logging.info(f"Top-p scores: {sorted_top_p_scores}")
                    logging.info(f"Top-p probabilities: {sorted_top_p_probs}")
                
                # Sample from the top-p tokens
                selected_token_id = sorted_top_p_token_ids[torch.multinomial(torch.softmax(sorted_top_p_scores, dim=-1), 1)].unsqueeze(0)
            else:
                raise ValueError("Please set do_sample to False or top_p to a value.")

            if self.verbose:
                logging.info(f"Selected token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")
            generated_sequence.append(selected_token_id.item())

            # if the chosen token id is eos, then stop
            if selected_token_id.item() == self.tokenizer.eos_token_id:
                break

            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0)], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1)

            step += 1

            # Free up memory
            del output_base, output_expert


        # Use the normal model to generate the rest of the tokens
        # Early stop if the last token is eos
        if generated_sequence[-1] == self.tokenizer.eos_token_id:
            logging.info("Early stop triggered.")
        else:
            remaining_steps = max_token_len - min(max_token_len, self.first_m)

            if remaining_steps>0:
                gen_config.max_new_tokens = remaining_steps
                gen_config.do_sample = do_sample
                output_base = self.model.generate(**inputs,
                                        adapter_names=["__base__"],
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True,)
            
                generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # logging.info generated sequence
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence, skip_special_tokens=True), len(generated_sequence)

    @torch.no_grad()
    def secdecoding_lora(self, inputs, gen_config=None, small_inputs = None, MMLU = None, alpha_base_val=10.0, gamma_val=10.0, beta_val=0.05):
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens
        # max_token_len = 100
        do_sample = gen_config.do_sample

        # Override the generation config for our decoding
        gen_config.max_new_tokens = 1  # We generate one token at a time
        gen_config.do_sample = False  # We use greedy decoding

        generated_sequence = []
        similarity_list = []
        alpha_list = []
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")
        if not small_inputs:
            small_inputs = copy.deepcopy(inputs)

        if not isinstance(self.model, GPT):
            inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
            small_inputs = {k:v.cuda(self.model.device) for k,v in small_inputs.items()}
            input_len = inputs['input_ids'].shape[1]

            step = 1  # Keep track of generation steps
            # while step <= min(max_token_len, self.first_m):  # Loop until we reach the first m tokens



            alpha_manager = Dynamic_alpha(alpha_base_val=alpha_base_val, gamma_val=gamma_val, beta_val=beta_val, tokenizer = self.tokenizer)




            while step <= max_token_len:
                # Generate the next token
                # duplicate inputs for two original and expert model
                
                inputs_duplicated = {k:v.repeat(2,1).cuda(self.small_model.device) for k,v in small_inputs.items()}

                base_out = self.small_model(
                    small_inputs['input_ids'],  adapter_names = ['__base__']
                )
                base_scores = base_out.logits[:, -1:, :][0][0]  # (N, vocab)

                expert_out = self.small_model(
                    small_inputs['input_ids'],  adapter_names = ['expert']
                )
                expert_scores = expert_out.logits[:, -1:, :][0][0]  # (N, vocab)
                # print(f'1:{expert_scores.shape}, 2:{base_scores.shape}')







                
                
                # small_outputs = self.small_model.generate(**inputs_duplicated,
                #                         adapter_names=self.adapter_names,
                #                         generation_config=gen_config,
                #                         pad_token_id=self.tokenizer.pad_token_id,
                #                         return_dict_in_generate=True,
                #                         output_scores=True,)
                
                # base_scores = small_outputs['scores'][0][0]
                # expert_scores = small_outputs['scores'][0][1]
                base_prob = torch.softmax(base_scores,dim=-1)
                expert_prob = torch.softmax(expert_scores,dim=-1)


                k = self.top_k
                topk_prob_base, topk_indices_base = base_prob.topk(k) 
                topk_prob_expert, topk_indices_expert = expert_prob.topk(k) 



                # outputs = self.model.generate(**inputs,
                #     generation_config=gen_config,
                #     pad_token_id=self.tokenizer.pad_token_id,
                #     return_dict_in_generate=True,
                #     output_scores=True,)
                # model_scores = outputs['scores'][0][0] 

                model_out = self.model(
                    inputs['input_ids']
                )
                model_scores = model_out.logits[:, -1:, :][0][0]  # (N, vocab)

                # print(model_scores.shape) 

                model_prob = torch.softmax(model_scores,dim=-1)
                topk_prob_model, topk_indices_model = model_prob.topk(k) 


                # print("--- Running Scheme A (Additive with TVD) ---")

                alpha = alpha_manager.get_alpha(
                    p1=expert_prob,
                    p2=base_prob,
                    k=step,
                    threshold=0.05,
                    verbose=self.verbose
                ).item()


                final_prob =  model_prob[:len(expert_prob)].to(expert_prob.device)  + alpha*(expert_prob - base_prob)
                topk_prob_final, topk_indices_final = final_prob.topk(k) 

                wasserstein_dist = torch.sum(torch.abs(base_prob - expert_prob)).item()
                similarity_list.append(wasserstein_dist)
                alpha_list.append(alpha)


                if self.verbose:
                    logging.info("\n-----------------------------------------------")
                    logging.info(f"Generation Step {step}")
                    logging.info("Original Model")
                    logging.info("|No. | Token ID | Token    | Prob    |")
                    logging.info("|----|----------|---------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_prob_base, topk_indices_base)):
                        token = self.tokenizer.decode(token_id.item())
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s}  | {prob:.2%} |")


                    logging.info("Expert Model")
                    logging.info("|No. | Token ID | Token   | Prob    |")
                    logging.info("|----|----------|---------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_prob_expert, topk_indices_expert)):
                        token = self.tokenizer.decode(token_id.item())
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {prob:.2%} |")

                    logging.info("Big Model")
                    logging.info("|No. | Token ID | Token   | Prob    |")
                    logging.info("|----|----------|---------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_prob_model, topk_indices_model)):
                        token = self.tokenizer.decode(token_id.item())
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {prob:.2%} |")


                    logging.info(f"alpha: {alpha}")
                    logging.info("Final Model")
                    logging.info("|No. | Token ID | Token   | Prob    |")
                    logging.info("|----|----------|---------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_prob_final, topk_indices_final)):
                        token = self.tokenizer.decode(token_id.item())
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {prob:.2%} |")


                if MMLU:
                    if MMLU == 1:
                        letters = ["A","B","C","D"]
                    else:
                        letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
                    letter_token_ids = [self.tokenizer.encode(letter,add_special_tokens=False)[0] for letter in letters]
                    selected_probs = final_prob[letter_token_ids]
                    max_idx_in_letters = torch.argmax(selected_probs).item()
                    selected_token_id = torch.tensor(letter_token_ids[max_idx_in_letters]).unsqueeze(0)
                else:
                    selected_token_id = topk_indices_final[0].unsqueeze(0)
                generated_sequence.append(selected_token_id.item())

                if selected_token_id.item() == self.tokenizer.eos_token_id:
                    logging.info("Early stop triggered.")
                    break


                inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0).to(inputs['input_ids'].device)], dim=1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1)

                small_inputs['input_ids'] = torch.cat([small_inputs['input_ids'], selected_token_id.unsqueeze(0).to(inputs['input_ids'].device)], dim=1)
                small_inputs['attention_mask'] = torch.cat([small_inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1)

                step += 1

                if alpha < 1e-6:
                    break




            if inputs['input_ids'][0][-1].item() == self.tokenizer.eos_token_id:
                logging.info("Early stop triggered.")
            else:
                remaining_steps = max_token_len - len(generated_sequence)
                if remaining_steps>0:

                    for _ in range(remaining_steps):
                        draft_out = self.model(
                            inputs['input_ids'], 
                        )


                        next_logits = draft_out.logits[:, -1, :]   # 取最后一步



                        next_token = next_logits.argmax(-1, keepdim=True)
                        # print(inputs['input_ids'].shape, next_token.shape)
                        # 合并
                        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=1)
                        # inputs['input_ids'].append(next_token)
                        generated_sequence.append(next_token.item())

                        if next_token.item() == self.tokenizer.eos_token_id:
                            logging.info("Stop")
                            break



                    # gen_config.max_new_tokens = remaining_steps
                    # gen_config.do_sample = do_sample
                    # output_base = self.model.generate(**inputs,
                    #                         generation_config=gen_config,
                    #                         pad_token_id=self.tokenizer.pad_token_id,
                    #                         return_dict_in_generate=True,
                    #                         output_scores=True,)
                
                    # generated_sequence = output_base.sequences[0].tolist()[input_len:]



            # logging.info generated sequence
            logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

            return self.tokenizer.decode(generated_sequence, skip_special_tokens=True), len(generated_sequence), similarity_list, alpha_list

        else:
                
            generated = ''
            inputs = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)

            

            small_inputs = {k:v.cuda(self.small_model.device) for k,v in small_inputs.items()}
            input_len = small_inputs['input_ids'].shape[1]

            step = 1  
            alpha_manager = Dynamic_alpha(alpha_base_val=alpha_base_val, gamma_val=gamma_val, beta_val=beta_val, tokenizer = self.tokenizer)




            while step <= max_token_len:

                
                inputs_duplicated = {k:v.repeat(2,1).cuda(self.small_model.device) for k,v in small_inputs.items()}
                small_outputs = self.small_model.generate(**inputs_duplicated,
                                        adapter_names=self.adapter_names,
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True,)
                
                base_scores = small_outputs['scores'][0][0]
                expert_scores = small_outputs['scores'][0][1]
                base_prob = torch.softmax(base_scores,dim=-1)
                expert_prob = torch.softmax(expert_scores,dim=-1)


                k = self.top_k
                topk_prob_base, topk_indices_base = base_prob.topk(k) 
                topk_prob_expert, topk_indices_expert = expert_prob.topk(k) 



                outputs, tokens, logprobs = self.model.generate(inputs+generated, 1)

                if not outputs:
                    break

                


                # print("--- Running Scheme A (Additive with TVD) ---")

                alpha = alpha_manager.get_alpha(
                    p1=expert_prob,
                    p2=base_prob,
                    k=step,
                    threshold=0.05,
                    verbose=self.verbose
                ).item()

                final_prob = alpha*(expert_prob - base_prob)
                for token, prob in logprobs[0].items():
                    token_list = self.tokenizer.encode(token, add_special_tokens = False)
                    final_prob[token_list] += exp(prob)
                    

                topk_prob_final, topk_indices_final = final_prob.topk(k) 


                if self.verbose:
                    logging.info("\n-----------------------------------------------")
                    logging.info(f"Generation Step {step}")
                    logging.info("Original Model")
                    logging.info("|No. | Token ID | Token    | Prob    |")
                    logging.info("|----|----------|---------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_prob_base, topk_indices_base)):
                        token = self.tokenizer.decode(token_id.item())
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s}  | {prob:.2%} |")


                    logging.info("Expert Model")
                    logging.info("|No. | Token ID | Token   | Prob    |")
                    logging.info("|----|----------|---------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_prob_expert, topk_indices_expert)):
                        token = self.tokenizer.decode(token_id.item())
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {prob:.2%} |")

                    logging.info("Big Model")
                    logging.info("|No. | Token   | Prob    |")
                    logging.info("|----|---------|---------|")
                    for idx, (token,log_prob) in enumerate(logprobs[0].items()):
                        logging.info(f"{idx+1:4d} |  {token:7s} | {exp(log_prob):.2%} |")


                    logging.info(f"alpha: {alpha}")
                    logging.info("Final Model")
                    logging.info("|No. | Token ID | Token   | Prob    |")
                    logging.info("|----|----------|---------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(topk_prob_final, topk_indices_final)):
                        token = self.tokenizer.decode(token_id.item())
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {prob:.2%} |")


                if MMLU:
                    if MMLU == 1:
                        letters = ["A","B","C","D"]
                    else:
                        letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
                    letter_token_ids = [self.tokenizer.encode(letter,add_special_tokens=False)[0] for letter in letters]
                    selected_probs = final_prob[letter_token_ids]
                    max_idx_in_letters = torch.argmax(selected_probs).item()
                    selected_token_id = torch.tensor(letter_token_ids[max_idx_in_letters]).unsqueeze(0)
                else:
                    selected_token_id = topk_indices_final[0].unsqueeze(0)
                generated_sequence.append(selected_token_id.item())
                generated += self.tokenizer.decode(selected_token_id.item())

                if selected_token_id.item() == self.tokenizer.eos_token_id:
                    logging.info("Early stop triggered.")
                    break


                small_inputs['input_ids'] = torch.cat([small_inputs['input_ids'], selected_token_id.unsqueeze(0).to(small_inputs['input_ids'].device)], dim=1)
                small_inputs['attention_mask'] = torch.cat([small_inputs['attention_mask'], torch.tensor([[1]], device=self.small_model.device)], dim=1)

                step += 1

                if alpha < 1e-6:
                    break




            if small_inputs['input_ids'][0][-1].item() == self.tokenizer.eos_token_id:
                logging.info("Early stop triggered.")
            else:
                remaining_steps = max_token_len - len(generated_sequence)
                if remaining_steps>0:
                    outputs, _, _ = self.model.generate(inputs+generated, remaining_steps)
                    generated += outputs

            # logging.info generated sequence
            logging.info(f"Generated sequence: {generated}")

            return generated, len(generated), similarity_list, alpha_list
    

    @torch.no_grad()
    def secdecoding_speculative_greedy(self, inputs, gen_config=None, small_inputs = None, alpha_base_val=10.0, gamma_val=10.0, beta_val=0.05,gamma=5):


        eos_token_id = self.tokenizer.eos_token_id
        if not small_inputs:
            small_inputs = copy.deepcopy(inputs)

        small_ids = small_inputs['input_ids'].to(self.small_model.device)  
        large_ids = inputs['input_ids'].to(self.model.device)                  # 当前已生成的全部token


        small_len = small_ids.shape[1]
        large_len = large_ids.shape[1]


        
        generated = []                                  # 新生成部分（不包括输入）
        main_kv_cache = None 
        base_kv_cache = None 
        expert_kv_cache = None                          
        max_new_tokens = gen_config.max_new_tokens
        alpha_manager = Dynamic_alpha(alpha_base_val=alpha_base_val, gamma_val=gamma_val, beta_val=beta_val, tokenizer = self.tokenizer)
        draft_input = small_ids
        additional_token = None
        use_SecDecoding = True

        for _ in range(max_new_tokens):
            # Step1: 草稿模型投机gamma步（遇eos提前停）
            draft_ids = []
            draft_logits = []


            for _ in range(gamma):
                draft_out = self.small_model(
                    draft_input, past_key_values=expert_kv_cache, use_cache=True, adapter_names = ['expert']
                )


                next_logits = draft_out.logits[:, -1, :]   # 取最后一步



                next_token = next_logits.argmax(-1, keepdim=True)
                draft_logits.append(next_logits)
                draft_ids.append(next_token)
                if next_token.item() == eos_token_id:
                    break
                expert_kv_cache = draft_out.past_key_values
                draft_input = next_token
                if self.verbose:
                    logging.info(f'猜: {self.tokenizer.decode(next_token[0])}')


            # 2. 构造主模型的批量并行验证输入
            batch_draft = torch.cat(draft_ids, dim=-1)   # (1, N)
            if not additional_token:
                verify_input_large = torch.cat([large_ids, batch_draft.to(large_ids.device)], dim=-1) 
                verify_input_small = torch.cat([small_ids, batch_draft.to(small_ids.device)], dim=-1) 
            else:
                verify_input_large = torch.cat([additional_token, batch_draft.to(additional_token.device)], dim=-1) 
                verify_input_small = torch.cat([additional_token, batch_draft.to(additional_token.device)], dim=-1) 


            main_out = self.model(
                verify_input_large,
                past_key_values=main_kv_cache,
                use_cache=True,
            )  
            if use_SecDecoding:
                base_out = self.small_model(
                    verify_input_small, past_key_values=base_kv_cache, use_cache=True, adapter_names = ['__base__']
                )

            # 主模型只需要比较 len(draft_ids) 步的输出
            main_logits = main_out.logits[:, -len(draft_ids)-1:, :][0] # (N, vocab)
            main_kv_cache = main_out.past_key_values
            if use_SecDecoding:
                base_logits = base_out.logits[:, -len(draft_ids)-1:, :][0]  # (N, vocab)
                base_kv_cache = base_out.past_key_values

            accept = True
            for _, token in enumerate(draft_ids):

                if use_SecDecoding:
                    base_prob = torch.softmax(base_logits[_],dim=-1)
                    expert_prob = torch.softmax(draft_logits[_][0],dim=-1)

                    alpha = alpha_manager.get_alpha(
                        p1=expert_prob,
                        p2=base_prob,
                        k=len(generated)+1,
                        threshold=0.05,
                        verbose=self.verbose
                    ).item()


                    final_prob =  torch.softmax(main_logits[_][:len(expert_prob)],dim=-1).to(expert_prob.device)  + alpha*(expert_prob - base_prob)      
                    pred_token = final_prob.argmax(-1)
                else:
                    pred_token =  torch.softmax(main_logits[_],dim=-1).argmax(-1)

                if alpha < 1e-6:
                    use_SecDecoding = False


                if self.verbose:
                    logging.info(f'正确的: {self.tokenizer.decode(pred_token)}')

                # print(pred_token)
                if pred_token.item() == token.item():
                    # 验证通过，纳入生成
                    if self.verbose:
                        logging.info('通过')
                    generated.append(token.item())
                    if token.item() == eos_token_id:
                        # draft与main_model都认为结束，直接终止

                        logging.info(self.tokenizer.decode(generated, skip_special_tokens=True))
                        return self.tokenizer.decode(
                            generated, skip_special_tokens=True
                        ),len(generated)
                    if len(generated) >= max_new_tokens:
                        logging.info(self.tokenizer.decode(generated, skip_special_tokens=True))
                        return self.tokenizer.decode(
                            generated, skip_special_tokens=True
                        ),len(generated)
                else:
                    if self.verbose:
                        logging.info('失败')
                    accept = False
                    # 未通过，用主模型自己的预测token，并中断草稿

                    if use_SecDecoding:
                        base_kv_cache = rollback(base_kv_cache,len(generated)+small_len)
                    main_kv_cache = rollback(main_kv_cache,len(generated)+large_len)
                    expert_kv_cache = rollback(expert_kv_cache,len(generated)+small_len)

                    additional_token = pred_token.unsqueeze(0).unsqueeze(0)
                    draft_input = pred_token.unsqueeze(0).unsqueeze(0)
                    generated.append(pred_token.item())
                    if self.verbose:
                        logging.info(f'修正：{self.tokenizer.decode(pred_token)}')
                    if pred_token.item() == eos_token_id:
                        logging.info(self.tokenizer.decode(generated, skip_special_tokens=True))
                        return self.tokenizer.decode(
                            generated, skip_special_tokens=True
                        ),len(generated)
                    break
            if accept:
                if self.verbose:
                    logging.info('全猜对了')
                additional_token = pred_token.unsqueeze(0).unsqueeze(0)
                draft_input = pred_token.unsqueeze(0).unsqueeze(0)
                if use_SecDecoding:
                    base_kv_cache = rollback(base_kv_cache,len(generated)+small_len-1)
                main_kv_cache = rollback(main_kv_cache,len(generated)+large_len-1)
            if len(generated) >= max_new_tokens:
                break
           
        logging.info(self.tokenizer.decode(generated, skip_special_tokens=True))
        return self.tokenizer.decode(generated, skip_special_tokens=True),len(generated)

            

        #     # Step2: 主模型验证草稿。遇不一致或eos，立即终止
        #     verify_input_large = large_ids
        #     verify_input_small = small_ids


        #     for _, token in enumerate(draft_ids):
        #         with torch.no_grad():
        #             main_out = self.model(
        #                 verify_input_large, past_key_values=main_kv_cache, use_cache=True
        #             )
        #             base_out = self.small_model(
        #                 verify_input_small, past_key_values=base_kv_cache, use_cache=True, adapter_names = ['__base__']
        #             )

        #             expert_prob = torch.softmax(draft_logits[_][0],dim=-1)
        #             base_prob = torch.softmax(base_out.logits[0][-1],dim=-1)
        #             alpha = alpha_manager.get_alpha(
        #                 p1=expert_prob,
        #                 p2=base_prob,
        #                 k=len(generated)+1,
        #                 threshold=0.05,
        #                 verbose=self.verbose
        #             ).item()

        #             # print(main_out.logits[0][-1].max(dim=-1))


        #             final_prob =  torch.softmax(main_out.logits[0][-1][:len(expert_prob)],dim=-1).to(expert_prob.device)  + alpha*(expert_prob - base_prob)
                    
                
        #         pred_token = final_prob.argmax(-1)
        #         # print(pred_token)
        #         if pred_token.item() == token.item():
        #             # 验证通过，纳入生成
        #             generated.append(token.item())
        #             small_ids = torch.cat([small_ids, token], dim=-1)
        #             large_ids = torch.cat([large_ids, token], dim=-1)
        #             main_kv_cache = main_out.past_key_values
        #             verify_input_large = token
        #             verify_input_small = token
        #             if token.item() == eos_token_id:
        #                 # draft与main_model都认为结束，直接终止
        #                 return self.tokenizer.decode(
        #                     generated, skip_special_tokens=True
        #                 ),len(generated)
        #             if len(generated) >= max_new_tokens:
        #                 return self.tokenizer.decode(
        #                     generated, skip_special_tokens=True
        #                 ),len(generated)
        #         else:
        #             # 未通过，用主模型自己的预测token，并中断草稿
        #             generated.append(pred_token.item())

        #             small_ids = torch.cat([small_ids, pred_token.unsqueeze(0).unsqueeze(0)], dim=-1)
        #             large_ids = torch.cat([large_ids, pred_token.unsqueeze(0).unsqueeze(0)], dim=-1)
        #             main_kv_cache = main_out.past_key_values
        #             if pred_token.item() == eos_token_id:
        #                 return self.tokenizer.decode(
        #                     generated, skip_special_tokens=True
        #                 ),len(generated)
        #             break
        #     if len(generated) >= max_new_tokens:
        #         break

        # return self.tokenizer.decode(generated, skip_special_tokens=True),len(generated)

    @torch.no_grad()
    def secdecoding_prefix(self, inputs, safe_inputs, gen_config=None):
            if gen_config is None:
                gen_config = self.model.generation_config

            max_token_len = gen_config.max_new_tokens
            do_sample = gen_config.do_sample

            # Override the generation config for our decoding
            gen_config.max_new_tokens = 1  # We generate one token at a time
            gen_config.do_sample = False  # We use greedy decoding

            generated_sequence = []
            if self.verbose:
                logging.info(f"Generation config: {gen_config}")
        

            parallel_inputs = pad_and_merge([inputs, safe_inputs], self.tokenizer.pad_token_id)

            inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
            parallel_inputs = {k:v.cuda(self.model.device) for k,v in parallel_inputs.items()}
            input_len = inputs['input_ids'].shape[1]

            step = 1  # Keep track of generation steps
            while step <= min(max_token_len, self.first_m):  # Loop until we reach the first m tokens
                # Generate the next token
                # duplicate inputs for two original and expert model
                
                outputs = self.model.generate(**parallel_inputs,
                                        adapter_names=['base'],
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True,)
                
                output_base = copy.deepcopy(outputs)
                output_expert = copy.deepcopy(outputs)
                output_base.sequences = output_base.sequences[0].unsqueeze(0)
                output_base.scores = output_base.scores[0][0].unsqueeze(0)
                output_expert.sequences = output_expert.sequences[1].unsqueeze(0)
                output_expert.scores = output_expert.scores[0][1].unsqueeze(0)

                # Process the scores to get the top tokens
                k = self.top_k  # Change this to display more or less tokens
                scores_base = output_base.scores[-1].squeeze()  # Get the scores of the last token
                scores_base = torch.nn.functional.log_softmax(scores_base, dim=-1)
                topk_scores_base, topk_indices_base = scores_base.topk(k) 
                
                scores_expert = output_expert.scores[-1].squeeze()  # Get the scores of the last token
                scores_expert = torch.nn.functional.log_softmax(scores_expert, dim=-1)
                topk_scores_expert, topk_indices_expert = scores_expert.topk(k) 

                sorted_indices_base = torch.argsort(scores_base, descending=True)
                sorted_indices_expert = torch.argsort(scores_expert, descending=True)

                # Step 1: Define Sample Space
                common_tokens = set()
                iter_range = self.num_common_tokens
                while len(common_tokens) < self.num_common_tokens:
                    current_indices_base = sorted_indices_base[:iter_range]
                    current_indices_expert = sorted_indices_expert[:iter_range]

                    common_in_iteration = set(current_indices_base.tolist()) & set(current_indices_expert.tolist())
                    common_tokens.update(common_in_iteration)

                    iter_range += 1

                    if iter_range > min(len(sorted_indices_base), len(sorted_indices_expert)):
                        break

                # Display the top tokens
                if self.verbose and step == 1:
                    logging.info("\n-----------------------------------------------")
                    logging.info(f"Generation Step {step}")
                    logging.info("Original Model")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (score, token_id) in enumerate(zip(topk_scores_base, topk_indices_base)):
                        token = self.tokenizer.decode(token_id.item())
                        prob = torch.exp(score)
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                    logging.info("Expert Model")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (score, token_id) in enumerate(zip(topk_scores_expert, topk_indices_expert)):
                        token = self.tokenizer.decode(token_id.item())
                        prob = torch.exp(score)
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                intersection_indices = torch.tensor(list(common_tokens), device=self.model.device)
                
                # Step 2: New Probability Calculation
                updated_scores = []
                for token_id in intersection_indices:
                    # Steer scores
                    # new_score = (1-self.alpha) * scores_base[token_id] + self.alpha * scores_expert[token_id]
                    # updated_scores.append(new_score)

                    # Steer probabilities
                    prob_diff = torch.exp(scores_expert[token_id]) - torch.exp(scores_base[token_id])
                    updated_prob = torch.exp(scores_base[token_id]) + self.alpha * prob_diff
                    # Floor the probability to 1e-8 to avoid log(0)
                    updated_prob = updated_prob if updated_prob > 0 else torch.tensor(1e-8, device=self.model.device)
                    updated_score = torch.log(updated_prob)
                    updated_scores.append(updated_score)

                    if self.verbose:
                        logging.info(f"----------------token id: {token_id}-----------------")
                        logging.info(f"Prob Base: {torch.exp(scores_base[token_id])}")
                        logging.info(f"Prob Expert: {torch.exp(scores_expert[token_id])}")
                        logging.info(f"Base score: {scores_base[token_id]}")
                        logging.info(f"Expert score: {scores_expert[token_id]}")
                        logging.info(f"Updated Probability: {updated_prob}")
                        logging.info(f"Updated Score: {updated_score}")

                # Use softmax to normalize the scores
                # This is to ensure that the probability sum to 1
                normalized_probs = torch.nn.functional.softmax(torch.tensor(updated_scores).float(), dim=0)

                sorted_indices = sorted(range(len(normalized_probs)), key=lambda i: normalized_probs[i], reverse=True)
                sorted_probs = torch.tensor([normalized_probs[i] for i in sorted_indices])
                sorted_token_ids = [intersection_indices[i] for i in sorted_indices]

                if self.verbose:
                    logging.info("\n-----------------------------------------------")
                    logging.info(f"Generation Step {step}")
                    logging.info("|No. | Token ID | Token   | Log Prob | Prob    |")
                    logging.info("|----|----------|---------|----------|---------|")
                    for idx, (prob, token_id) in enumerate(zip(sorted_probs, sorted_token_ids)):
                        token = self.tokenizer.decode(token_id.item())
                        score = torch.log(prob)
                        logging.info(f"{idx+1:4d} | {token_id:8d} | {token:7s} | {score:.3f}    | {prob:.2%} |")

                ### Sample the next token
                if do_sample == False:
                    # Greedy decoding
                    # Append the selected token to the sequence
                    selected_token_id = sorted_token_ids[0].unsqueeze(0)
                elif gen_config.top_p != None and do_sample == True:
                    # Top-p sampling, sample from the top-p tokens
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    p_index = torch.where(cumulative_probs >= gen_config.top_p)[0][0]
                    sorted_top_p_token_ids = sorted_token_ids[:p_index + 1]
                    sorted_top_p_probs = sorted_probs[:p_index + 1]
                    sorted_top_p_scores = torch.log(sorted_top_p_probs)
                    if self.verbose:
                        logging.info(f"Top-p token ids: {sorted_top_p_token_ids}")
                        logging.info(f"Top-p scores: {sorted_top_p_scores}")
                        logging.info(f"Top-p probabilities: {sorted_top_p_probs}")
                    
                    # Sample from the top-p tokens
                    selected_token_id = sorted_top_p_token_ids[torch.multinomial(torch.softmax(sorted_top_p_scores, dim=-1), 1)].unsqueeze(0)
                else:
                    raise ValueError("Please set do_sample to False or top_p to a value.")

                if self.verbose:
                    logging.info(f"Selected token: {self.tokenizer.decode(selected_token_id.item())}, ID: {selected_token_id.item()}")
                generated_sequence.append(selected_token_id.item())

                # if the chosen token id is eos, then stop
                if selected_token_id.item() == self.tokenizer.eos_token_id:
                    break

                inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0)], dim=1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1)

                step += 1

                # Free up memory
                del output_base, output_expert


            # Use the normal model to generate the rest of the tokens
            # Early stop if the last token is eos
            if generated_sequence[-1] == self.tokenizer.eos_token_id:
                logging.info("Early stop triggered.")
            else:
                remaining_steps = max_token_len - min(max_token_len, self.first_m)
                gen_config.max_new_tokens = remaining_steps
                gen_config.do_sample = do_sample
                output_base = self.model.generate(**inputs,
                                        adapter_names=["base"],
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        eos_token_id = self.tokenizer.eos_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True,)
                
                generated_sequence = output_base.sequences[0].tolist()[input_len:]

            # logging.info generated sequence
            logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

            return self.tokenizer.decode(generated_sequence), len(generated_sequence)
    
    @torch.no_grad()
    def generate_baseline(self, inputs,  gen_config=None, MMLU = None):
        if gen_config is None:
            gen_config = self.model.generation_config
        
        # gen_config.max_new_tokens = 10
        # gen_config.do_sample = False 
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        if isinstance(self.model, GPT):
            text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            print(text)
            outputs, _ ,_ = self.model.generate(text,gen_config.max_new_tokens)
            logging.info(f"Generated sequence: {outputs}")
            return outputs, len(outputs)


            


        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}


        output = self.model.generate(**inputs,
            generation_config=gen_config,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores = True,
            return_dict_in_generate = True)

        if MMLU:
            model_scores = output['scores'][0][0] 
            if MMLU == 1:
                letters = ["A","B","C","D"]
            else:
                letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
            letter_token_ids = [self.tokenizer.encode(letter,add_special_tokens=False)[0] for letter in letters]
            selected_probs = model_scores[letter_token_ids]
            max_idx_in_letters = torch.argmax(selected_probs).item()
            generated_sequence = torch.tensor(letter_token_ids[max_idx_in_letters]).unsqueeze(0)
        else:
            output_ids = output.sequences[0]
            generated_sequence = output_ids[inputs["input_ids"].shape[1]:]
        


        # output_base = self.model.generate(**inputs,
        #                     adapter_names=adapter_name,
        #                     generation_config=gen_config,
        #                     pad_token_id=self.tokenizer.pad_token_id,
        #                     return_dict_in_generate=True,
        #                     output_scores=True,)
        
        # generated_sequence = output_base.sequences[0][inputs["input_ids"].shape[1]:]

        
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        
        return self.tokenizer.decode(generated_sequence, skip_special_tokens=True), len(generated_sequence)

    @torch.no_grad()
    def generate_batch(self, inputs,  gen_config=None, MMLU = None):
        if not MMLU:
            batch_inputs = self.tokenizer(
                inputs, 
                padding=True, 
                truncation=False, 
                return_tensors='pt',
                add_special_tokens = False,
                
            )
            batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
            batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

            # Forward pass through the LLM
            try:
                outputs = self.model.generate(
                    batch_input_ids, 
                    attention_mask=batch_attention_mask, 
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            except RuntimeError:
                return []

            # Decode the outputs produced by the LLM
            batch_outputs = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )
            gen_start_idx = [
                len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
                for i in range(len(batch_input_ids))
            ]
            batch_outputs = [
                output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
            ]

            return batch_outputs
        else:
            inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=False, add_special_tokens = False,).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]  # [batch, vocab]
            
            if MMLU == 1:
                letters = ["A","B","C","D"]
            else:
                letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M"]
            option_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in letters]
            probs = F.softmax(logits, dim=-1)[:, option_ids]   # [batch, 4]
            max_idx = probs.argmax(dim=-1)
            chosen_ids = [option_ids[i.item()] for i in max_idx]
            chosen = self.tokenizer.batch_decode([[i] for i in chosen_ids], skip_special_tokens=True)
            return chosen



    @torch.no_grad()
    def generate_case(self, inputs,  gen_config=None, MMLU = None):
        if gen_config is None:
            gen_config = self.model.generation_config


        safety_words = ["I", "apologize", "apologise", "cannot", "can't", "illegal", "unethical",
            "fulfiil", "AI", "Sorry", "sorry", "am", "'m", "'t", "'d", "'s", "As", "an",
            "assistant", "Assistant", "unable", "able", "allowed",  "not",
            "do", "don't", "must", "Sure", "sure", "can", "provide",  "but", "assist", "help", "support",
            "complete", ",", "'", "understand", "your", "My", "with", "harm", "harmful",
            "purpose", "positive", "apologies", "legal", "engage", "prefer", "recommend",
                        "I'm sorry", 
                        "Sorry",
                        "sorry",
                        "I am sorry",
                        "I apologize",
                        "As an",
                        "As a language model",
                        "As an Assistant",
                        "I cannot",
                        "cannot fulfill",
                        "can’t assist with",
                        "can't assist with",
                        "can’t do that",
                        "can't do that",
                        "can't provide",
                        "can’t provide",
                        "can't fulfill",
                        "can’t fulfill",
                        "can't help with",
                        "can’t help with",
                        "can’t support",
                        "can't support",
                        "I'm unable to",
                        "I am unable to",
                        "I am not allowed to",
                        "I am not able to",
                        "I am not authorized",
                        "I can't",
                        "I can’t",
                        "I can\'t"]
        safety_tokens = []
        for word in safety_words:
            # if self.tokenizer.convert_tokens_to_ids(word):
            #     safety_tokens.append(self.tokenizer.convert_tokens_to_ids(word))
            # else:
            safety_tokens+=self.tokenizer.encode(word, add_special_tokens=False)
        safety_tokens = list(set(safety_tokens))
        inputs_duplicated = {k:v.repeat(2,1).cuda(self.model.device) for k,v in inputs.items()}


        outputs = self.model.generate(**inputs_duplicated,
                                adapter_names=self.adapter_names,
                                generation_config=gen_config,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id = self.tokenizer.eos_token_id,
                                return_dict_in_generate=True,
                                output_scores=True,)
        


        prob_list = []
        for i, step in enumerate(outputs.scores):
            prob_list.append(torch.softmax(step[0],dim=0)[safety_tokens].sum().item())

            if i>=5:
                break


# torch.where((p1 >= threshold) | (p2 >= threshold))[0].cpu().tolist()

        generated_sequence = outputs.sequences[1][inputs["input_ids"].shape[1]:]
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence, skip_special_tokens=True), len(generated_sequence), prob_list, 





    @torch.no_grad()
    def generate_case1(self, inputs,  gen_config=None, MMLU = None):
        if gen_config is None:
            gen_config = self.model.generation_config

        max_token_len = gen_config.max_new_tokens

        gen_config.max_new_tokens = 1  # We generate one token at a time
        gen_config.do_sample = False  # We use greedy decoding

        generated_sequence = []

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        

        step = 1  
        similarity_list = []
        while step <= max_token_len:
            # Generate the next token
            # duplicate inputs for two original and expert model
            
            inputs_duplicated = {k:v.repeat(2,1).cuda(self.model.device) for k,v in inputs.items()}
            outputs = self.model.generate(**inputs_duplicated,
                                    adapter_names=self.adapter_names,
                                    generation_config=gen_config,
                                    # pad_token_id=self.tokenizer.pad_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,)
            
            base_scores = outputs['scores'][0][0]
            expert_scores = outputs['scores'][0][1]
            base_prob = torch.softmax(base_scores,dim=-1)
            expert_prob = torch.softmax(expert_scores,dim=-1)
            wasserstein_dist = torch.sum(torch.abs(base_prob - expert_prob)).item()
            similarity_list.append(wasserstein_dist)

            selected_token_id = expert_scores.argmax(dim=-1)


            if selected_token_id.item() == self.tokenizer.eos_token_id:
                logging.info("Early stop triggered.")
                break

            generated_sequence.append(selected_token_id.item())

            inputs['input_ids'] = torch.cat([inputs['input_ids'], selected_token_id.unsqueeze(0).unsqueeze(0).to(inputs['input_ids'].device)], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([[1]], device=self.model.device)], dim=1)

            step += 1
           
            
        print(similarity_list)

# torch.where((p1 >= threshold) | (p2 >= threshold))[0].cpu().tolist()

        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence, skip_special_tokens=True), len(generated_sequence),  similarity_list




class Dynamic_alpha():
    def __init__(self, alpha_base_val = 10.0, gamma_val = 10.0, beta_val = 0.05, tokenizer = None):
        self.alpha_base_val = alpha_base_val
        self.gamma_val = gamma_val
        self.beta_val = beta_val
        self.tokenizer = tokenizer



        self.safety_words = [
    "I", "apologize", "apologise", "cannot", "can't", "illegal", "unethical",
    "fulfiil", "AI", "Sorry", "sorry", "am", "'m", "'t", "'d", "'s", "As", "an",
    "assistant", "Assistant", "unable", "able", "allowed",  "not",
    "do", "don't", "must", "Sure", "sure", "can", "provide", "Here", "but",
    "Unfortunately", "Creating", "Hacking", "Stealing", "Developing", "Generating",
    "Producing", "To", "develop", "access", "perform", "offer", "supply", "give",
    "create", "generate", "write", "produce", "assist", "help", "support",
    "complete", ",", "'", "understand", "your", "My", "with", "harm", "harmful",
    "purpose", "positive", "apologies", "legal", "engage", "prefer", "recommend",
    "promote", "will", "would"
                "I'm sorry", 
                "Sorry",
                "sorry",
                "I am sorry",
                "I apologize",
                "As an",
                "As a language model",
                "As an Assistant",
                "I cannot",
                "cannot fulfill",
                "can’t assist with",
                "can't assist with",
                "can’t do that",
                "can't do that",
                "can't provide",
                "can’t provide",
                "can't fulfill",
                "can’t fulfill",
                "can't help with",
                "can’t help with",
                "can’t support",
                "can't support",
                "I'm unable to",
                "I am unable to",
                "I am not allowed to",
                "I am not able to",
                "I am not authorized",
                "I can't",
                "I can’t",
                "I can\'t",

        ]

        self.safety_token_ids = self.get_safety_token_ids()
    # --------------------------------------------------------------------------
    # 辅助函数 (Dynamic Alpha Calculation - Input param name changed)
    # --------------------------------------------------------------------------

    def _calculate_dynamic_alpha(
        self,
        distance_metric: torch.Tensor, # Changed param name from kl_divergence
        k: int,
        verbose: False
    ) -> torch.Tensor:
        """
        计算动态修正系数 alpha_k based on a distance metric (like TVD or KL).

        Args:
            distance_metric (torch.Tensor): The calculated distance (e.g., TVD) between p1 and p2 distributions.
            k (int): 当前是生成的第几个token (从1开始计数)。
            alpha_base (float): 基准修正系数。
            gamma (float): Distance metric's influence on w_diff sensitivity.
                        NOTE: Optimal gamma might differ between KL and TVD.
            beta (float): 步数衰减权重w_decay的衰减率。

        Returns:
            torch.Tensor: 动态系数 alpha_k (标量Tensor)。
        """

        w_diff = 1.0 - torch.exp(-self.gamma_val * distance_metric)
        if verbose:
            logging.info(f"w_diff:{w_diff}")
        if k < 1:
            raise ValueError("Token step k must be >= 1.")
        w_decay = torch.exp(-self.beta_val * torch.tensor(k - 1, dtype=torch.float32, device=distance_metric.device))
        # alpha_k = alpha_base * w_diff * w_decay
        alpha_k = self.alpha_base_val * w_diff * w_decay
        return alpha_k
    

    def get_safety_token_ids(self,):
        safety_tokens = []
        for word in self.safety_words:
            # if self.tokenizer.convert_tokens_to_ids(word):
            #     safety_tokens.append(self.tokenizer.convert_tokens_to_ids(word))
            # else:
            safety_tokens+=self.tokenizer.encode(word, add_special_tokens=False)
        return set(safety_tokens)
    # --------------------------------------------------------------------------
    # 主修正函数 (Updated to use TVD)
    # --------------------------------------------------------------------------

    def get_alpha(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        k: int,
        verbose: bool ,
        threshold: float = 0.05,
    ) -> torch.Tensor:
        """
        使用方案A（加法修正+归一化）修正大模型p3的概率分布, using TVD as the distance metric.

        Args:
            p1 (torch.Tensor): 小模型1 (安全) 概率分布。
            p2 (torch.Tensor): 小模型2 (非安全) 概率分布。
            k (int): 当前生成token的步数 (从1开始)。
            alpha_base (float): 基准修正系数。
            gamma (float): TVD影响权重w_diff的敏感度参数 (see note).
            beta (float): 步数衰减权重w_decay的衰减率。
            epsilon (float): Small constant for safe normalization sum.

        Returns:
            torch.Tensor: 修正后的概率分布p3' (1D Tensor, size=vocab_size)。
        """


        # Step 1: Find common tokens and search range
        # 获取概率 ≥ threshold 的 token 索引
        selected_indices = list(set(torch.where((p1 >= threshold) | (p2 >= threshold))[0].cpu().tolist()) & self.safety_token_ids)
        # selected_indices = torch.where((p1 >= threshold) | (p2 >= threshold))[0]
        if verbose:
            logging.info(f"selected_indices：\n{selected_indices}")
        # Calculate Wasserstein
        wasserstein_dist = torch.sum(torch.abs(p1[selected_indices] - p2[selected_indices]))
        if verbose:
            logging.info(f"wasserstein_dist:{wasserstein_dist}")
        
        # Step 3: Calculate dynamic alpha using TVD <<< CHANGE HERE
        alpha_k = self._calculate_dynamic_alpha(wasserstein_dist, k, verbose)
        return alpha_k



# import torch

# def speculative_greedy_decode(
#     main_model,
#     draft_model,
#     tokenizer,
#     input_ids,
#     gamma=4,
#     max_new_tokens=100,
#     device="cuda"
# ):
#     """
#     贪婪+投机解码（Speculative Greedy Decoding with KV cache and EOS check）
#     - main_model: 主模型（带KV缓存解码）
#     - draft_model: 草稿小模型（同主模型接口）
#     - tokenizer: 大多数HF分词器
#     - input_ids: shape=(1, seq_len)，编码过的输入
#     - gamma: 每次投机的token数量
#     - max_new_tokens: 最大生成token数量
#     - device: 推理设备
#     """

#     eos_token_id = tokenizer.eos_token_id
#     cur_ids = input_ids.to(device)                  # 当前已生成的全部token
#     generated = []                                  # 新生成部分（不包括输入）
#     main_kv_cache = None                            # KV缓存初始化

#     for _ in range(max_new_tokens):
#         # Step1: 草稿模型投机gamma步（遇eos提前停）
#         draft_ids = []
#         kv_cache = None
#         draft_input = cur_ids
#         for _ in range(gamma):
#             with torch.no_grad():
#                 draft_out = draft_model(
#                     draft_input, past_key_values=kv_cache, use_cache=True
#                 )
#             next_logits = draft_out.logits[:, -1, :]   # 取最后一步
#             next_token = next_logits.argmax(-1, keepdim=True)
#             draft_ids.append(next_token)
#             if next_token.item() == eos_token_id:
#                 break
#             kv_cache = draft_out.past_key_values
#             draft_input = next_token

#         # Step2: 主模型验证草稿。遇不一致或eos，立即终止
#         verify_input = cur_ids
#         for token in draft_ids:
#             with torch.no_grad():
#                 main_out = main_model(
#                     verify_input, past_key_values=main_kv_cache, use_cache=True
#                 )
#             pred_token = main_out.logits[:, -1, :].argmax(-1, keepdim=True)
#             if pred_token.item() == token.item():
#                 # 验证通过，纳入生成
#                 generated.append(token.item())
#                 cur_ids = torch.cat([cur_ids, token], dim=-1)
#                 main_kv_cache = main_out.past_key_values
#                 verify_input = token
#                 if token.item() == eos_token_id:
#                     # draft与main_model都认为结束，直接终止
#                     return tokenizer.decode(
#                         generated, skip_special_tokens=True
#                     )
#                 if len(generated) >= max_new_tokens:
#                     return tokenizer.decode(
#                         generated, skip_special_tokens=True
#                     )
#             else:
#                 # 未通过，用主模型自己的预测token，并中断草稿
#                 generated.append(pred_token.item())
#                 cur_ids = torch.cat([cur_ids, pred_token], dim=-1)
#                 main_kv_cache = main_out.past_key_values
#                 if pred_token.item() == eos_token_id:
#                     return tokenizer.decode(
#                         generated, skip_special_tokens=True
#                     )
#                 break
#         if len(generated) >= max_new_tokens:
#             break

#     return tokenizer.decode(generated, skip_special_tokens=True)

# # 用法示例
# # input_text = "Once upon a time,"
# # input_ids = tokenizer.encode(input_text, return_tensors="pt")
# # result = speculative_greedy_decode(main_model, draft_model, tokenizer, input_ids, gamma=4, max_new_tokens=100, device="cuda")
# # print(result)


def rollback(past_key_values, end_pos ):
    past_key_values_trimmed = []
    for kv in past_key_values:
        k, v = kv

        # k, v (batch, head, seq, hidden_dim)
        k = k[:, :, :end_pos, :]
        v = v[:, :, :end_pos, :]
        kv_trimmed = (k, v)
        past_key_values_trimmed.append(kv_trimmed)
    
    return past_key_values_trimmed
    