import torch
import numpy as np
import copy
import logging
from peft import PeftModel, PeftModelForCausalLM
from utils.string_utils import pad_and_merge

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

    def safedecoding_lora(self, inputs, gen_config=None):
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
                                    return_dict_in_generate=True,
                                    output_scores=True,)
            
            generated_sequence = output_base.sequences[0].tolist()[input_len:]

        # logging.info generated sequence
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

        return self.tokenizer.decode(generated_sequence), len(generated_sequence)


    def secdecoding_lora(self, inputs, gen_config=None, small_inputs = None):
            if gen_config is None:
                gen_config = self.model.generation_config

            max_token_len = gen_config.max_new_tokens
            # max_token_len = 100
            do_sample = gen_config.do_sample

            # Override the generation config for our decoding
            gen_config.max_new_tokens = 1  # We generate one token at a time
            gen_config.do_sample = False  # We use greedy decoding

            generated_sequence = []
            if self.verbose:
                logging.info(f"Generation config: {gen_config}")
            if not small_inputs:
                small_inputs = copy.deepcopy(inputs)

            inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
            small_inputs = {k:v.cuda(self.model.device) for k,v in small_inputs.items()}
            input_len = inputs['input_ids'].shape[1]

            step = 1  # Keep track of generation steps
            # while step <= min(max_token_len, self.first_m):  # Loop until we reach the first m tokens



            alpha_manager = Dynamic_alpha(alpha_base_val=10.0, gamma_val=10.0, beta_val=0.05, tokenizer = self.tokenizer)




            while step <= max_token_len:
                # Generate the next token
                # duplicate inputs for two original and expert model
                
                inputs_duplicated = {k:v.repeat(2,1).cuda(self.small_model.device) for k,v in small_inputs.items()}
                small_outputs = self.small_model.generate(**inputs_duplicated,
                                        adapter_names=self.adapter_names,
                                        generation_config=gen_config,
                                        # pad_token_id=self.tokenizer.pad_token_id,
                                        return_dict_in_generate=True,
                                        output_scores=True,)
                
                base_scores = small_outputs['scores'][0][0]
                expert_scores = small_outputs['scores'][0][1]
                base_prob = torch.softmax(base_scores,dim=-1)
                expert_prob = torch.softmax(expert_scores,dim=-1)

                k = self.top_k
                topk_prob_base, topk_indices_base = base_prob.topk(k) 
                topk_prob_expert, topk_indices_expert = expert_prob.topk(k) 



                outputs = self.model.generate(**inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,)
                model_scores = outputs['scores'][0][0]  

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

                    
                # # Free up memory
                # del output_base, output_expert


            # Use the normal model to generate the rest of the tokens
            # Early stop if the last token is eos


            if inputs['input_ids'][0][-1].item() == self.tokenizer.eos_token_id:
                logging.info("Early stop triggered.")
            else:
                remaining_steps = max_token_len - len(generated_sequence)
                if remaining_steps>0:
                    gen_config.max_new_tokens = remaining_steps
                    gen_config.do_sample = do_sample
                    output_base = self.model.generate(**inputs,
                                            generation_config=gen_config,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            return_dict_in_generate=True,
                                            output_scores=True,)
                
                    generated_sequence = output_base.sequences[0].tolist()[input_len:]

            # logging.info generated sequence
            logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")

            return self.tokenizer.decode(generated_sequence), len(generated_sequence)

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
    
    
    def generate_baseline(self, inputs, adapter_name = ["base"], gen_config=None):
        if gen_config is None:
            gen_config = self.model.generation_config
        
        # gen_config.max_new_tokens = 10
        # gen_config.do_sample = False 
        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}


    
        output_ids = self.model.generate(**inputs,
            generation_config=gen_config,
            pad_token_id=self.tokenizer.pad_token_id)[0]
        generated_sequence = output_ids[inputs["input_ids"].shape[1]:]
        

        # output_base = self.model.generate(**inputs,
        #                     adapter_names=adapter_name,
        #                     generation_config=gen_config,
        #                     pad_token_id=self.tokenizer.pad_token_id,
        #                     return_dict_in_generate=True,
        #                     output_scores=True,)
        
        # generated_sequence = output_base.sequences[0][inputs["input_ids"].shape[1]:]

        
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        
        return self.tokenizer.decode(generated_sequence), len(generated_sequence)




class Dynamic_alpha():
    def __init__(self, alpha_base_val = 10.0, gamma_val = 10.0, beta_val = 0.05, tokenizer = None):
        self.alpha_base_val = alpha_base_val
        self.gamma_val = gamma_val
        self.beta_val = beta_val
        self.tokenizer = tokenizer



        self.safety_words = [
            "I", "apologize", "apologise", "cannot", "can't", "illegal", "unethical",
            "fulfiil", "Sorry", "sorry", "am", "'m", "'t", "'d", "'s", "As", "an",
            "assistant", "Assistant", "unable", "able", "allowed", "It", "is", "not",
            "do", "don't", "must", "Sure", "sure", "can", "provide", "Here", "but",
            "Unfortunately", "Creating", "Hacking", "Stealing", "Developing", "Generating",
            "Producing", "To", "develop", "access", "perform", "offer", "supply", "give",
            "create", "generate", "write", "produce", "assist", "help", "support",
            "complete", ",", "'", "understand", "your", "My", "with", "harmful",
            "purpose", "positive", "apologies", "legal", "engage", "prefer", "recommend",
            "promote", "will", "would",
                "I'm sorry", 
                "Sorry",
                "sorry",
                "I am sorry",
                "I apologize",
                "As an",
                "I'm an",
                "I'm just",
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

        self.safety_token_ids = list(self.get_safety_token_ids())
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
        # selected_indices = torch.where((p1 >= threshold) | (p2 >= threshold))[0]
        selected_indices = self.safety_token_ids
        if verbose:
            logging.info(f"selected_indices：\n{selected_indices}")
        # Calculate Wasserstein
        wasserstein_dist = torch.sum(torch.abs(p1[selected_indices] - p2[selected_indices]))
        if verbose:
            logging.info(f"wasserstein_dist:{wasserstein_dist}")
        
        # Step 3: Calculate dynamic alpha using TVD <<< CHANGE HERE
        alpha_k = self._calculate_dynamic_alpha(wasserstein_dist, k, verbose)
        return alpha_k