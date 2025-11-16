import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import add_lora_gptneox
import os
from datasets import load_dataset

def load_model_with_lora(model_path, lora_weights_path=None):
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    if lora_weights_path and os.path.exists(lora_weights_path):
        model = add_lora_gptneox(model, r=8, alpha=16, dropout=0.05)
        lora_weights = torch.load(lora_weights_path, map_location="cpu")
        model.load_state_dict(lora_weights, strict=False)
    
    model = model.to("cuda")
    model.eval()
    return tok, model

def split_prompt_answer(text: str):
    last_assistant = text.rfind("Assistant:")
    if last_assistant == -1:
        return None
    
    last_human = text.rfind("Human:", 0, last_assistant)
    if last_human == -1:
        return None
    
    prompt = text[last_human:last_assistant + len("Assistant:")].strip()
    answer = text[last_assistant + len("Assistant:"):].strip()
    
    if not answer or len(answer) < 10:
        return None
    
    return prompt, answer

def get_comparison_examples(num_examples=10):
    test_ds = load_dataset("Anthropic/hh-rlhf", split="test")
    
    examples = []
    for example in test_ds:
        if len(examples) >= num_examples:
            break
            
        chosen_pa = split_prompt_answer(example["chosen"])
        rejected_pa = split_prompt_answer(example["rejected"])
        
        if chosen_pa is not None and rejected_pa is not None:
            prompt, chosen_answer = chosen_pa
            _, rejected_answer = rejected_pa
            
            examples.append({
                "prompt": prompt,
                "chosen": chosen_answer,
                "rejected": rejected_answer
            })
    
    return examples

def generate_response(tok, model, prompt, max_length=300):
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tok.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        return response.split("Assistant:")[-1].strip()
    return response

def compare_all_models():
    MODEL_ID = "EleutherAI/pythia-1.4b"
    SFT_WEIGHTS = "/kaggle/input/hw4-ex1/lora_sft_pythia1_4b.pt"
    DPO_WEIGHTS = "/kaggle/working/dpo-pythia-1.4b_2/lora_dpo_epoch_1.pt"
    
    ref_tok, ref_model = load_model_with_lora(MODEL_ID, SFT_WEIGHTS)
    
    policy_tok, policy_model = load_model_with_lora(MODEL_ID, DPO_WEIGHTS)
    
    examples = get_comparison_examples(5)
    
    for i, example in enumerate(examples):
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        print(f"--- Тест {i+1} ---")
        
        print(f"\nПРОМПТ:")
        print(f"{prompt[-200:]}...")
        
        print(f"\nCHOSEN:")
        print(f"{chosen[:200]}...")
        
        print(f"\nREJECTED:")
        print(f"{rejected[:200]}...")
        
        print(f"\nREFERENCE:")
        ref_answer = generate_response(ref_tok, ref_model, prompt)
        print(f"{ref_answer[:200]}...")
        
        print(f"\nPOLICY:")
        policy_answer = generate_response(policy_tok, policy_model, prompt)
        print(f"{policy_answer[:200]}...")
        
if __name__ == "__main__":
    compare_all_models()