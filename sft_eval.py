import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import add_lora_gptneox
from datasets import load_dataset

def comprehensive_sft_evaluation():
    MODEL_ID = "EleutherAI/pythia-1.4b"
    SFT_WEIGHTS = "/kaggle/input/hw4-ex1/lora_sft_pythia1_4b.pt"
    
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model = add_lora_gptneox(model, r=8, alpha=16, dropout=0.05)
    
    if os.path.exists(SFT_WEIGHTS):
        sft_state = torch.load(SFT_WEIGHTS, map_location="cpu")
        model.load_state_dict(sft_state, strict=False)
    
    model = model.to("cuda")
    model.eval()
    
    test_ds = load_dataset("Anthropic/hh-rlhf", split="test[:20]")
    
    def split_prompt_answer(text: str):
        last_assistant = text.rfind("Assistant:")
        if last_assistant == -1:
            return None
        prompt = text[:last_assistant + len("Assistant:")].strip()
        answer = text[last_assistant + len("Assistant:"):].strip()
        return prompt, answer if answer else None
    
    print("ОЦЕНКА SFT МОДЕЛИ")
    print("=" * 60)
    
    scores = {
        'relevant': 0,
        'coherent': 0, 
        'helpful': 0,
        'total': 0
    }
    
    for i, example in enumerate(test_ds):
        if i >= 10: 
            break
            
        pa = split_prompt_answer(example["chosen"])
        if pa is None:
            continue
            
        prompt, ground_truth = pa
        scores['total'] += 1
        
        input_ids = tok(prompt, return_tensors="pt", max_length=256, truncation=True).input_ids.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tok.eos_token_id
            )
        
        model_answer = tok.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        print(f"\n--- Тест {i+1} ---")
        print(f"Промпт: {prompt[-150:]}...")
        print(f"Ожидаемый: {ground_truth[:100]}...")
        print(f"Модель: {model_answer[:100]}...")

if __name__ == "__main__":
    comprehensive_sft_evaluation()