import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import add_lora_gptneox, count_trainable
from tqdm.auto import tqdm
from torch import amp
import numpy as np

MODEL_ID = "EleutherAI/pythia-1.4b"
SFT_WEIGHTS_PATH = "/kaggle/input/hw4-ex1/lora_sft_pythia1_4b.pt"
OUTPUT_DIR = "/kaggle/working/dpo-pythia-1.4b_2"
MAX_LEN = 256
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 16
LR = 5e-6
DPO_BETA = 0.05
LOG_EVERY = 10
TRAIN_SPLIT = "train[:5000]"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def split_prompt_answer(text: str):
    markers = ["Assistant:", "Human:"]
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

raw_ds = load_dataset("Anthropic/hh-rlhf", split=TRAIN_SPLIT)

def has_valid_pairs(example):
    chosen_valid = split_prompt_answer(example["chosen"]) is not None
    rejected_valid = split_prompt_answer(example["rejected"]) is not None
    return chosen_valid and rejected_valid

raw_ds = raw_ds.filter(has_valid_pairs)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token

def encode_dpo_pair(ex):
    chosen_prompt, chosen_answer = split_prompt_answer(ex["chosen"])
    rejected_prompt, rejected_answer = split_prompt_answer(ex["rejected"])
    
    if chosen_prompt != rejected_prompt:
        rejected_prompt = chosen_prompt
    
    prompt_enc = tok(chosen_prompt, truncation=True, max_length=MAX_LEN//3, 
                    add_special_tokens=False)
    prompt_ids = prompt_enc["input_ids"]
    
    def encode_answer(answer):
        max_answer_len = MAX_LEN - len(prompt_ids) - 10
        answer_enc = tok(answer + tok.eos_token, truncation=True, 
                        max_length=max_answer_len, 
                        add_special_tokens=False)
        return answer_enc["input_ids"]
    
    chosen_answer_ids = encode_answer(chosen_answer)
    rejected_answer_ids = encode_answer(rejected_answer)
    
    def create_full_seq(answer_ids):
        full_ids = prompt_ids + answer_ids
        
        if len(full_ids) < MAX_LEN:
            pad_len = MAX_LEN - len(full_ids)
            full_ids = full_ids + [tok.pad_token_id] * pad_len
        else:
            full_ids = full_ids[:MAX_LEN]
            
        return full_ids
    
    chosen_ids = create_full_seq(chosen_answer_ids)
    rejected_ids = create_full_seq(rejected_answer_ids)
    
    def create_loss_mask(prompt_len, answer_len):
        loss_mask = [0] * min(prompt_len, MAX_LEN) + [1] * min(answer_len, MAX_LEN - prompt_len)
        if len(loss_mask) < MAX_LEN:
            loss_mask = loss_mask + [0] * (MAX_LEN - len(loss_mask))
        return loss_mask[:MAX_LEN]
    
    loss_mask_chosen = create_loss_mask(len(prompt_ids), len(chosen_answer_ids))
    loss_mask_rejected = create_loss_mask(len(prompt_ids), len(rejected_answer_ids))
    
    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "loss_mask_chosen": loss_mask_chosen,
        "loss_mask_rejected": loss_mask_rejected,
        "prompt_len": len(prompt_ids)
    }

dpo_ds = raw_ds.map(encode_dpo_pair, remove_columns=raw_ds.column_names, batched=False)
dpo_ds = dpo_ds.filter(lambda x: sum(x["loss_mask_chosen"]) > 0 and sum(x["loss_mask_rejected"]) > 0)

train_size = int(0.9 * len(dpo_ds))
train_ds = dpo_ds.select(range(train_size))
valid_ds = dpo_ds.select(range(train_size, len(dpo_ds)))

def collate_dpo(batch):
    input_dict = {}
    for key in batch[0].keys():
        if key == "prompt_len":
            input_dict[key] = torch.tensor([x[key] for x in batch], dtype=torch.long)
        else:
            input_dict[key] = torch.tensor([x[key] for x in batch], dtype=torch.long)
    return input_dict

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, 
    collate_fn=collate_dpo
)

valid_dl = torch.utils.data.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_dpo
)

def get_logps(model, input_ids, loss_mask):
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    
    attention_mask = (inputs != tok.pad_token_id).long()
    
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    labels = labels.unsqueeze(-1)
    token_logps = torch.gather(log_probs, -1, labels).squeeze(-1)
    
    shift_loss_mask = loss_mask[:, 1:]
    
    if token_logps.shape != shift_loss_mask.shape:
        shift_loss_mask = shift_loss_mask[:, :token_logps.shape[1]]
    
    masked_logps = (token_logps * shift_loss_mask).sum(dim=-1)
    
    return masked_logps

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.05):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = beta * (policy_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    
    ratio_penalty = torch.abs(policy_logratios).mean() * 0.01
    
    return losses.mean() + ratio_penalty

policy_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
policy_model = add_lora_gptneox(policy_model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)

if os.path.exists(SFT_WEIGHTS_PATH):
    sft_state = torch.load(SFT_WEIGHTS_PATH, map_location="cpu")
    policy_model.load_state_dict(sft_state, strict=False)

policy_model.gradient_checkpointing_enable()

for name, param in policy_model.named_parameters():
    if '.A.' in name or '.B.' in name:
        param.requires_grad = True

policy_model = policy_model.to("cuda")
policy_model.train()

reference_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
reference_model = add_lora_gptneox(reference_model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)

if os.path.exists(SFT_WEIGHTS_PATH):
    reference_model.load_state_dict(sft_state, strict=False)

for param in reference_model.parameters():
    param.requires_grad = False

reference_model.eval()

trainable_params = [p for p in policy_model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
scaler = amp.GradScaler('cuda')

global_step = 0

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    policy_model.train()
    epoch_loss = 0
    epoch_acc = 0
    step_count = 0
    
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in enumerate(pbar):
        torch.cuda.empty_cache()
        
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        try:
            with amp.autocast('cuda', dtype=torch.float16):
                policy_chosen_logps = get_logps(
                    policy_model, 
                    batch["chosen_ids"], 
                    batch["loss_mask_chosen"]
                )
                
                policy_rejected_logps = get_logps(
                    policy_model,
                    batch["rejected_ids"],
                    batch["loss_mask_rejected"]
                )
                
                with torch.no_grad():
                    chosen_ids_cpu = batch["chosen_ids"].cpu()
                    rejected_ids_cpu = batch["rejected_ids"].cpu()
                    loss_mask_chosen_cpu = batch["loss_mask_chosen"].cpu()
                    loss_mask_rejected_cpu = batch["loss_mask_rejected"].cpu()
                    
                    ref_chosen_logps = get_logps(
                        reference_model,
                        chosen_ids_cpu,
                        loss_mask_chosen_cpu
                    ).to("cuda")
                    
                    ref_rejected_logps = get_logps(
                        reference_model,
                        rejected_ids_cpu, 
                        loss_mask_rejected_cpu
                    ).to("cuda")
                
                loss = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=DPO_BETA
                )
            
            scaler.scale(loss / GRAD_ACCUM).backward()
            
            epoch_loss += loss.item()
            accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean().item()
            epoch_acc += accuracy
            step_count += 1
            
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)
            
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
                
                if global_step % LOG_EVERY == 0:
                    avg_loss = epoch_loss / step_count
                    avg_acc = epoch_acc / step_count
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "acc": f"{avg_acc:.3f}",
                        "step": global_step
                    })
            
            del policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
            torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                continue
    
    policy_model.eval()
    val_loss = 0
    val_steps = 0
    val_acc = 0
    
    with torch.no_grad():
        for val_batch in tqdm(valid_dl, desc="Validation"):
            try:
                torch.cuda.empty_cache()
                val_batch = {k: v.to("cuda") for k, v in val_batch.items()}
                
                with amp.autocast('cuda', dtype=torch.float16):
                    policy_chosen_logps = get_logps(
                        policy_model, 
                        val_batch["chosen_ids"],
                        val_batch["loss_mask_chosen"]
                    )
                    
                    policy_rejected_logps = get_logps(
                        policy_model,
                        val_batch["rejected_ids"],
                        val_batch["loss_mask_rejected"]
                    )
                    
                    chosen_ids_cpu = val_batch["chosen_ids"].cpu()
                    rejected_ids_cpu = val_batch["rejected_ids"].cpu()
                    loss_mask_chosen_cpu = val_batch["loss_mask_chosen"].cpu()
                    loss_mask_rejected_cpu = val_batch["loss_mask_rejected"].cpu()
                    
                    ref_chosen_logps = get_logps(
                        reference_model,
                        chosen_ids_cpu,
                        loss_mask_chosen_cpu
                    ).to("cuda")
                    
                    ref_rejected_logps = get_logps(
                        reference_model,
                        rejected_ids_cpu,
                        loss_mask_rejected_cpu
                    ).to("cuda")
                    
                    loss = dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps, beta=DPO_BETA
                    )
                    
                    accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean().item()
                    
                val_loss += loss.item()
                val_acc += accuracy
                val_steps += 1
                
                val_losses.append(loss.item())
                val_accuracies.append(accuracy)
                
                del policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    continue
    
    if val_steps > 0:
        avg_val_loss = val_loss / val_steps
        avg_val_acc = val_acc / val_steps
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.3f}")
    
    checkpoint_state = {}
    for name, param in policy_model.state_dict().items():
        if '.A.' in name or '.B.' in name or 'lora' in name.lower():
            checkpoint_state[name] = param.cpu()
    
    torch.save(checkpoint_state, os.path.join(OUTPUT_DIR, f"lora_dpo_epoch_{epoch+1}.pt"))

dpo_state = {}
for name, param in policy_model.state_dict().items():
    if '.A.' in name or '.B.' in name or 'lora' in name.lower():
        dpo_state[name] = param.cpu()

torch.save(dpo_state, os.path.join(OUTPUT_DIR, "lora_dpo_pythia1_4b.pt"))