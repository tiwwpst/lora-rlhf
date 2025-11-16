import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import add_lora_gptneox, count_trainable, LoRALinear
from tqdm.auto import tqdm
from torch import amp

MODEL_ID = "EleutherAI/pythia-1.4b"
OUTPUT_DIR = "/kaggle/working/sft-lora-pythia-1.4b"
MAX_LEN = 256
EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 1e-4
LOG_EVERY = 50
EVAL_EXAMPLES = ["Explain the difference between supervised and unsupervised learning.", "Give me 3 tips to stay focused while studying."]
TRAIN_SPLIT = "train[:20000]"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_prompt_answer(chosen_text: str):
    marker = "Assistant:"
    last = chosen_text.rfind(marker)
    if last == -1:
        return None
    prompt = chosen_text[:last + len(marker)].strip()
    answer = chosen_text[last + len(marker):].strip()
    if not answer:
        return None
    return prompt, answer

print("Загрузка датасета:", TRAIN_SPLIT)
raw = load_dataset("Anthropic/hh-rlhf", split=TRAIN_SPLIT)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token

def encode_example(ex):
    pa = split_prompt_answer(ex["chosen"])
    if pa is None:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    prompt, answer = pa

    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    answer_ids = tok(answer + tok.eos_token, add_special_tokens=False).input_ids

    if len(answer_ids) >= MAX_LEN - 1:
        answer_ids = answer_ids[:MAX_LEN - 1]
    remain = MAX_LEN - len(answer_ids)
    if len(prompt_ids) > remain:
        prompt_ids = prompt_ids[-remain:]

    input_ids = prompt_ids + answer_ids
    attn = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + answer_ids[:]

    if len(input_ids) < MAX_LEN:
        pad = MAX_LEN - len(input_ids)
        input_ids += [tok.pad_token_id] * pad
        attn += [0] * pad
        labels += [-100] * pad

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

print("Кодировка примеров:")
ds = raw.map(encode_example, remove_columns=raw.column_names)
ds = ds.filter(lambda ex: sum(ex["attention_mask"]) > 0)
print("Размер обучающей выборки:", len(ds))

VALID_N = min(2000, len(ds) // 20)
if VALID_N > 0:
    train_ds = ds.select(range(len(ds) - VALID_N))
    valid_ds = ds.select(range(len(ds) - VALID_N, len(ds)))
else:
    train_ds, valid_ds = ds, None

def collate(batch):
    import numpy as np
    input_ids = torch.tensor(np.stack([x["input_ids"] for x in batch]), dtype=torch.long)
    attention_mask = torch.tensor(np.stack([x["attention_mask"] for x in batch]), dtype=torch.long)
    labels = torch.tensor(np.stack([x["labels"] for x in batch]), dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

dl = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    collate_fn=collate, num_workers=2, pin_memory=True, persistent_workers=True
)
if valid_ds is not None:
    dl_val = torch.utils.data.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        collate_fn=collate, num_workers=2, pin_memory=True, persistent_workers=True
    )

print("Загрузка модели:", MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
for p in model.parameters():
    p.requires_grad = False
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
model.config.use_cache = False

model = add_lora_gptneox(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
model = model.to("cuda", dtype=torch.float32)

trainable, total = count_trainable(model)
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

lora_params = [p for n, p in model.named_parameters()
               if p.requires_grad and (".A." in n or ".B." in n
                                       or n.endswith("A.weight") or n.endswith("B.weight"))]
assert len(lora_params) > 0, "LoRA-параметры не найдены"

opt = torch.optim.AdamW(lora_params, lr=LR, fused=False)
    
scaler = amp.GradScaler('cuda')

device = next(model.parameters()).device
model.train()
global_step = 0
running_loss = 0.0

print("Обучение:")
for epoch in range(EPOCHS):
    pbar = tqdm(enumerate(dl), total=len(dl), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    for step, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with amp.autocast('cuda', dtype=torch.float16):
            out = model(**batch)
            loss = out.loss

        loss_to_backprop = loss / GRAD_ACCUM
        scaler.scale(loss_to_backprop).backward()
        running_loss += loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 5 == 0:
                pbar.set_postfix({"loss": f"{running_loss / max(1, min(LOG_EVERY, step+1)): .4f}"})
            if global_step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY
                tqdm.write(f"step {global_step:>6} | loss {avg:.4f}")
                running_loss = 0.0


    if valid_ds is not None:
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for vb in tqdm(dl_val, total=len(dl_val), desc="Валидация", leave=False):
                vb = {k: v.to(device) for k, v in vb.items()}
                vout = model(**vb)
                val_loss += vout.loss.item()
                val_steps += 1
        print(f"[epoch {epoch+1}] val_loss: {val_loss/val_steps:.4f}")
        model.train()

def generate_answer(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    if not prompt.strip().startswith("Human:"):
        prompt = f"Human: {prompt.strip()}\n\nAssistant:"
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    last = text.rfind("Assistant:")
    return text[last + len("Assistant:"):].strip()

print("Примеры ответов:")
model.eval()
for q in tqdm(EVAL_EXAMPLES, desc="Генерация примеров"):
    a = generate_answer(q)[:600]
    print("Q:", q, "\nA:", a, "\n", "-"*60)

state = {k: v.cpu() for k, v in model.state_dict().items()
         if (".A." in k or ".B." in k or k.endswith("A.weight") or k.endswith("B.weight"))}
save_path = os.path.join(OUTPUT_DIR, "lora_sft_pythia1_4b.pt")
torch.save(state, save_path)