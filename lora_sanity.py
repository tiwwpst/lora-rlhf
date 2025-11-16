import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict
from lora import add_lora_gptneox, count_trainable, LoRALinear

model_id = "EleutherAI/pythia-70m"
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

for p in model.parameters():
    p.requires_grad = False

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

model = add_lora_gptneox(model, r=8, alpha=16, dropout=0.05)

model = model.to("cuda")
model = model.to(dtype=torch.float32)
device = next(model.parameters()).device
dtype  = next(model.parameters()).dtype
for m in model.modules():
    if isinstance(m, LoRALinear):
        m.A.to(device=device, dtype=dtype).requires_grad_(True)
        m.B.to(device=device, dtype=dtype).requires_grad_(True)

trainable, allp = count_trainable(model)
print(f"Обучаемых параметров: {trainable:,} из {allp:,} ({100*trainable/allp:.2f}% )")

lora_param_names = [n for n,p in model.named_parameters() if p.requires_grad]
print("Пример trainable параметров:", lora_param_names[:6])

ds = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
if "text" not in ds.column_names:
    ds = ds.map(lambda ex: {"text": ex["output"]})
ds = DatasetDict({"train": ds})

def enc(ex):
    ids = tok(ex["text"], truncation=True, max_length=512, padding="max_length", return_tensors=None)
    return ids

ds = ds.map(enc, batched=True, remove_columns=ds["train"].column_names)
ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

dl = torch.utils.data.DataLoader(ds["train"], batch_size=4, shuffle=True, drop_last=True)

lora_params = [p for n,p in model.named_parameters()
               if p.requires_grad and (".A." in n or ".B." in n
                                       or n.endswith("A.weight") or n.endswith("B.weight"))]
assert len(lora_params) > 0, "LoRA-параметры не найдены — проверь add_lora_gptneox()"

opt = torch.optim.AdamW(lora_params, lr=1e-4)

model.train()
for step, batch in enumerate(dl):
    if step == 300:
        break

    batch = {k: v.to(device) for k, v in batch.items()}

    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    out = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels)
    loss = out.loss

    loss.backward()

    if (step + 1) % 8 == 0:
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    if step % 20 == 0:
        print(step, float(loss))
