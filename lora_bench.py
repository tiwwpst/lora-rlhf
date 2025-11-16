import time, argparse, numpy as np, torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import add_lora_gptneox, count_trainable, LoRALinear

def make_alpaca_ds(tok, subset=1000, max_len=512):
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{subset}]")
    if "text" not in ds.column_names:
        ds = ds.map(lambda ex: {"text": ex["output"]})
    ds = DatasetDict({"train": ds})
    def enc(ex):
        ids = tok(ex["text"], truncation=True, max_length=max_len,
                  padding="max_length", return_tensors=None)
        return ids
    ds = ds.map(enc, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds

def bench(model, dataloader, steps=100, grad_accum=8, lr=1e-4):
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = next(model.parameters()).device
    trainables = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainables, lr=lr)
    tok_count = 0
    t0 = time.perf_counter()

    it = iter(dataloader)
    for step in range(steps):
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels)
        loss = out.loss
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        tok_count += int(batch["attention_mask"].sum().item())

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "steps": steps,
        "sec_per_step": dt/steps,
        "tokens_per_sec": tok_count/dt,
        "peak_mem_MB": round(torch.cuda.max_memory_allocated()/1024**2, 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="EleutherAI/pythia-70m")
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--bench_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r_values", type=str, default="2,4,8,16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.pad_token = tok.eos_token
    ds = make_alpaca_ds(tok, subset=args.subset, max_len=args.max_len)
    dl = torch.utils.data.DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, drop_last=True)

    model_lora = AutoModelForCausalLM.from_pretrained(args.model_id)
    for p in model_lora.parameters(): p.requires_grad = False
    model_lora.gradient_checkpointing_enable(); model_lora.enable_input_require_grads(); model_lora.config.use_cache = False
    model_lora = add_lora_gptneox(model_lora, r=8, alpha=16, dropout=0.05)
    model_lora = model_lora.to("cuda", dtype=torch.float32)
    t_lora, a_lora = count_trainable(model_lora)
    print(f"[LoRA r=8] trainable: {t_lora:,}/{a_lora:,} ({100*t_lora/a_lora:.2f}%)")
    res_lora = bench(model_lora, dl, steps=args.bench_steps, lr=args.lr)
    print("1) LoRA bench:", res_lora)

    model_full = AutoModelForCausalLM.from_pretrained(args.model_id)
    model_full.config.use_cache = False
    model_full = model_full.to("cuda", dtype=torch.float32)
    t_full, a_full = count_trainable(model_full)
    print(f"[Full FT] trainable: {t_full:,}/{a_full:,} ({100*t_full/a_full:.2f}%)")
    res_full = bench(model_full, dl, steps=max(50, args.bench_steps//2), lr=args.lr)
    print("2) (LoRA, Full):", (res_lora, res_full))

    r_vals = [int(x) for x in args.r_values.split(",")]
    rows = []
    for r in r_vals:
        m = AutoModelForCausalLM.from_pretrained(args.model_id)
        for p in m.parameters(): p.requires_grad = False
        m = add_lora_gptneox(m, r=r, alpha=2*r, dropout=0.05)
        t,a = count_trainable(m)
        rows.append({"r": r, "alpha": 2*r, "trainable": t, "total": a, "pct": 100*t/a})
    print("3) r-sweep:", rows)

if __name__ == "__main__":
    main()
