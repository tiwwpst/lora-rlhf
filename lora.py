import math
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        device = base_linear.weight.device
        dtype  = base_linear.weight.dtype
        self.A = nn.Linear(self.in_features, r, bias=False, device=device, dtype=dtype)
        self.B = nn.Linear(r, self.out_features, bias=False, device=device, dtype=dtype)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        self.bias = self.base.bias

    def forward(self, x):
        y = self.base(x)
        delta = self.B(self.drop(self.A(x))) * self.scaling
        return y + delta

def add_lora_gptneox(model: nn.Module, r=8, alpha=16, dropout=0.05,
                     target_suffixes=("attention.query_key_value",
                                      "attention.dense",
                                      "mlp.dense_h_to_4h",
                                      "mlp.dense_4h_to_h")):
    
    name_to_module = dict(model.named_modules())
    replaced = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(sfx) for sfx in target_suffixes):
            continue

        parent_name, attr = name.rsplit('.', 1)
        parent = name_to_module[parent_name]
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)

    print(f"LoRA вставлена в {len(replaced)} слоёв:")
    for n in replaced: print("  -", n)
    return model

def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return t, a
