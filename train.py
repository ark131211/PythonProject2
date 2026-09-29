"""Обучение MathNet с нуля на сгенерированных задачах.

    python train.py --steps 15000
    python train.py --resume --steps 10000   # дообучить с сохранённых весов

Результат: weights/mathnet.npz (+ .json с конфигом) — их читает mathnet/model.py.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mathnet.data import BLOCK_SIZE, EOS, PAD, SEP, STOI, TASKS, VOCAB_SIZE, decode, encode, sample


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.fc2(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class TorchGPT(nn.Module):
    def __init__(self, d=256, n_layer=6, n_head=8):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos_emb = nn.Parameter(torch.zeros(BLOCK_SIZE, d))
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.apply(self._init)
        nn.init.normal_(self.pos_emb, std=0.02)
        for name, p in self.named_parameters():  # масштабирование как в GPT-2
            if name.endswith(("proj.weight", "fc2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ self.tok_emb.weight.T


def make_batch(bs, device):
    x = torch.full((bs, BLOCK_SIZE), STOI[PAD], dtype=torch.long)
    y = torch.full((bs, BLOCK_SIZE), -100, dtype=torch.long)
    for i in range(bs):
        q, a = sample()
        ids = encode(q + SEP + a + EOS)
        x[i, : len(ids)] = torch.tensor(ids)
        # учим предсказывать только решение: цели начинаются с символа после "?"
        n_q = len(q) + 1
        y[i, n_q - 1: len(ids) - 1] = torch.tensor(ids[n_q:])
    L = int((y != -100).nonzero()[:, 1].max()) + 1  # обрезаем паддинг справа
    return x[:, :L].to(device), y[:, :L].to(device)


@torch.no_grad()
def evaluate(model, device, n=200):
    """Точность «ответ совпал целиком» по каждому типу задач, жадная генерация."""
    model.eval()
    res = {}
    for task in TASKS:
        probs = [sample(task) for _ in range(n)]
        x = torch.full((n, BLOCK_SIZE), STOI[PAD], dtype=torch.long)
        pos = torch.zeros(n, dtype=torch.long)
        for i, (q, _) in enumerate(probs):
            ids = encode(q + SEP)
            x[i, : len(ids)] = torch.tensor(ids)
            pos[i] = len(ids)
        x, pos = x.to(device), pos.to(device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        ar = torch.arange(n, device=device)
        while not done.all() and pos.max() < BLOCK_SIZE:
            logits = model(x)[ar, (pos - 1).clamp(max=BLOCK_SIZE - 1)]
            nxt = logits.argmax(-1)
            active = ~done & (pos < BLOCK_SIZE)
            x[ar[active], pos[active]] = nxt[active]
            pos = pos + active.long()
            done |= (nxt == STOI[EOS]) | (pos >= BLOCK_SIZE)
        x = x.cpu()
        ok = 0
        for i, (q, a) in enumerate(probs):
            out = decode(x[i, len(q) + 1:].tolist()).split(EOS)[0]
            ok += out == a
        res[task] = ok / n
    model.train()
    return res


def load_weights(model, path: Path):
    """Загружает веса; если словарь или контекст выросли — копирует пересекающуюся часть."""
    own = model.state_dict()
    for k, v in np.load(path).items():
        k = "tok_emb.weight" if k == "tok_emb" else k
        v = torch.from_numpy(v.astype(np.float32))
        if v.shape != own[k].shape:
            print(f"  {k}: {tuple(v.shape)} -> {tuple(own[k].shape)} (новые строки — случайные)")
            own[k][: v.shape[0]] = v
        else:
            own[k] = v
    model.load_state_dict(own)


def export(model, cfg, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k.replace("tok_emb.weight", "tok_emb"): v.detach().cpu().numpy().astype(np.float16)
          for k, v in model.state_dict().items()}
    np.savez_compressed(path, **sd)
    path.with_suffix(".json").write_text(json.dumps(cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--out", default="weights/mathnet.npz")
    ap.add_argument("--resume", action="store_true", help="начать с весов из --out")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    out = Path(args.out)
    if args.resume:
        cfg = json.loads(out.with_suffix(".json").read_text())
    else:
        cfg = {"d": args.d, "n_layer": args.layers, "n_head": args.heads}
    model = TorchGPT(cfg["d"], cfg["n_layer"], cfg["n_head"])
    if args.resume:
        load_weights(model, out)
        print(f"resumed from {out}")
    model = model.to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    warmup = 500

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = make_batch(args.bs, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step):.2e} "
                  f"{(time.time() - t0) / step * 1000:.0f}ms/step", flush=True)
        if step % 2000 == 0 or step == args.steps:
            acc = evaluate(model, device)
            print("  acc:", " ".join(f"{k}={v:.0%}" for k, v in acc.items()), flush=True)
            export(model, cfg, out)
    print(f"saved {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
