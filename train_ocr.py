"""Обучение OCR-сети с нуля на синтетических картинках задач.

    python train_ocr.py --steps 20000

Результат: weights/ocr.npz (+ .json) — их читает mathnet/ocr.py.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from mathnet.ocr import OCR_CHARS, OCRNet, ctc_greedy, prepare
from mathnet.ocr_data import HAND_FONTS, TRAIN_FONTS, VAL_FONTS, ocr_sample, usable_fonts

CHANNELS = [48, 96, 128, 128, 192, 192]
POOLS = [(2, 2), (2, 2), (1, 1), (2, 1), (2, 1), (2, 1)]  # 32x320 -> 1x80
SEQ = 256
N_CLASSES = len(OCR_CHARS) + 1
C2I = {c: i + 1 for i, c in enumerate(OCR_CHARS)}


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        c_in = 1
        for c in CHANNELS:
            self.convs.append(nn.Conv2d(c_in, c, 3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(c))
            c_in = c
        self.seq = nn.ModuleList([nn.Conv1d(c_in, SEQ, 5, padding=2), nn.Conv1d(SEQ, SEQ, 5, padding=2)])
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(SEQ, N_CLASSES)

    def forward(self, x):  # (B,1,32,320) -> (B,T,classes)
        for conv, bn, (ph, pw) in zip(self.convs, self.bns, POOLS):
            x = F.relu(bn(conv(x)))
            if ph > 1 or pw > 1:
                x = F.max_pool2d(x, (ph, pw))
        x = x.squeeze(2)
        for s in self.seq:
            x = self.drop(F.relu(s(x)))
        return self.head(x.transpose(1, 2))


MODES = {"print": 0.45, "hand": 0.40, "handfont": 0.15}


class Synth(IterableDataset):
    def __init__(self, fonts, hand_fonts):
        self.fonts = {"print": fonts, "hand": None, "handfont": hand_fonts}

    def __iter__(self):
        while True:
            mode = random.choices(list(MODES), list(MODES.values()))[0]
            img, label = ocr_sample(self.fonts[mode], mode)
            yield torch.from_numpy(prepare(img))[None], label


def collate(batch):
    xs, labels = zip(*batch)
    targets = torch.tensor([C2I[c] for l in labels for c in l], dtype=torch.long)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return torch.stack(xs), targets, lengths, labels


def seed_worker(i):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def make_val(fonts, n, seed, mode="print"):
    random.seed(seed)
    np.random.seed(seed)
    items = [ocr_sample(fonts, mode, val=True) for _ in range(n)]
    return torch.from_numpy(np.stack([prepare(i) for i, _ in items]))[:, None], [l for _, l in items]


@torch.no_grad()
def accuracy(model, device, x, labels):
    model.eval()
    logits = torch.cat([model(x[i:i + 256].to(device)).float().cpu() for i in range(0, len(x), 256)])
    model.train()
    return sum(ctc_greedy(l.numpy())[0] == y for l, y in zip(logits, labels)) / len(labels)


def export(model, path: Path):
    """Сворачиваем BatchNorm в свёртки и сохраняем в формате mathnet/ocr.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {}
    for i, (conv, bn) in enumerate(zip(model.convs, model.bns)):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        sd[f"conv{i}.weight"] = conv.weight * scale[:, None, None, None]
        sd[f"conv{i}.bias"] = bn.bias - bn.running_mean * scale
    for i, s in enumerate(model.seq):
        sd[f"seq{i}.weight"], sd[f"seq{i}.bias"] = s.weight, s.bias
    sd["head.weight"], sd["head.bias"] = model.head.weight, model.head.bias
    np.savez_compressed(path, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in sd.items()})
    path.with_suffix(".json").write_text(json.dumps({"pools": POOLS}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="weights/ocr.npz")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    train_fonts, val_fonts = usable_fonts(TRAIN_FONTS), usable_fonts(VAL_FONTS)
    hand_fonts = usable_fonts(HAND_FONTS, need="0123456789+=()x")
    print(f"fonts: train={len(train_fonts)} val={len(val_fonts)} hand={len(hand_fonts)}")
    val_unseen = make_val(val_fonts, 1000, seed=2)
    val_hand = make_val(None, 1000, seed=3, mode="hand")  # почерк людей, которых нет в обучении

    torch.manual_seed(0)
    model = CRNN().to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = 300

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))

    loader = DataLoader(Synth(train_fonts, hand_fonts), batch_size=args.bs, num_workers=args.workers,
                        collate_fn=collate, worker_init_fn=seed_worker, persistent_workers=True,
                        prefetch_factor=4)
    t0, out = time.time(), Path(args.out)
    for step, (x, targets, lengths, _) in enumerate(loader, start=1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        logits = model(x.to(device))
        # CTC-loss на MPS не реализован — считаем на CPU, градиент проходит через .cpu()
        logp = F.log_softmax(logits.float(), -1).transpose(0, 1).cpu()
        in_len = torch.full((x.shape[0],), logp.shape[0], dtype=torch.long)
        loss = F.ctc_loss(logp, targets, in_len, lengths, blank=0, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step):.2e} "
                  f"{(time.time() - t0) / step * 1000:.0f}ms/step", flush=True)
        if step % 1500 == 0 or step == args.steps:
            a1 = accuracy(model, device, *val_unseen)
            a2 = accuracy(model, device, *val_hand)
            print(f"  acc: print_unseen_fonts={a1:.1%} handwriting_unseen_writers={a2:.1%}", flush=True)
            export(model, out)
        if step >= args.steps:
            break

    # проверка, что NumPy-версия считает так же, как PyTorch
    net = OCRNet(out)
    model.eval()
    x = val_unseen[0][:4]
    with torch.no_grad():
        ref = model(x.to(device)).float().cpu().numpy()
    diff = max(np.abs(net.logits(x[i, 0].numpy()) - ref[i]).max() for i in range(4))
    print(f"numpy parity max diff {diff:.4f}; saved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
