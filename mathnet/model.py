"""Трансформер (decoder-only, как GPT) на чистом NumPy — для инференса.

Архитектура повторяет TorchGPT из train.py один в один: эмбеддинги токенов и
позиций, N блоков [LayerNorm -> causal self-attention -> LayerNorm -> MLP(GELU)],
финальный LayerNorm и выходной слой, связанный с эмбеддингами токенов.
"""
import json
from pathlib import Path

import numpy as np

from .data import BLOCK_SIZE, EOS, SEP, STOI, decode, encode


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class MathNet:
    def __init__(self, weights_path: str | Path):
        weights_path = Path(weights_path)
        data = np.load(weights_path)
        self.w = {k: data[k].astype(np.float32) for k in data.files}
        cfg = json.loads(weights_path.with_suffix(".json").read_text())
        self.n_layer, self.n_head = cfg["n_layer"], cfg["n_head"]
        self.mask = np.triu(np.full((BLOCK_SIZE, BLOCK_SIZE), -np.inf, np.float32), 1)

    def forward(self, ids: np.ndarray, cache: list | None = None) -> np.ndarray:
        """ids: (T,) новых токенов -> логиты (T, vocab).

        cache — список [k, v] по слоям с ключами/значениями уже обработанных
        позиций (KV-кэш); дополняется на месте. Без кэша считаем с нуля.
        """
        w, T = self.w, len(ids)
        if cache is None:
            cache = [[None, None] for _ in range(self.n_layer)]
        past = 0 if cache[0][0] is None else cache[0][0].shape[1]
        x = w["tok_emb"][ids] + w["pos_emb"][past:past + T]
        C = x.shape[-1]
        hs = C // self.n_head
        for i in range(self.n_layer):
            p = f"blocks.{i}."
            h = layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = h @ w[p + "attn.weight"].T + w[p + "attn.bias"]
            q, k, v = (t.reshape(T, self.n_head, hs).transpose(1, 0, 2)
                       for t in np.split(qkv, 3, axis=-1))
            if cache[i][0] is not None:
                k = np.concatenate([cache[i][0], k], axis=1)
                v = np.concatenate([cache[i][1], v], axis=1)
            cache[i] = [k, v]
            att = q @ k.transpose(0, 2, 1) / np.sqrt(hs) + self.mask[past:past + T, :past + T]
            y = (softmax(att) @ v).transpose(1, 0, 2).reshape(T, C)
            x = x + y @ w[p + "proj.weight"].T + w[p + "proj.bias"]
            h = layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = gelu(h @ w[p + "fc.weight"].T + w[p + "fc.bias"])
            x = x + h @ w[p + "fc2.weight"].T + w[p + "fc2.bias"]
        x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
        return x @ w["tok_emb"].T

    def solve(self, problem: str, temperature: float = 0.0, rng: np.random.Generator | None = None) -> tuple[str, float]:
        """Генерация решения: жадно (temperature=0) или сэмплированием. Возвращает (решение, уверенность 0..1)."""
        ids = encode(problem + SEP)
        start, logp = len(ids), 0.0
        cache = [[None, None] for _ in range(self.n_layer)]
        logits = self.forward(np.array(ids), cache)
        while len(ids) < BLOCK_SIZE:
            if temperature > 0:
                probs = softmax(logits[-1] / temperature)
                nxt = int((rng or np.random.default_rng()).choice(len(probs), p=probs))
            else:
                probs = softmax(logits[-1])
                nxt = int(probs.argmax())
            logp += float(np.log(probs[nxt]))
            ids.append(nxt)
            if nxt == STOI[EOS]:
                break
            logits = self.forward(np.array([nxt]), cache)
        out = decode(ids[start:]).rstrip(EOS)
        return out, float(np.exp(logp))
