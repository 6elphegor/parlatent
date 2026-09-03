"""Looped decoder-only transformer: RMSNorm, RoPE, MHA, ReLU^2 FFN, untied embeddings.

forward():
    tok_emb = wte[toks]; x = tok_emb
    for l in range(n_loops):
        x = norm_x(x) + norm_emb(tok_emb)
        for block in blocks: x = block(x)
        logits_l = lm_head(norm_f(x))
    loss = mean_l CE(logits_l[:, :-1], targets[:, 1:])   # next-token shift
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 15
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    n_loops: int = 4
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    causal: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def rope_cache(head_dim, max_len, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)                      # (T, hd/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd)
    T = x.size(2)
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.causal = cfg.causal
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model, bias=False)
        self.down = nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_x = RMSNorm(cfg.d_model)
        self.norm_emb = RMSNorm(cfg.d_model)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)   # untied
        cos, sin = rope_cache(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_base, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for blk in self.blocks:   # zero-init residual output projections
            nn.init.zeros_(blk.attn.o.weight)
            nn.init.zeros_(blk.ffn.down.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, toks, targets=None, loss_mask=None, n_loops=None, return_all=False):
        """toks: (B, T).  targets: (B, T-1) = full_target[:, 1:].  loss_mask: (T-1,) or (B, T-1) bool.
        Returns (logits_last, loss, per_loop_losses, all_logits_or_None)."""
        n_loops = n_loops or self.cfg.n_loops
        cos, sin = self.rope_cos, self.rope_sin
        tok_emb = self.wte(toks)
        emb_n = self.norm_emb(tok_emb)
        x = tok_emb
        losses, all_logits = [], []
        logits = None
        for _ in range(n_loops):
            x = self.norm_x(x) + emb_n
            for blk in self.blocks:
                x = blk(x, cos, sin)
            logits = self.lm_head(self.norm_f(x))
            if return_all:
                all_logits.append(logits)
            if targets is not None:
                lg = logits[:, :-1].float()
                ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets.reshape(-1), reduction="none")
                ce = ce.view(targets.shape)
                if loss_mask is not None:
                    m = loss_mask.to(ce.dtype).expand_as(ce)
                    ce = (ce * m).sum() / m.sum()
                else:
                    ce = ce.mean()
                losses.append(ce)
        loss = torch.stack(losses).mean() if losses else None
        return logits, loss, losses, (all_logits if return_all else None)

    def param_groups(self):
        """(muon_params, adamw_params): 2D block matrices -> Muon; rest -> AdamW."""
        muon = [p for n, p in self.named_parameters() if p.ndim == 2 and n.startswith("blocks.")]
        rest = [p for n, p in self.named_parameters() if not (p.ndim == 2 and n.startswith("blocks."))]
        return muon, rest
