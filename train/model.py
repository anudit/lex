"""
Neural lexer: a token-classification model sized for WebGPU inference.

Shape of the design, and what each choice is answering:

  * Factorized embedding. The multi-field feature table is the model's lexical
    memory and dominates the parameter count. Embedding into a narrow dim and
    projecting up costs one small matmul and lets the table carry more bits per
    weight inside the same byte budget -- which is what actually decides whether
    keywords stay separable from identifiers.
  * Log-depth associative scan. The recurrence h_t = a*h_{t-1} + b is associative,
    so it runs in log2(T) parallel steps instead of T sequential ones. The old
    Python loop issued 4*T kernel launches per forward.
  * h = a*h + b, not h = g*h + (1-g)*u. Tying the input gain to (1-decay) means a
    channel that learns long memory simultaneously loses its input signal, so the
    model cannot represent "remember that a block comment opened 300 tokens ago".
  * Explicit global context. gpu-lexer spends two of its seven passes on a
    file-level tree reduction for exactly this reason. Dropping it is a capability
    regression, so it comes back as a masked pool plus gated broadcast -- which is
    length-invariant, and therefore behaves the same on a 40-token snippet and a
    4000-token file.
  * FiLM conditioning on a document signature. A linear probe showed the pooled
    context already identifies the language with 90.8% accuracy -- 100% for
    PowerShell -- while the model still got only 75.7% of PowerShell tokens right.
    The information was present and unused: the context reached the classifier
    only as a concatenated channel at the very last layer, so it could not tell
    the depthwise conv or the recurrence that `$` is not an operator here. The
    signature is now pooled *before* the layers (84.2% language-separable on its
    own) and modulates every layer's activations.

Magika (ICSE 2025) is what prompted that: it shows content-type detection from a
1.5 KB byte sample is essentially solved -- 99% F1 on text types, and its largest
gains over prior tools land on exactly our weak languages (Markdown +45, YAML +25,
Perl +20, SQL +18, CSS +15). The lesson taken here is not to bolt on a language
classifier but that language identity is cheap, highly separable, and worth
routing through the whole network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import QuantEmbedding, QuantLinear

# Feature vocabulary sizes, matching tokenizer.py.
FIELD_SIZES: dict[str, int] = {
    'kind': 4,
    'len_bucket': 8,
    'first_char': 128,
    'last_char': 128,
    'hash1': 512,
    'hash2': 128,
    'trans_prev': 16,
    'trans_next': 16,
    'sym_prev': 32,
    'sym_next': 32,
}
N_FLAG_BITS = 8


@dataclass
class LexerConfig:
    dim: int = 64
    embed_dim: int = 32
    embed_bits: int = 3
    proj_bits: int = 1
    head_bits: int = 1
    conv_bits: int = 4
    kernel_size: int = 5
    n_layers: int = 3
    head_hidden: int = 96
    num_classes: int = 9
    # Decay timescales, as retention per step. A single initialization at 0.9
    # gives every channel a ~10-token memory, which cannot represent "a block
    # comment opened 300 tokens ago". Spreading the channels geometrically over
    # this range gives the layer both local and file-scale memory from the start;
    # training moves them, but only from wherever they were initialized.
    decay_min: float = 0.5     # ~2 tokens
    decay_max: float = 0.995   # ~200 tokens
    # Rank of the FiLM projection from the document signature to per-layer
    # scale/shift. Full rank costs ~7.7 KB packed; 64 buys the same conditioning
    # for ~5.5 KB. Zero disables conditioning entirely, for ablation.
    film_rank: int = 64

    def rows(self) -> int:
        return sum(FIELD_SIZES.values()) + N_FLAG_BITS


def _decay_logits(dim: int, lo: float, hi: float) -> torch.Tensor:
    """Logits whose sigmoids span [lo, hi] geometrically in (1 - decay).

    Spacing in log(1 - gamma) rather than in gamma spreads the channels evenly
    across *timescales*: half the range covers memories shorter than ~20 tokens
    and half covers longer, instead of crowding everything near the top.
    """
    span = torch.linspace(math.log(1.0 - lo), math.log(1.0 - hi), dim)
    gamma = (1.0 - torch.exp(span)).clamp(1e-4, 1 - 1e-4)
    return torch.log(gamma / (1.0 - gamma))


def _shift_right(x: torch.Tensor, n: int, fill: float) -> torch.Tensor:
    """Shift along time, filling the exposed head with the operator identity."""
    if n <= 0:
        return x
    pad = x.new_full((x.shape[0], n, x.shape[2]), fill)
    return torch.cat([pad, x[:, :-n]], dim=1)


def assoc_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of h_t = a_t * h_{t-1} + b_t, with h_{-1} = 0.

    Hillis-Steele over the pair operator (a1,b1) then (a2,b2) -> (a2*a1, a2*b1+b2).
    log2(T) steps, all of them one fused elementwise kernel.
    """
    T = a.shape[1]
    step = 1
    while step < T:
        a_prev = _shift_right(a, step, 1.0)
        b_prev = _shift_right(b, step, 0.0)
        b = b + a * b_prev
        a = a * a_prev
        step *= 2
    return b


class FeatureEmbedding(nn.Module):
    """Multi-field sparse additive embedding, factorized through a narrow dim.

    All fields share one table so the sparse gather is a single indexed read per
    field on the GPU; per-field offsets keep the id spaces disjoint.
    """

    def __init__(self, cfg: LexerConfig):
        super().__init__()
        self.cfg = cfg
        offset = 0
        offsets = {}
        for name, size in FIELD_SIZES.items():
            offsets[name] = offset
            offset += size
        self.flag_offset = offset
        offset += N_FLAG_BITS
        self.total_rows = offset
        self.field_offsets = offsets
        # One scale group per feature field: a per-row scale would add 2 bytes to
        # every 12-byte embedding row, a 17% tax on the largest tensor here.
        group_ids = torch.zeros(offset, dtype=torch.long)
        for gi, (name, size) in enumerate(FIELD_SIZES.items()):
            group_ids[offsets[name]:offsets[name] + size] = gi
        group_ids[self.flag_offset:] = len(FIELD_SIZES)
        self.table = QuantEmbedding(offset, cfg.embed_dim, bits=cfg.embed_bits,
                                    groups=group_ids)
        self.up = QuantLinear(cfg.embed_dim, cfg.dim, bits=cfg.proj_bits, bias=True)
        self.register_buffer('_flag_ids',
                             torch.arange(N_FLAG_BITS) + self.flag_offset,
                             persistent=False)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        kind = feats['kind']
        acc = None
        for name in FIELD_SIZES:
            ids = feats[name] + self.field_offsets[name]
            e = self.table(ids)
            if name in ('hash1', 'hash2'):
                # Word hashes are only defined for word tokens; on a symbol they
                # would alias bucket 0 into every punctuation embedding.
                e = e * (kind == 0).unsqueeze(-1).to(e.dtype)
            acc = e if acc is None else acc + e
        # Flag bits: a multi-hot gather over the 8 flag rows.
        flags = feats['flags']
        bits = ((flags.unsqueeze(-1) >> torch.arange(N_FLAG_BITS, device=flags.device))
                & 1).to(acc.dtype)
        flag_vecs = self.table(self._flag_ids)
        acc = acc + torch.einsum('btf,fd->btd', bits, flag_vecs)
        return self.up(acc)


class GlobalContext(nn.Module):
    """File-scale summary, broadcast back to every position through a gate.

    Four pooled views, because they answer different questions:
      mean        -- what kind of file is this
      max         -- did a construct like an unterminated string appear anywhere
      prefix mean -- what has been seen *before* this token
      suffix mean -- what comes after it

    The first two are constant across the sequence. A single global average
    cannot say whether a given identifier was declared earlier, which is exactly
    the call `type` gets wrong -- it leaks 8.7% into `plain`. The prefix and
    suffix means vary with position and are still length-invariant, so a
    40-token snippet and a 4000-token file behave the same way.
    """

    def __init__(self, cfg: LexerConfig):
        super().__init__()
        self.summary = QuantLinear(cfg.dim * 4, cfg.dim, bits=cfg.proj_bits)
        self.gate = QuantLinear(cfg.dim, cfg.dim, bits=cfg.proj_bits)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        m = valid.unsqueeze(-1).to(x.dtype)
        xm = x * m
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (xm.sum(dim=1, keepdim=True) / denom).expand_as(x)
        mx = x.masked_fill(~valid.unsqueeze(-1), float('-inf')).amax(dim=1, keepdim=True)
        mx = torch.nan_to_num(mx, neginf=0.0).expand_as(x)

        # Running means, normalized by the count of valid positions so far, so
        # neither padding nor sequence length changes their scale.
        counts = m.cumsum(dim=1).clamp_min(1.0)
        prefix = xm.cumsum(dim=1) / counts
        rcounts = m.flip(1).cumsum(dim=1).flip(1).clamp_min(1.0)
        suffix = xm.flip(1).cumsum(dim=1).flip(1) / rcounts

        ctx = torch.tanh(self.summary(torch.cat([mean, mx, prefix, suffix], dim=-1)))
        return ctx * torch.sigmoid(self.gate(x))


class DocSignature(nn.Module):
    """Masked mean and max of the embedding output, pooled over valid positions.

    Computed before any recurrent layer, so it can condition all of them. A linear
    probe recovers the language from this vector at 84.2% accuracy, which is what
    makes it a useful thing to condition on.
    """

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        m = valid.unsqueeze(-1).to(x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=1, keepdim=True) / denom
        mx = x.masked_fill(~valid.unsqueeze(-1), float('-inf')).amax(dim=1, keepdim=True)
        mx = torch.nan_to_num(mx, neginf=0.0)
        return torch.cat([mean, mx], dim=-1).squeeze(1)      # (B, 2 * dim)


class FiLM(nn.Module):
    """Per-layer, per-channel scale and shift derived from the document signature.

    Low-rank because the full projection does not fit the size budget. Each layer
    starts as the identity: `strength` is initialized to zero, so conditioning is
    switched off until training finds a use for it, which keeps the early epochs
    behaving exactly like the unconditioned model.
    """

    def __init__(self, cfg: LexerConfig):
        super().__init__()
        self.dim = cfg.dim
        self.n_layers = cfg.n_layers
        # The signature arrives with mean |x| around 2.4, which drives the
        # projection below into tanh saturation -- measured at 59% of units pinned
        # past 0.99, so most of the low-rank code was the same for every document
        # and the conditioning could not tell them apart. Normalizing first keeps
        # the code in the responsive part of the curve.
        self.norm = nn.RMSNorm(cfg.dim * 2)
        self.down = QuantLinear(cfg.dim * 2, cfg.film_rank, bits=cfg.proj_bits)
        self.up = QuantLinear(cfg.film_rank, cfg.n_layers * 2 * cfg.dim,
                              bits=cfg.proj_bits)
        self.strength = nn.Parameter(torch.zeros(cfg.n_layers, cfg.dim))

    def forward(self, sig: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        h = self.up(torch.tanh(self.down(self.norm(sig))))   # (B, L * 2 * dim)
        h = h.view(-1, self.n_layers, 2, self.dim)
        out = []
        for i in range(self.n_layers):
            s = self.strength[i].view(1, 1, -1)
            gamma = 1.0 + s * torch.tanh(h[:, i, 0]).unsqueeze(1)
            beta = s * torch.tanh(h[:, i, 1]).unsqueeze(1)
            out.append((gamma, beta))
        return out


class BidiGLUBlock(nn.Module):
    """Depthwise local context, then a bidirectional gated linear recurrence."""

    def __init__(self, cfg: LexerConfig):
        super().__init__()
        d = cfg.dim
        self.norm = nn.RMSNorm(d)
        self.dw = QuantLinear(cfg.kernel_size, d, bits=cfg.conv_bits, bias=True)
        self.kernel_size = cfg.kernel_size
        self.proj_in = QuantLinear(d, d * 2, bits=cfg.proj_bits)
        self.proj_out = QuantLinear(d * 2, d, bits=cfg.proj_bits)
        # Per-channel output gate. A full d x d gate matrix costs ~0.6 KB packed
        # for a job a scalar per channel does just as well.
        self.out_gate = nn.Parameter(torch.zeros(d))
        # Decays are stored as logits; init near 0.9 forward / 0.9 backward gives
        # a receptive field of ~10 tokens before training moves them.
        # Decay logits, initialized across a geometric range of timescales
        # rather than all at one value -- see LexerConfig.decay_min/decay_max.
        init = _decay_logits(d, cfg.decay_min, cfg.decay_max)
        self.decay_f = nn.Parameter(init.clone())
        self.decay_b = nn.Parameter(init.clone())

    def _depthwise(self, h: torch.Tensor) -> torch.Tensor:
        # Depthwise conv expressed as a quantized (dim, kernel) weight so it goes
        # through the same quantizer and exporter as every other tensor.
        k = self.kernel_size
        w = self.dw.effective()                       # (dim, k)
        pad = k // 2
        hp = F.pad(h.transpose(1, 2), (pad, pad))     # (B, dim, T + 2p)
        taps = hp.unfold(-1, k, 1)                    # (B, dim, T, k)
        out = (taps * w.view(1, -1, 1, k)).sum(-1)
        return (out + self.dw.bias.view(1, -1, 1)).transpose(1, 2)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                film: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        h = self.norm(x)
        if film is not None:
            # Applied after normalization and before everything else in the
            # block, so the document signature reaches the depthwise conv and the
            # recurrence rather than only the classifier.
            gamma, beta = film
            h = h * gamma + beta
        h = self._depthwise(h)
        cand, gate = self.proj_in(h).chunk(2, dim=-1)
        b = torch.tanh(cand) * torch.sigmoid(gate)
        # Padding must not leak into the scan state.
        b = b * valid.unsqueeze(-1).to(b.dtype)

        a_f = torch.sigmoid(self.decay_f).view(1, 1, -1).expand_as(b)
        a_b = torch.sigmoid(self.decay_b).view(1, 1, -1).expand_as(b)
        fwd = assoc_scan(a_f, b)
        bwd = assoc_scan(a_b.flip(1), b.flip(1)).flip(1)

        y = self.proj_out(torch.cat([fwd, bwd], dim=-1))
        return x + y * torch.sigmoid(self.out_gate)


class NeuralLexer(nn.Module):
    def __init__(self, cfg: LexerConfig | None = None):
        super().__init__()
        self.cfg = cfg or LexerConfig()
        c = self.cfg
        self.embedding = FeatureEmbedding(c)
        self.signature = DocSignature()
        self.film = FiLM(c) if c.film_rank > 0 else None
        self.layers = nn.ModuleList([BidiGLUBlock(c) for _ in range(c.n_layers)])
        self.global_ctx = GlobalContext(c)
        self.head_norm = nn.RMSNorm(c.dim)
        self.head_hidden = QuantLinear(c.dim * 2, c.head_hidden, bits=c.head_bits)
        self.head_out = QuantLinear(c.head_hidden, c.num_classes, bits=c.head_bits)

    def forward(self, feats: dict[str, torch.Tensor],
                valid: torch.Tensor | None = None,
                return_signature: bool = False):
        if valid is None:
            valid = torch.ones_like(feats['kind'], dtype=torch.bool)
        x = self.embedding(feats)
        sig = self.signature(x, valid)
        films = self.film(sig) if self.film is not None else [None] * len(self.layers)
        for layer, film in zip(self.layers, films):
            x = layer(x, valid, film)
        ctx = self.global_ctx(x, valid)
        h = self.head_norm(x)
        h = F.gelu(self.head_hidden(torch.cat([h, ctx], dim=-1)))
        logits = self.head_out(h)
        # The auxiliary language head in the trainer needs the signature; handing
        # it back avoids a second forward pass over the embedding table, which is
        # the largest tensor in the model.
        return (logits, sig) if return_signature else logits

    def doc_signature(self, feats: dict[str, torch.Tensor],
                      valid: torch.Tensor) -> torch.Tensor:
        """The pooled signature on its own, for probes and analysis. Training uses
        `forward(..., return_signature=True)` instead, which is a single pass."""
        return self.signature(self.embedding(feats), valid)

    def set_quant(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantEmbedding)):
                m.quant_enabled = enabled

    def size_report(self) -> dict:
        """Exact packed footprint, counting every parameter that must ship.

        Quantized weights cost their own bit width; everything else -- scales,
        biases, norm gains, decay logits, and the depthwise kernel -- ships as
        fp16 and is counted here rather than quietly omitted.
        """
        quant_bits = 0
        quant_params = 0
        fp16_params = 0
        breakdown = {}
        quant_weights = set()
        for name, mod in self.named_modules():
            if isinstance(mod, (QuantLinear, QuantEmbedding)):
                quant_weights.add(f'{name}.weight')
                n = mod.weight.numel()
                n_scales = getattr(mod, 'n_scales', mod.weight.shape[0])
                # Bit-planes are word-aligned per plane, so count the real cost.
                words = (n + 31) // 32
                quant_bits += words * 32 * mod.bits
                quant_params += n
                fp16_params += n_scales
                breakdown[name] = {
                    'params': n, 'bits': mod.bits,
                    'bytes': words * 4 * mod.bits + n_scales * 2,
                }
        for name, p in self.named_parameters():
            if name not in quant_weights:
                fp16_params += p.numel()
        total_bytes = quant_bits / 8 + fp16_params * 2
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'quantized_params': quant_params,
            'fp16_params': fp16_params,
            'packed_bytes': int(total_bytes),
            'packed_kb': total_bytes / 1024.0,
            'breakdown': breakdown,
        }
