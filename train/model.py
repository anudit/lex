"""
lex-lite (v2): a token-classification model sized for WebGPU inference.

  * Factorized embedding. One shared 3-bit table holds every feature field at
    disjoint offsets (one scale per field) and is projected up to `dim`. It is
    the model's lexical memory and its precision is load-bearing.
  * Whitespace-free sequence. The tokenizer (tokenize_v2) drops whitespace and
    newline runs; what they carried arrives as per-token fields -- gap to each
    neighbour, line indentation, first symbol on the line, bracket depth -- so
    every conv tap and every recurrent step is spent on a real token.
  * Document signature + FiLM. Masked mean/max of the embeddings, pooled before
    any layer, is projected to a per-layer scale and shift. Language identity is
    highly separable from it, and FiLM routes it into the conv and recurrence
    (`$` in shell vs Markdown) instead of only the classifier.
  * Bidirectional gated linear recurrence, h = a*h + b, run as a log2(T)
    associative scan. Decay and input gain are decoupled so a channel can hold
    long memory without losing its input; forward channels span ~10-500 tokens,
    backward ~1-15. A low-rank input-conditioned erase gate lets closing
    delimiters clear string/comment state.
  * Global context from prefix and suffix means (declaration-gated prefix), so a
    token can tell "declared earlier" from "appears from nowhere"; it is
    length-invariant, behaving the same on a snippet and a long file.
  * Quantization-aware throughout: 3-bit embedding, 1-bit projections, 4-bit
    depthwise kernels, 8-bit scales/biases/norms/decays (scalar_bits=8).

lex-large (../train_large) builds on the same blocks with wider hashes and
additional structural fields. Historical feature-version-1 checkpoints still
load through the compatibility paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import QuantEmbedding, QuantLinear, fake_int8

# Base feature layout (tokenizer.tokenize, feature_version=1): lex-large's input,
# which it extends by reassigning this dict.
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
# lex-lite's layout (tokenizer.tokenize_v2, feature_version=2): whitespace-free
# sequence, 256 primary hash buckets, and the structural fields.
FIELD_SIZES_V2: dict[str, int] = {
    **{k: v for k, v in FIELD_SIZES.items()},
    'hash1': 256,
    'gap_prev': 4,
    'gap_next': 4,
    'indent': 8,
    'line_first': 32,
    'brace_depth': 4,
    'paren_depth': 4,
}
N_FLAG_BITS = 8


def field_sizes(version: int) -> dict[str, int]:
    return FIELD_SIZES if version == 1 else FIELD_SIZES_V2


@dataclass
class LexerConfig:
    dim: int = 64
    embed_dim: int = 32
    embed_bits: int = 3
    proj_bits: int = 1
    head_bits: int = 1
    input_bits: int | None = None
    output_bits: int | None = None
    embedding_scale: str = 'field'
    binary_ste: bool = True
    conv_bits: int = 4
    kernel_size: int = 5
    n_layers: int = 3
    head_hidden: int = 96
    num_classes: int = 9
    # Asymmetric decay timescales: forward channels span 10-500 tokens for long-range
    # syntax retention; backward channels span 1.5-15 tokens for sharp local lookahead.
    decay_f_min: float = 0.90   # ~10 tokens
    decay_f_max: float = 0.998  # ~500 tokens
    decay_b_min: float = 0.20   # ~1.2 tokens
    decay_b_max: float = 0.93   # ~15 tokens
    decay_min: float = 0.5      # fallback for symmetric
    decay_max: float = 0.995    # fallback for symmetric
    # Dilated depthwise convolution rates per layer: a +-14 token receptive
    # field at zero parameter cost.
    dilations: tuple[int, ...] = (1, 2, 4)
    # Rank 32 is word-aligned in the packed runtime.
    film_rank: int = 32
    # Rank of each layer's input-conditioned erase gate; 0 disables it.
    erase_rank: int = 8
    # Accepted for serialized configs; has no effect.
    use_dynamic_reset: bool = True
    use_decl_gate: bool = True       # declaration-biased prefix in GlobalContext
    use_highway: bool = True         # embedding-to-head direct highway
    # Training-only regularization for the unconstrained FP teacher. Applied to
    # each residual branch and the classifier's hidden layer; inactive in eval()
    # and absent from exports, so it never changes what ships.
    dropout: float = 0.0
    # 2 = lex-lite's whitespace-free sequence + structural fields;
    # 1 = the base layout lex-large uses.
    feature_version: int = 2
    # 'full' pools mean, max, prefix and suffix; 'prefix_suffix' keeps only the
    # two position-dependent views (mean and max already feed the signature).
    ctx_views: str = 'prefix_suffix'
    # 16 ships scales/biases/norms/decays as fp16; 8 as 8-bit codes (QAT'd).
    scalar_bits: int = 8

    def fields(self) -> dict[str, int]:
        return field_sizes(self.feature_version)

    def rows(self) -> int:
        return sum(self.fields().values()) + N_FLAG_BITS


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
        self.fields = cfg.fields()
        offset = 0
        offsets = {}
        for name, size in self.fields.items():
            offsets[name] = offset
            offset += size
        self.flag_offset = offset
        offset += N_FLAG_BITS
        self.total_rows = offset
        self.field_offsets = offsets
        # One scale group per feature field: a per-row scale would add 2 bytes to
        # every 12-byte embedding row, a 17% tax on the largest tensor here.
        group_ids = torch.zeros(offset, dtype=torch.long)
        for gi, (name, size) in enumerate(self.fields.items()):
            group_ids[offsets[name]:offsets[name] + size] = gi
        group_ids[self.flag_offset:] = len(self.fields)
        if cfg.embedding_scale not in ('field', 'row'):
            raise ValueError('embedding_scale must be field or row')
        self.table = QuantEmbedding(offset, cfg.embed_dim, bits=cfg.embed_bits,
                                    groups=group_ids if cfg.embedding_scale == 'field' else None)
        self.up = QuantLinear(cfg.embed_dim, cfg.dim,
                              bits=cfg.input_bits or cfg.proj_bits, bias=True)
        self.register_buffer('_flag_ids',
                             torch.arange(N_FLAG_BITS) + self.flag_offset,
                             persistent=False)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        kind = feats['kind']
        acc = None
        for name in self.fields:
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
    """

    def __init__(self, cfg: LexerConfig):
        super().__init__()
        self.views = getattr(cfg, 'ctx_views', 'full')
        n_views = 4 if self.views == 'full' else 2
        self.summary = QuantLinear(cfg.dim * n_views, cfg.dim, bits=cfg.proj_bits)
        self.gate = QuantLinear(cfg.dim, cfg.dim, bits=cfg.proj_bits)
        self.decl_gate = nn.Parameter(torch.zeros(cfg.dim)) if getattr(cfg, 'use_decl_gate', False) else None

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        m = valid.unsqueeze(-1).to(x.dtype)
        xm = x * m

        if self.decl_gate is not None:
            dg = torch.sigmoid(x * self.decl_gate)
            prefix = (xm * dg).cumsum(dim=1) / (m * dg).cumsum(dim=1).clamp_min(1.0)
        else:
            counts = m.cumsum(dim=1).clamp_min(1.0)
            prefix = xm.cumsum(dim=1) / counts

        rcounts = m.flip(1).cumsum(dim=1).flip(1).clamp_min(1.0)
        suffix = xm.flip(1).cumsum(dim=1).flip(1) / rcounts

        if self.views == 'full':
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (xm.sum(dim=1, keepdim=True) / denom).expand_as(x)
            mx = x.masked_fill(~valid.unsqueeze(-1), float('-inf')).amax(dim=1, keepdim=True)
            mx = torch.nan_to_num(mx, neginf=0.0).expand_as(x)
            views = [mean, mx, prefix, suffix]
        else:
            views = [prefix, suffix]
        ctx = torch.tanh(self.summary(torch.cat(views, dim=-1)))
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
    """Per-layer, per-channel scale and shift derived from the document signature."""

    def __init__(self, cfg: LexerConfig):
        super().__init__()
        self.dim = cfg.dim
        self.n_layers = cfg.n_layers
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
    """Depthwise local context with dilation, then a bidirectional gated linear recurrence with reset."""

    def __init__(self, cfg: LexerConfig, dilation: int = 1):
        super().__init__()
        d = cfg.dim
        self.norm = nn.RMSNorm(d)
        self.dw = QuantLinear(cfg.kernel_size, d, bits=cfg.conv_bits, bias=True)
        self.kernel_size = cfg.kernel_size
        self.dilation = dilation
        self.proj_in = QuantLinear(d, d * 2, bits=cfg.proj_bits)
        self.proj_out = QuantLinear(d * 2, d, bits=cfg.proj_bits)
        self.out_gate = nn.Parameter(torch.zeros(d))
        self.dropout = getattr(cfg, 'dropout', 0.0)
        self.erase_rank = getattr(cfg, 'erase_rank', 0)
        if self.erase_rank > 0:
            self.erase_down = QuantLinear(d, self.erase_rank, bits=cfg.proj_bits)
            self.erase_up = QuantLinear(self.erase_rank, d * 2, bits=cfg.proj_bits)
            # Start as a near-identity (erase ~0). Zero
            # latent weights remain negligible when first quantized because
            # their learned row scale is clamped near zero.
            nn.init.zeros_(self.erase_up.weight)
            nn.init.constant_(self.erase_up.bias, -4.0)
        else:
            self.erase_down = None
            self.erase_up = None

        f_min = getattr(cfg, 'decay_f_min', cfg.decay_min)
        f_max = getattr(cfg, 'decay_f_max', cfg.decay_max)
        b_min = getattr(cfg, 'decay_b_min', cfg.decay_min)
        b_max = getattr(cfg, 'decay_b_max', cfg.decay_max)
        init_f = _decay_logits(d, f_min, f_max)
        init_b = _decay_logits(d, b_min, b_max)
        self.decay_f = nn.Parameter(init_f.clone())
        self.decay_b = nn.Parameter(init_b.clone())

    def _depthwise(self, h: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        dil = self.dilation
        w = self.dw.effective()                       # (dim, k)
        pad = (k // 2) * dil
        hp = F.pad(h.transpose(1, 2), (pad, pad))     # (B, dim, T + 2p)
        taps = hp.unfold(-1, (k - 1) * dil + 1, 1)[:, :, :, ::dil]  # (B, dim, T, k)
        out = (taps * w.view(1, -1, 1, k)).sum(-1)
        return (out + self.dw.bias.view(1, -1, 1)).transpose(1, 2)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                film: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        h = self.norm(x)
        if film is not None:
            gamma, beta = film
            h = h * gamma + beta
        h = self._depthwise(h)
        cand, gate = self.proj_in(h).chunk(2, dim=-1)
        b = torch.tanh(cand) * torch.sigmoid(gate)
        b = b * valid.unsqueeze(-1).to(b.dtype)

        if self.erase_down is not None:
            erase = torch.sigmoid(self.erase_up(torch.tanh(self.erase_down(h))))
            erase_f, erase_b = erase.chunk(2, dim=-1)
            base_f = torch.sigmoid(self.decay_f).view(1, 1, -1)
            base_b = torch.sigmoid(self.decay_b).view(1, 1, -1)
            a_f = base_f * (1.0 - erase_f)
            a_b = base_b * (1.0 - erase_b)
        else:
            a_f = torch.sigmoid(self.decay_f).view(1, 1, -1).expand_as(b)
            a_b = torch.sigmoid(self.decay_b).view(1, 1, -1).expand_as(b)

        fwd = assoc_scan(a_f, b)
        bwd = assoc_scan(a_b.flip(1), b.flip(1)).flip(1)

        y = self.proj_out(torch.cat([fwd, bwd], dim=-1))
        if self.dropout > 0:
            y = F.dropout(y, self.dropout, self.training)
        return x + y * torch.sigmoid(self.out_gate)


class NeuralLexer(nn.Module):
    def __init__(self, cfg: LexerConfig | None = None):
        super().__init__()
        self.cfg = cfg or LexerConfig()
        c = self.cfg
        self.embedding = FeatureEmbedding(c)
        self.signature = DocSignature()
        self.film = FiLM(c) if c.film_rank > 0 else None
        dils = getattr(c, 'dilations', (1, 2, 4))
        self.layers = nn.ModuleList([
            BidiGLUBlock(c, dilation=dils[i] if i < len(dils) else 1)
            for i in range(c.n_layers)
        ])
        self.global_ctx = GlobalContext(c)
        self.head_norm = nn.RMSNorm(c.dim)
        self.highway_scale = nn.Parameter(torch.zeros(1)) if getattr(c, 'use_highway', False) else None
        self.head_hidden = QuantLinear(c.dim * 2, c.head_hidden, bits=c.head_bits)
        self.head_out = QuantLinear(c.head_hidden, c.num_classes,
                                    bits=c.output_bits or c.head_bits)
        for module in self.modules():
            if isinstance(module, QuantLinear):
                module.binary_ste = c.binary_ste
        self._scalar_fq = False
        self._in_scalar_fq = False

    def scalar_param_names(self) -> list[str]:
        """Every parameter that is not a quantized weight matrix: these ship as
        fp16 or, with scalar_bits=8, as per-tensor 8-bit codes."""
        quant = {f'{n}.weight' for n, m in self.named_modules()
                 if isinstance(m, (QuantLinear, QuantEmbedding))}
        return [n for n, _ in self.named_parameters() if n not in quant]

    def forward(self, feats: dict[str, torch.Tensor],
                valid: torch.Tensor | None = None,
                return_signature: bool = False):
        if self._scalar_fq and not self._in_scalar_fq:
            # Re-enter forward with every scalar tensor replaced by its 8-bit
            # value; gradients reach the latent parameters straight through.
            names = set(self.scalar_param_names())
            params = {n: fake_int8(p) for n, p in self.named_parameters() if n in names}
            self._in_scalar_fq = True
            try:
                return torch.func.functional_call(
                    self, params, (feats, valid, return_signature), strict=False)
            finally:
                self._in_scalar_fq = False
        if valid is None:
            valid = torch.ones_like(feats['kind'], dtype=torch.bool)
        x = self.embedding(feats)
        x_orig = x
        sig = self.signature(x, valid)
        films = self.film(sig) if self.film is not None else [None] * len(self.layers)
        for layer, film in zip(self.layers, films):
            x = layer(x, valid, film)
        ctx = self.global_ctx(x, valid)
        if self.highway_scale is not None:
            token_repr = self.head_norm(x + self.highway_scale * x_orig)
        else:
            token_repr = self.head_norm(x)
        h = F.gelu(self.head_hidden(torch.cat([token_repr, ctx], dim=-1)))
        if self.cfg.dropout > 0:
            h = F.dropout(h, self.cfg.dropout, self.training)
        logits = self.head_out(h)
        # The auxiliary language head reads `sig` (the pooled document signature);
        # the auxiliary structural-state head in the trainer reads `token_repr`
        # (the per-token hidden state right before the classifier merges in the
        # global context) -- handing both back avoids a second forward pass over
        # the embedding table, which is the largest tensor in the model.
        return (logits, sig, token_repr) if return_signature else logits

    def doc_signature(self, feats: dict[str, torch.Tensor],
                      valid: torch.Tensor) -> torch.Tensor:
        """The pooled signature on its own, for probes and analysis. Training uses
        `forward(..., return_signature=True)` instead, which is a single pass."""
        return self.signature(self.embedding(feats), valid)

    def set_quant(self, enabled: bool) -> None:
        scalar8 = enabled and getattr(self.cfg, 'scalar_bits', 16) == 8
        self._scalar_fq = scalar8
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantEmbedding)):
                m.quant_enabled = enabled
                m.scale_quant = scalar8

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
        scalar8 = getattr(self.cfg, 'scalar_bits', 16) == 8
        # 8-bit scalars: one byte each, plus an fp16 header per tensor (two for
        # the log-domain scale vectors).
        scalar_bytes = 0
        quant_weights = set()
        for name, mod in self.named_modules():
            if isinstance(mod, (QuantLinear, QuantEmbedding)):
                quant_weights.add(f'{name}.weight')
                n = mod.weight.numel()
                n_scales = getattr(mod, 'n_scales', mod.weight.shape[0])
                # Bit-planes are word-aligned per plane, so count the real cost.
                # A "linear" tensor (other than the depthwise kernel, read
                # scalar-by-scalar and so exempt) is addressed by the shader as
                # row * words_per_row, which needs every row to start on a
                # 32-bit boundary -- export.py pads a row that isn't a multiple
                # of 32 wide to the next one, and that padding must be counted
                # here too or this estimate silently under-reports the file
                # export.py actually writes.
                rows, cols = mod.weight.shape
                if isinstance(mod, QuantLinear) and not name.endswith('.dw') and cols % 32 != 0:
                    n = rows * (cols + (32 - cols % 32))
                words = (n + 31) // 32
                quant_bits += words * 32 * mod.bits
                quant_params += n
                fp16_params += n_scales
                scale_cost = n_scales + 4 if scalar8 else n_scales * 2
                scalar_bytes += scale_cost
                breakdown[name] = {
                    'params': n, 'bits': mod.bits,
                    'bytes': words * 4 * mod.bits + scale_cost,
                }
        for name, p in self.named_parameters():
            if name not in quant_weights:
                fp16_params += p.numel()
                scalar_bytes += p.numel() + 2 if scalar8 else p.numel() * 2
        total_bytes = quant_bits / 8 + scalar_bytes
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'quantized_params': quant_params,
            'fp16_params': fp16_params,
            'packed_bytes': int(total_bytes),
            'packed_kb': total_bytes / 1024.0,
            'breakdown': breakdown,
        }
