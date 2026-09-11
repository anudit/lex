"""
Quantization primitives shared by the model and the exporter.

The single most important property here is that training and export agree
exactly. The previous implementation used one scale in `forward`
(0.5 * (|learned| + mean|W|)) and a different one when packing (mean|W|), so the
exported weights were not the weights that were trained. Every quantizer below
exposes one `effective()` method that both paths call.

Weights are stored as bit-planes: a k-bit tensor becomes k separate 1-bit planes,
each packing 32 values per u32. This is exact for any k (3-bit does not divide 32
and would otherwise waste 6% of the file), and a WGSL kernel reconstructs a value
with k shifts and masks instead of a variable-width field extract.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    """Round-to-nearest with a straight-through gradient, clipped to the grid.

    Gradient is passed through only where the latent weight is inside the
    representable range; outside it the weight is saturated and a gradient
    pushing it further out is noise.
    """

    @staticmethod
    def forward(ctx, x, qmax):
        ctx.save_for_backward(x)
        ctx.qmax = qmax
        return torch.clamp(torch.round(x), -qmax - 1, qmax)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        keep = (x >= -ctx.qmax - 1.5) & (x <= ctx.qmax + 0.5)
        return grad * keep.to(grad.dtype), None


def quantize(weight: torch.Tensor, bits: int, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric per-row k-bit quantization. bits=1 degenerates to sign(W) * scale."""
    if bits == 1:
        # sign(0) must not be 0: a zero weight has to pick a side, and the packed
        # format has no representation for it.
        q = torch.where(weight >= 0, 1.0, -1.0)
        return q * scale
    qmax = float(2 ** (bits - 1) - 1)
    q = RoundSTE.apply(weight / scale, qmax)
    return q * scale


def row_scale(weight: torch.Tensor, bits: int,
              groups: torch.Tensor | None = None,
              n_groups: int | None = None) -> torch.Tensor:
    """Scale derived from the weights, so it needs no separate training.

    For 1-bit the mean absolute value is the least-squares optimal scale for
    sign(W). For k>1 the grid spans +/-(2^(k-1)-1), so the scale is set from the
    group maximum to keep the largest weight representable.

    `groups` maps each row to a scale group. A per-row scale on a 32-wide
    embedding row costs 2 bytes against 12 bytes of payload -- a 17% overhead on
    the largest tensor in the model -- so the embedding table shares one scale per
    feature field instead.
    """
    if groups is None:
        if bits == 1:
            s = weight.abs().mean(dim=-1, keepdim=True)
        else:
            qmax = float(2 ** (bits - 1) - 1)
            s = weight.abs().amax(dim=-1, keepdim=True) / qmax
        return s.clamp_min(1e-8)

    # Passing the module's static group count avoids a device-to-host .item()
    # synchronization on every CUDA forward and keeps torch.compile in one graph.
    if n_groups is None:
        n_groups = int(groups.max().item()) + 1
    flat = weight.abs()
    if bits == 1:
        sums = torch.zeros(n_groups, device=weight.device, dtype=weight.dtype)
        sums = sums.index_add(0, groups, flat.mean(dim=-1))
        counts = torch.zeros(n_groups, device=weight.device, dtype=weight.dtype)
        counts = counts.index_add(0, groups, torch.ones_like(flat[:, 0]))
        per_group = sums / counts.clamp_min(1.0)
    else:
        qmax = float(2 ** (bits - 1) - 1)
        per_group = torch.full((n_groups,), 1e-8, device=weight.device,
                               dtype=weight.dtype)
        per_group = per_group.index_reduce(
            0, groups, flat.amax(dim=-1), 'amax', include_self=True) / qmax
    return per_group[groups].unsqueeze(-1).clamp_min(1e-8)


class QuantLinear(nn.Module):
    """Linear layer with k-bit quantization-aware training."""

    def __init__(self, in_features: int, out_features: int, bits: int = 1,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        std = (2.0 / in_features) ** 0.5
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.quant_enabled = True

    def effective(self) -> torch.Tensor:
        if not self.quant_enabled:
            return self.weight
        return quantize(self.weight, self.bits, row_scale(self.weight, self.bits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.effective(), self.bias)

    def export_tensors(self) -> dict:
        w = self.weight.detach()
        s = row_scale(w, self.bits)
        codes = _codes(w, s, self.bits)
        return {
            'kind': 'linear', 'bits': self.bits,
            'shape': [self.out_features, self.in_features],
            'codes': codes, 'scale': s.squeeze(-1).cpu().numpy(),
            'bias': None if self.bias is None else self.bias.detach().cpu().numpy(),
        }


class QuantEmbedding(nn.Module):
    """Embedding table with k-bit quantization-aware training.

    Embeddings get more bits than the projections on purpose: this table is the
    model's entire lexical memory, and a 1-bit row reduces each hash bucket to a
    random sign vector -- which is where a binarized lexer loses its keyword
    discrimination.
    """

    def __init__(self, num_embeddings: int, dim: int, bits: int = 3,
                 groups: torch.Tensor | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = dim
        self.bits = bits
        self.weight = nn.Parameter(torch.randn(num_embeddings, dim) * 0.06)
        self.quant_enabled = True
        if groups is None:
            self.register_buffer('groups', None, persistent=False)
            self.n_scales = num_embeddings
        else:
            self.register_buffer('groups', groups, persistent=False)
            self.n_scales = int(groups.max().item()) + 1

    def _scale(self) -> torch.Tensor:
        return row_scale(self.weight, self.bits, self.groups, self.n_scales)

    def effective(self) -> torch.Tensor:
        if not self.quant_enabled:
            return self.weight
        return quantize(self.weight, self.bits, self._scale())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(idx, self.effective())

    def export_tensors(self) -> dict:
        w = self.weight.detach()
        s = row_scale(w, self.bits, self.groups, self.n_scales)
        # Only the distinct group scales ship; the group id of a row is implied by
        # the field offset table the shader already needs.
        if self.groups is None:
            scale_out = s.squeeze(-1).cpu().numpy()
        else:
            g = self.groups
            idx = torch.zeros(self.n_scales, dtype=torch.long, device=g.device)
            idx.scatter_(0, g, torch.arange(g.numel(), device=g.device))
            scale_out = s.squeeze(-1)[idx].cpu().numpy()
        return {
            'kind': 'embedding', 'bits': self.bits,
            'shape': [self.num_embeddings, self.embedding_dim],
            'codes': _codes(w, s, self.bits),
            'scale': scale_out, 'groups': None if self.groups is None else
                self.groups.cpu().numpy(), 'bias': None,
        }


def _codes(w: torch.Tensor, s: torch.Tensor, bits: int) -> np.ndarray:
    """Integer codes in [0, 2^bits) -- the on-disk representation."""
    if bits == 1:
        return (w >= 0).to(torch.uint8).cpu().numpy()
    qmax = float(2 ** (bits - 1) - 1)
    q = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
    return (q + (qmax + 1)).to(torch.uint8).cpu().numpy()


def dequantize_codes(codes: np.ndarray, scale: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of `_codes`; used by the export round-trip test."""
    if bits == 1:
        q = np.where(codes.astype(np.int32) == 1, 1.0, -1.0)
    else:
        qmax = float(2 ** (bits - 1) - 1)
        q = codes.astype(np.float32) - (qmax + 1)
    return q.astype(np.float32) * scale.reshape(-1, 1).astype(np.float32)


def pack_bitplanes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer codes into `bits` bit-planes, 32 values per uint32.

    The tensor is flattened row-major and packed densely: plane b holds bit b of
    value i at word i>>5, bit i&31. Every matrix in this model has a row width
    that is a multiple of 32, so a row still starts on a word boundary and a
    shader can index it as row * (cols/32). Padding each row to 32 instead would
    cost 1.7 KB on the 5-tap depthwise kernels alone.
    """
    flat = codes.astype(np.uint32).ravel()
    n = flat.size
    words = (n + 31) // 32
    out = np.zeros((bits, words), dtype=np.uint32)
    for b in range(bits):
        plane = (flat >> b) & 1
        padded = np.pad(plane, (0, words * 32 - n)).reshape(words, 32)
        out[b] = (padded << np.arange(32, dtype=np.uint32)).sum(axis=1, dtype=np.uint32)
    return out


def unpack_bitplanes(packed: np.ndarray, bits: int, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    words = packed.shape[-1]
    flat = np.zeros(n, dtype=np.uint8)
    idx = np.arange(n)
    for b in range(bits):
        plane = packed[b]
        vals = ((plane[idx >> 5] >> (idx & 31)) & 1).astype(np.uint8)
        flat |= vals << b
    return flat.reshape(rows, cols)
