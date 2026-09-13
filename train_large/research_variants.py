"""Isolated research variants; none changes the production student default.

Repeated-depth checkpoints require a matching inference graph and must not be
loaded through the production exporter. JEPA auxiliary modules are train-only.
"""
from dataclasses import asdict
import torch
import torch.nn.functional as F
from model import NeuralLexer, student_config
from quant import QuantEmbedding, _codes

BASE = asdict(student_config())
VARIANTS = {
    'r_control': {},
    'r_local': {'config': {'dilations': (1, 1, 2, 4)}},
    'r_row': {'config': {'embedding_scale': 'row', 'film_rank': 32}},
    'r_repeat': {'repeats': 2},
    'r_jepa': {'jepa': 0.1},
    'r_ce': {'class_exponent': 0.0},
    'r_soft_ce': {'class_exponent': 0.25},
    'r_float': {'full_precision': True},
    'r_rms': {'embedding_rms_clip': True},
    # Trade hidden width for precision, keeping hashes and embedding width.
    # Ordinary 2-bit quantization here, not BitNet's ternary quantizer.
    'r_precision': {'config': {'dim': 64, 'proj_bits': 2, 'embed_bits': 4,
                              'film_rank': 32, 'erase_rank': 16}},
}


class RMSQuantEmbedding(QuantEmbedding):
    """Fixed 2.5-RMS clipping per field, versus the baseline field maximum.

    This is a clipping ablation, not learned-step-size quantization. It stores
    exactly the same number of scales and codes as the production embedding.
    """
    def _scale(self):
        square = self.weight.detach().square().mean(-1)
        sums = square.new_zeros(self.n_scales).index_add(0, self.groups, square)
        counts = square.new_zeros(self.n_scales).index_add(0, self.groups, torch.ones_like(square))
        scales = 2.5 * (sums / counts.clamp_min(1)).sqrt() / (2 ** (self.bits - 1) - 1)
        return scales[self.groups, None].clamp_min(1e-8)

    def export_tensors(self):
        result = super().export_tensors()
        scale = self._scale()
        result['codes'] = _codes(self.weight.detach(), scale, self.bits)
        ids = torch.zeros(self.n_scales, dtype=torch.long, device=self.weight.device)
        ids.scatter_(0, self.groups, torch.arange(len(self.groups), device=self.weight.device))
        result['scale'] = scale.squeeze(-1)[ids].cpu().numpy()
        return result


class ResearchLexer(NeuralLexer):
    def __init__(self, cfg, *, repeats=1, full_precision=False, embedding_rms_clip=False):
        super().__init__(cfg)
        self.repeats = repeats
        self.full_precision = full_precision
        self.embedding_rms_clip = embedding_rms_clip
        if embedding_rms_clip:
            old = self.embedding.table
            table = RMSQuantEmbedding(old.num_embeddings, old.embedding_dim,
                                      bits=old.bits, groups=old.groups)
            table.load_state_dict(old.state_dict())
            self.embedding.table = table

    def set_quant(self, enabled):
        super().set_quant(enabled and not self.full_precision)

    def forward(self, feats, valid=None, return_signature=False):
        if self.repeats == 1:
            return super().forward(feats, valid, return_signature)
        if valid is None:
            valid = torch.ones_like(feats['kind'], dtype=torch.bool)
        x = self.embedding(feats)
        original = x
        sig = self.signature(x, valid)
        films = self.film(sig)
        for _ in range(self.repeats):
            for layer, film in zip(self.layers, films):
                x = layer(x, valid, film)
        ctx = self.global_ctx(x, valid)
        token_repr = self.head_norm(x + self.highway_scale * original)
        logits = self.head_out(F.gelu(self.head_hidden(torch.cat([token_repr, ctx], -1))))
        return (logits, sig, token_repr) if return_signature else logits
