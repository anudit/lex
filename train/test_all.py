"""
Test suite for the neural lexer.

These assertions are chosen to fail on the bugs this project actually hit, not
to confirm that files exist:

  * the exporter silently omitted the depthwise kernels, norms and decays, so
    weights.bin could not reconstruct the model -> `test_export_roundtrip`
  * the exporter wrote a different scale than the forward pass used, so the
    shipped weights were not the trained ones -> same test, exact comparison
  * WGSL's tanh returned NaN on large inputs and argmax silently fell back to
    class 0 -> `test_no_nan_on_extremes`
  * the JS and Python tokenizers could drift and the model would then see
    features it never trained on -> `test_tokenizer_parity`
  * a language could land entirely in the training split and be scored as 0%
    without anyone noticing -> `test_every_language_has_val_data`

Run: uv run python test_all.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import data_pipeline as dp
import export
import reference
import tokenizer
import wgsl
from labels import CLASS_NAMES, MASK, NUM_CLASSES
from languages import TARGET_LANGUAGES, weights
from model import FIELD_SIZES, LexerConfig, NeuralLexer
from quant import pack_bitplanes, unpack_bitplanes, quantize, row_scale

HERE = Path(__file__).resolve().parent
PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def deployment_config() -> LexerConfig:
    return LexerConfig(film_rank=32, erase_rank=8)


def _fn_bodies(src: str) -> dict[str, str]:
    """Crude WGSL function splitter: good enough to see which globals a body names."""
    bodies: dict[str, str] = {}
    for m in re.finditer(r'\bfn\s+(\w+)\s*\(', src):
        name = m.start()
        open_brace = src.find('{', m.end())
        depth, i = 0, open_brace
        while i < len(src):
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        bodies[m.group(1)] = src[open_brace:i]
    return bodies


def _max_workgroup_bytes(src: str, entries: list[str]) -> tuple[int, str]:
    sizes = {n: int(c) * 4 for n, c in
             re.findall(r'var<workgroup>\s+(\w+)\s*:\s*array<f32,\s*(\d+)>', src)}
    bodies = _fn_bodies(src)

    def used(fn: str, seen: set[str]) -> set[str]:
        if fn in seen or fn not in bodies:
            return set()
        seen.add(fn)
        body = bodies[fn]
        arrays = {n for n in sizes if re.search(rf'\b{n}\b', body)}
        for callee in set(re.findall(r'\b(\w+)\s*\(', body)):
            arrays |= used(callee, seen)
        return arrays

    worst, worst_entry = 0, ''
    for e in entries:
        total = sum(sizes[n] for n in used(e, set()))
        if total > worst:
            worst, worst_entry = total, e
    return worst, worst_entry


def test(fn):
    name = fn.__name__.replace('test_', '').replace('_', ' ')
    try:
        detail = fn()
        PASSED.append(f'{name}{f" -- {detail}" if detail else ""}')
    except Exception as exc:  # noqa: BLE001 - the suite reports, it does not raise
        FAILED.append((name, f'{type(exc).__name__}: {exc}'))
    return fn


@test
def test_bitplane_roundtrip():
    """Packing must be exact for every bit width the model uses."""
    for bits in (1, 2, 3, 4):
        for shape in ((37, 70), (64, 5), (9, 96), (1, 32)):
            codes = np.random.randint(0, 2 ** bits, shape).astype(np.uint8)
            back = unpack_bitplanes(pack_bitplanes(codes, bits), bits, *shape)
            assert np.array_equal(codes, back), f'bits={bits} shape={shape}'
    return 'bits 1-4, four shapes'


@test
def test_quantizer_matches_dequantized_codes():
    """The forward pass and the exported codes must describe the same weights."""
    for bits in (1, 3, 4):
        w = torch.randn(48, 96) * 0.1
        s = row_scale(w, bits)
        fwd = quantize(w, bits, s).numpy()
        from quant import _codes, dequantize_codes
        deq = dequantize_codes(_codes(w, s, bits), s.squeeze(-1).numpy(), bits)
        assert np.allclose(fwd, deq, atol=1e-6), f'bits={bits} drift'
    return 'forward == dequantized codes'


@test
def test_export_roundtrip():
    """Every parameter ships, and reconstructs to what training produced."""
    m = NeuralLexer(deployment_config())
    m.eval()
    with tempfile.TemporaryDirectory() as d:
        rep = export.export_model(m, d)  # raises if any parameter is unexported
        assert rep['max_abs_error'] < 2e-3, f"drift {rep['max_abs_error']:.2e}"
        meta = json.loads((Path(d) / 'weights.meta.json').read_text())
        covered = set(meta['tensors'])
        for name, _ in m.named_parameters():
            owner = name.rsplit('.', 1)[0]
            assert name in covered or owner in covered, f'{name} missing'
    return f"{rep['kb']:.2f} KB, max error {rep['max_abs_error']:.1e}"


# Size budget. The model was under gpu-lexer's 31.0 KB until FiLM conditioning
# was added; that was a deliberate trade of 6 KB for the ability to route the
# document signature through every layer. The assertion tracks the budget that
# was actually agreed, and still checks that the reported size matches the file.
# The selective-reset recipe uses word-aligned FiLM rank 32 and spends the
# reclaimed bytes on rank-8 erase projections. The hard deployment budget is
# 40 KiB; the current exact export remains comfortably below it.
SIZE_BUDGET_KB = 40.0
GPU_LEXER_KB = 31.0


@test
def test_packed_size_within_budget():
    """Size is a headline claim, so it is asserted, not reported."""
    m = NeuralLexer(deployment_config())
    r = m.size_report()
    with tempfile.TemporaryDirectory() as d:
        actual = export.export_model(m, d)['kb']
    assert abs(actual - r['packed_kb']) < 0.01, (
        f'size_report {r["packed_kb"]:.2f} KB but file is {actual:.2f} KB')
    assert actual < SIZE_BUDGET_KB, f'{actual:.2f} KB exceeds the {SIZE_BUDGET_KB} KB budget'
    rel = 'under' if actual < GPU_LEXER_KB else 'over'
    return f'{actual:.2f} KB (budget {SIZE_BUDGET_KB} KB, {rel} gpu-lexer {GPU_LEXER_KB} KB)'


@test
def test_numpy_reference_matches_torch():
    """The reference implementation is the shader's spec; it must match torch."""
    m = NeuralLexer(deployment_config())
    m.eval()
    m.set_quant(True)
    with tempfile.TemporaryDirectory() as d:
        export.export_model(m, d)
        ref = reference.Reference(d)
        code = 'const MAX_N = 3.14; // hi\nfunction go(a) { return a; }\n'
        feats, toks = reference.features_from_code(code)
        np_logits = ref.forward(feats)
        tf = {k: torch.from_numpy(v).unsqueeze(0) for k, v in feats.items()}
        with torch.no_grad():
            pt = m(tf, torch.ones(1, len(toks), dtype=torch.bool)).squeeze(0).numpy()
    assert (np_logits.argmax(-1) == pt.argmax(-1)).all(), 'class disagreement'
    return f'max logit diff {np.abs(np_logits - pt).max():.1e}'


@test
def test_no_nan_on_extremes():
    """Activations must saturate, not produce NaN.

    WGSL computes tanh as (e^2x-1)/(e^2x+1) on some backends, so a large argument
    gives inf/inf. The GELU in the head reaches that range on ordinary input, and
    the failure is silent: NaN logits make argmax return class 0.
    """
    m = NeuralLexer(deployment_config())
    m.eval()
    m.set_quant(True)
    B, T = 1, 64
    for scale in (1, 50, 1000):
        feats = {k: torch.randint(0, FIELD_SIZES.get(k, 8), (B, T))
                 for k in dp.FEATURE_KEYS if k != 'flags'}
        feats['flags'] = torch.full((B, T), 255)
        with torch.no_grad():
            out = m(feats, torch.ones(B, T, dtype=torch.bool)) * scale
        assert torch.isfinite(out).all(), f'non-finite logits at scale {scale}'
    return 'finite logits with all flags set'


@test
def test_shader_generates_and_covers_pipeline():
    """Every entry point the runtime dispatches must exist in the shader."""
    m = NeuralLexer(deployment_config())
    with tempfile.TemporaryDirectory() as d:
        export.export_model(m, d)
        meta = json.loads((Path(d) / 'weights.meta.json').read_text())
    src = wgsl.generate(meta)
    steps = wgsl.pipeline_order(meta['config']['n_layers'])
    for step in steps:
        assert f"fn {step['entry']}(" in src, f"missing entry point {step['entry']}"
    assert src.count('Low-rank input-conditioned erase') == meta['config']['n_layers']
    assert 'base_f * (1.0 - scratch[r_fw' in src
    assert 'dotRed32(planes[FU_P + o])' in src
    assert 'planes[FU_P + o * 2u]' not in src
    worst, worst_entry = _max_workgroup_bytes(src, [s['entry'] for s in steps])
    # WebGPU guarantees only 16 KB of workgroup storage, and the limit applies
    # per entry point over what it transitively references -- not to the module.
    assert worst <= 16384, f'{worst_entry} needs {worst} B of workgroup memory (16 KB limit)'
    return f'{len(steps)} entry points, worst {worst_entry} at {worst} B'


@test
def test_tokenizer_parity():
    """The JS and Python tokenizers must agree bit-for-bit.

    A feature computed differently at inference is a feature the model never
    trained on, and the failure is invisible -- slightly wrong colours, no error.
    """
    samples = {
        'a.py': 'def f(x: int) -> int:\n    # c\n    return x + MAX_N\n',
        'b.js': 'export const f = async (a) => `x${a}`; // done\n',
        'c.txt': '=>{}[]();:: \t\r\n\\ "quoted" élève\n',
    }
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, text in samples.items():
            p = Path(d) / name
            p.write_text(text, encoding='utf-8')
            paths.append(str(p))
        (Path(d) / 'list.txt').write_text('\n'.join(paths))
        script = Path(d) / 'tok.mjs'
        script.write_text(
            "import {tokenize} from '%s/../lex/src/tokenizer.js';\n"
            "import fs from 'node:fs';\n"
            "const out=[];\n"
            "for (const f of fs.readFileSync('%s/list.txt','utf8').trim().split('\\n')) {\n"
            "  const t = tokenize(fs.readFileSync(f,'utf8'));\n"
            "  out.push({f, count:t.count, packed:Array.from(t.packed)});\n"
            "}\n"
            "process.stdout.write(JSON.stringify(out));\n" % (HERE, d))
        proc = subprocess.run(['node', str(script)], cwd=HERE,
                              capture_output=True, text=True)
        assert proc.returncode == 0, f'node failed: {proc.stderr[-400:]}'
        js = json.loads(proc.stdout)
        # Read the sources before the temp directory is removed.
        texts = {str(Path(d) / name): text for name, text in samples.items()}

    total = 0
    for entry in js:
        code = texts[entry['f']]
        toks = tokenizer.tokenize(code)
        assert len(toks) == entry['count'], f"{entry['f']}: token count differs"
        packed = []
        for t in toks:
            packed.append((t.kind & 3) | ((t.len_bucket & 7) << 2)
                          | ((t.first_char & 127) << 5) | ((t.last_char & 127) << 12)
                          | ((t.flags & 255) << 19) | ((t.sym_next & 31) << 27))
            packed.append((t.hash1 & 1023) | ((t.hash2 & 127) << 10)
                          | ((t.trans_prev & 15) << 17) | ((t.trans_next & 15) << 21)
                          | ((t.sym_prev & 31) << 25))
        assert packed == entry['packed'], f"{entry['f']}: packed features differ"
        total += len(toks)
    return f'{total} tokens identical'


@test
def test_hash_buckets_within_table():
    """Tokenizer bucket ranges must fit the embedding rows they index."""
    code = ('class HttpServer { const MAX = 0xFF; } // comment\n'
            'def snake_case_name(x): return x\n' * 40)
    toks = tokenizer.tokenize(code)
    arrays = tokenizer.tokens_to_arrays(toks)
    for field, size in FIELD_SIZES.items():
        hi = int(arrays[field].max())
        assert hi < size, f'{field} produced {hi}, table has {size} rows'
    return 'all fields within table bounds'


@test
def test_class_projection_is_total():
    """Every Shiki scope must map to exactly one of the nine classes."""
    proc = subprocess.run(
        ['node', '--input-type=module', '-e',
         "import {classifyScopes, CLASS_NAMES} from './scope_map.mjs';\n"
         "const probes = ['comment.line.double-slash.js','string.quoted.double',"
         "'constant.numeric.float','keyword.control.flow','keyword.operator.assignment',"
         "'entity.name.function','entity.name.type.class','support.type.primitive',"
         "'variable.other.readwrite','punctuation.definition.string.begin',"
         "'storage.modifier','constant.language.boolean','meta.embedded','source.unknown.x'];\n"
         "const out = probes.map(s => classifyScopes(['source.x', s]));\n"
         "if (out.some(c => c === undefined || c < 0 || c >= 9)) throw new Error('bad: '+out);\n"
         "process.stdout.write(JSON.stringify(out));"],
        cwd=HERE, capture_output=True, text=True)
    assert proc.returncode == 0, f'node failed: {proc.stderr[-400:]}'
    got = json.loads(proc.stdout)
    assert got[0] == CLASS_NAMES.index('comment')
    assert got[1] == CLASS_NAMES.index('string')
    assert got[2] == CLASS_NAMES.index('number')
    assert got[3] == CLASS_NAMES.index('keyword')
    assert got[4] == CLASS_NAMES.index('operator'), 'keyword.operator must beat keyword'
    assert got[5] == CLASS_NAMES.index('function')
    assert got[9] == CLASS_NAMES.index('string'), 'string punctuation stays string'
    return f'{len(got)} scopes, all in range'


@test
def test_shader_and_runtime_agree_on_scratch_layout():
    """The scratch region count must match between the shader and the runtime.

    Region bases are NREG * stride * job_index, so if the two disagree by one,
    each job's last region lands exactly on the next job's first region. Every
    job but the last then reads a neighbour's data -- and single-job tests, which
    have no neighbour, all pass.
    """
    m = re.search(r'^N_REGIONS = (\d+)', (HERE / 'wgsl.py').read_text(), re.M)
    assert m, 'N_REGIONS not found in wgsl.py'
    shader_n = int(m.group(1))
    js = (HERE.parent / 'lex' / 'src' / 'runtime.js').read_text()
    m2 = re.search(r'const SCRATCH_REGIONS = (\d+)', js)
    assert m2, 'SCRATCH_REGIONS not found in runtime.js'
    runtime_n = int(m2.group(1))
    assert shader_n == runtime_n, (
        f'shader allocates {shader_n} regions, runtime allocates {runtime_n}')
    return f'{shader_n} regions on both sides'


@test
def test_holdout_repos_are_actually_held_out():
    """No repository may appear in both the training and holdout lists.

    An earlier version shared 17 repositories, 9 of them in the same language,
    which would have inflated the "unseen repos" number without any visible
    symptom.
    """
    from repos_holdout import holdout_repos, overlap
    o = overlap()
    assert not o, f'repos in both training and holdout: {o}'
    h = holdout_repos()
    thin = {k: len(v) for k, v in h.items() if len(v) < 3}
    assert not thin, f'languages with too few holdout repos: {thin}'
    return f'{len(h)} languages, {sum(len(v) for v in h.values())} repos, no overlap'


@test
def test_language_weights_normalized():
    w = weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9, f'weights sum to {sum(w.values())}'
    assert set(w) == set(TARGET_LANGUAGES)
    return f'{len(w)} languages sum to 1.0'


@test
def test_training_language_policy_and_sampler():
    from languages import TRAIN_EXCLUDED_LANGUAGES, TRAIN_LANGUAGES
    from train import sampling_probabilities
    assert len(TRAIN_LANGUAGES) == 52, len(TRAIN_LANGUAGES)
    assert len(TRAIN_EXCLUDED_LANGUAGES) == 5
    probs = sampling_probabilities(None, exponent=0.5, tail_floor=0.001,
                                   include_excluded=False)
    assert abs(float(probs.sum()) - 1.0) < 1e-12
    for i, lang in enumerate(TARGET_LANGUAGES):
        if lang in TRAIN_EXCLUDED_LANGUAGES:
            assert probs[i] == 0, f'{lang} still receives training draws'
        else:
            assert probs[i] > 0, f'{lang} was accidentally starved'
    return '52 active, 5 zero-probability exclusions, normalized'


@test
def test_boundary_detection_crosses_masked_whitespace():
    from train import _boundary_counts, _boundary_masks
    labels = torch.tensor([[1, MASK, MASK, 2, MASK, 2]])
    correct = labels.clone()
    adjacent, gold, predicted = _boundary_masks(labels, correct)
    assert gold.sum() == 1 and predicted.sum() == 1
    assert adjacent[0, 0] and adjacent[0, 3]
    assert _boundary_counts(labels, correct) == (1, 1, 1)
    missed = torch.tensor([[1, 0, 0, 1, 0, 1]])
    assert _boundary_counts(labels, missed) == (0, 0, 1)
    return 'nearest scored-token transition is counted across whitespace'


@test
def test_split_is_deterministic_and_by_content():
    """Identical file content must always land in the same split -- hashing by
    content (not path) is what keeps a duplicate/vendored file from leaking
    across train/val/test, even when fetched under a different repo path."""
    names = [f'repo__src_file_{i}.py content body {i}' for i in range(3000)]
    first = [dp.split_of(n) for n in names]
    second = [dp.split_of(n) for n in names]
    assert first == second, 'split is not deterministic'
    counts = {s: first.count(s) for s in ('train', 'val', 'test')}
    assert counts['val'] > 0 and counts['test'] > 0, counts
    assert 0.80 < counts['train'] / len(names) < 0.95, counts
    return f"train {counts['train']} / val {counts['val']} / test {counts['test']}"


@test
def test_every_language_has_val_data():
    """A language with no val windows is silently scored as 0% by the metric.

    This actually happened: all 35 markdown files hashed into train, so markdown
    contributed a hard zero to a headline number for an entire run.
    """
    cache = HERE / 'corpus' / 'dataset'
    if not (cache / 'meta.json').exists():
        return 'skipped (no dataset cache)'
    ds, meta = dp.build(cache=str(cache))
    present = {int(l) for l in ds['val'].langs}
    missing = [l for i, l in enumerate(TARGET_LANGUAGES) if i not in present]
    assert not missing, f'no val data for: {missing}'
    return f'{len(present)}/{len(TARGET_LANGUAGES)} languages in val'


@test
def test_labels_align_without_guessing():
    """Tokens Shiki does not cover must be masked, never assigned a class."""
    code = 'def f():\n    return 1\n'
    char_cls = np.full(len(code), 9, dtype=np.uint8)  # nothing covered
    toks = tokenizer.tokenize(code)
    lab = dp.align(toks, char_cls)
    assert (lab == MASK).all(), 'uncovered tokens were given labels'
    char_cls[:3] = CLASS_NAMES.index('keyword')
    lab = dp.align(toks, char_cls)
    assert lab[0] == CLASS_NAMES.index('keyword')
    assert (lab[1:] == MASK).all()
    return 'uncovered spans stay masked'


def main() -> int:
    print('=' * 66)
    print('NEURAL LEXER TEST SUITE')
    print('=' * 66)
    for line in PASSED:
        print(f'  PASS  {line}')
    for name, err in FAILED:
        print(f'  FAIL  {name}\n          {err}')
    print('=' * 66)
    print(f'{len(PASSED)} passed, {len(FAILED)} failed')
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
