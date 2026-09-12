"""
BESPOKE fork of train/wgsl.py for the lex-large checkpoint only
(dim=96, embed_dim=64, head_hidden=192, film_rank=64, erase_rank=16,
n_layers=4). Do not use this to lower any other checkpoint, and do not fold
it back into wgsl.py as a generalized path -- see train_large/note.txt's
"IMPORTANT DEPLOYMENT LIMIT" section for why the two are kept separate.

wgsl.py already reads DIM/EDIM/HEAD_HIDDEN/FILM_RANK from the checkpoint's
own config/tensor shapes for every *constant*, but several kernels hard-code
which fixed-width dot-product helper to call and how big the workgroup-shared
scratch arrays are, sized for the 64-wide model's actual tensor widths (a
64-wide DIM dot, a 128-wide 2*DIM dot, a 96-wide HEAD_HIDDEN dot, and an
8-word/256-wide global_ctx.summary unroll). For lex-large every one of those
widths is different (DIM dot is 96-wide, 2*DIM and HEAD_HIDDEN are both
192-wide, global_ctx.summary is 384-wide/12 words), so this fork:

  * adds dotA192 / dotB192 (6-word, 192-wide 1-bit dot products)
  * repoints every dot-product call site at the width lex-large's own
    tensors actually have (confirmed against weights.meta.json, not assumed
    from DIM/HEAD_HIDDEN by name)
  * extends the global_ctx.summary unroll from 8 words to 12
  * resizes tA/tB/tP/red for DIM=96, HEAD_HIDDEN=192
  * changes every @workgroup_size(64) to @workgroup_size(96) (=DIM), since
    each entry point's local_invocation_id.x is used as a per-DIM-lane index

Original module docstring follows.
---
Generates the WGSL compute shaders from the exported weight metadata.

The kernel is split into several entry points rather than one fused dispatch.
That reverses an earlier decision, and the reason is measured: profiling
gpu-lexer showed GPU work was ~3% of its latency, so a single workgroup looked
free. That profile was of *their* model. Ours does far more arithmetic per token,
and one workgroup is 64 lanes of a GPU that has thousands -- it ran 2x slower
than gpu-lexer on a single call and 7x slower on a batch.

The split follows where the work actually is. Per token per layer:

    matmuls (proj_in, proj_out, depthwise)   16,704 MACs   99.2%
    bidirectional recurrence                    128 MACs    0.8%

So only the matmuls need parallelizing. They are token-parallel and get one
workgroup per 16-token tile; the recurrence stays on a single workgroup walking
the sequence, which needs no chunked-scan machinery and costs almost nothing.

Every entry point shares one bind group and one command buffer, so the whole
model is still a single submit and a single mapAsync readback -- which is where
96% of the latency of a call actually goes. Batched calls use the second dispatch
dimension: workgroup_id.y selects the job.

Tensor offsets are baked in as literals at generation time, so the kernels do no
indirection to find a weight.
"""

from __future__ import annotations

import json
from pathlib import Path

TILE = 16        # tokens per workgroup for the token-parallel passes
TILE_POOL = 4    # smaller tile for the pooled-context pass; its rows are 4*DIM
MAX_JOBS = 64    # uniform job array needs a compile-time size
N_REGIONS = 8    # per job: nrm, b, fwd, bwd, ctx, pooled stats, film, x_orig.
                 # Must equal SCRATCH_REGIONS in lex/src/runtime.js: the
                 # region base is NREG * stride * job, so a mismatch makes
                 # one job's film block land on the next job's region 0.


def _consts(meta: dict) -> str:
    t = meta['tensors']
    cfg = meta['config']
    lines = [
        f"const DIM: u32 = {cfg['dim']}u;",
        f"const EDIM: u32 = {cfg['embed_dim']}u;",
        f"const HEAD_HIDDEN: u32 = {cfg['head_hidden']}u;",
        f"const NCLASS: u32 = {cfg['num_classes']}u;",
        f"const KSIZE: u32 = {cfg['kernel_size']}u;",
        f"const TILE: u32 = {TILE}u;",
        f"const TILE_POOL: u32 = {TILE_POOL}u;",
        f"const MAX_JOBS: u32 = {MAX_JOBS}u;",
        f"const NREG: u32 = {N_REGIONS}u;",
    ]

    def q(name: str, prefix: str) -> None:
        e = t[name]
        lines.append(f"const {prefix}_P: u32 = {e['plane_offset']}u;")
        lines.append(f"const {prefix}_W: u32 = {e['words_per_plane']}u;")
        lines.append(f"const {prefix}_B: u32 = {e['bits']}u;")
        lines.append(f"const {prefix}_S: u32 = {e['scale_offset']}u;")
        if 'bias_offset' in e:
            lines.append(f"const {prefix}_BI: u32 = {e['bias_offset']}u;")

    q('embedding.table', 'EMB')
    q('embedding.up', 'UP')
    q('film.down', 'FD')
    q('film.up', 'FU')
    q('global_ctx.summary', 'GS')
    q('global_ctx.gate', 'GG')
    q('head_hidden', 'HH')
    q('head_out', 'HO')
    lines.append(f"const HNORM_F: u32 = {t['head_norm.weight']['f16_offset']}u;")
    lines.append(f"const FS_F: u32 = {t['film.strength']['f16_offset']}u;")
    lines.append(f"const FN_F: u32 = {t['film.norm.weight']['f16_offset']}u;")
    lines.append(f"const NLAYER: u32 = {cfg['n_layers']}u;")
    # The down-projection's output width. The workgroup is fixed at 64 lanes,
    # but film_rank is not always 64 (it dropped to 48 without wgsl.py being
    # updated for it) -- threads past FILM_RANK must not write into `red`,
    # since film.down has no weight rows for them and dotRed64 unconditionally
    # sums all 64 slots for the up-projection that follows.
    lines.append(f"const FILM_RANK: u32 = {t['film.down']['shape'][0]}u;")
    if 'global_ctx.decl_gate' in t:
        lines.append(f"const DG_F: u32 = {t['global_ctx.decl_gate']['f16_offset']}u;")
    if 'highway_scale' in t:
        lines.append(f"const HW_F: u32 = {t['highway_scale']['f16_offset']}u;")

    fo = meta['field_offsets']
    order = list(meta['field_sizes'].keys())
    lines.append('const FIELD_ROW: array<u32, %d> = array<u32, %d>(%s);' % (
        len(order), len(order), ', '.join(f'{fo[k]}u' for k in order)))
    lines.append(f"const FLAG_ROW: u32 = {meta['flag_offset']}u;")
    lines.append(f"const NFIELD: u32 = {len(order)}u;")
    lines.append(f"const NFLAG: u32 = {meta['n_flag_bits']}u;")
    return '\n'.join(lines)


PRELUDE = r'''
// One job per workgroup_id.y. Batched calls dispatch every job together so they
// run concurrently and share one readback.
struct Job {
    token_count: u32,
    token_offset: u32,   // slice of the shared token/output buffers
    stage: u32,          // reserved for the stage-diff test harness
    stride: u32,         // per-job hidden-state stride, = maxTokens * DIM
};

@group(0) @binding(0) var<storage, read>       tokens : array<u32>;
@group(0) @binding(1) var<storage, read>       planes : array<u32>;
@group(0) @binding(2) var<storage, read>       fp     : array<f32>;
@group(0) @binding(3) var<storage, read_write> hid    : array<f32>;
@group(0) @binding(4) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_cls: array<u32>;
// A uniform array, not storage: Tint treats uniform loads indexed by a builtin
// as uniform, which is what allows `workgroupBarrier` inside job-dependent
// loops. A storage load here makes every barrier in the kernel illegal.
@group(0) @binding(6) var<uniform>             jobs   : array<Job, MAX_JOBS>;

var<workgroup> tA  : array<f32, 3072>;   // TILE x max(DIM, HEAD_HIDDEN) = 16 x 192
var<workgroup> tB  : array<f32, 3072>;   // TILE x 2*DIM = 16 x 192
var<workgroup> tP  : array<f32, 1536>;   // TILE_POOL x 4*DIM = 4 x 384
// Two slots per lane also accommodate the rank-128 FiLM candidate.
var<workgroup> red : array<f32, 192>;
var<workgroup> lg  : array<f32, 144>;    // TILE x NCLASS

// WGSL's tanh is computed as (e^2x - 1)/(e^2x + 1) on some backends, so a
// moderately large argument overflows to inf/inf = NaN instead of saturating.
// The GELU in the head reaches that range routinely -- its cubic term hits ~90 at
// an input of ~14 -- which silently turned whole tokens into NaN logits and made
// argmax fall back to class 0. Clamping is exact: tanh(15) and sigmoid(30) are
// already 1.0 in f32.
fn tanh_s(x: f32) -> f32 { return tanh(clamp(x, -15.0, 15.0)); }
fn sigmoid_s(x: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
// Matches torch.nn.functional.softplus's numerically-stable form: linear past
// the point where exp(x) would overflow or the log1p term becomes a no-op.
fn softplus_s(x: f32) -> f32 { return select(log(1.0 + exp(x)), x, x > 20.0); }

// k-bit weight from bit-planes. Used only where the row index depends on data:
// the embedding table and the depthwise kernel.
fn wgt(base: u32, wpp: u32, bits: u32, idx: u32, scale: f32) -> f32 {
    var c: u32 = 0u;
    let w = idx >> 5u;
    let b = idx & 31u;
    for (var p: u32 = 0u; p < bits; p = p + 1u) {
        c = c | (((planes[base + p * wpp + w] >> b) & 1u) << p);
    }
    if (bits == 1u) { return select(-scale, scale, c == 1u); }
    return (f32(c) - f32(1u << (bits - 1u))) * scale;
}

// Every projection is 1-bit and its rows are a multiple of 32 wide, so a whole
// row is 1-8 u32 held in registers for the life of the dispatch. Each MAC is
// then a register shift and a sign select rather than a dependent global load.
fn m1(w: u32, k: u32, x: f32) -> f32 { return select(-x, x, ((w >> k) & 1u) == 1u); }

fn dotA32(w: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w, k, tA[base + k]); }
    return a;
}
fn dotA64(w0: u32, w1: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w0, k, tA[base + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w1, k, tA[base + 32u + k]); }
    return a;
}
fn dotA96(w0: u32, w1: u32, w2: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w0, k, tA[base + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w1, k, tA[base + 32u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w2, k, tA[base + 64u + k]); }
    return a;
}
fn dotA128(w0: u32, w1: u32, w2: u32, w3: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w0, k, tA[base + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w1, k, tA[base + 32u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w2, k, tA[base + 64u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w3, k, tA[base + 96u + k]); }
    return a;
}
// 6-word/192-wide variant added for lex-large: film.down's input (2*DIM=192)
// and head_out's input (HEAD_HIDDEN=192) are both this width.
fn dotA192(w0: u32, w1: u32, w2: u32, w3: u32, w4: u32, w5: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w0, k, tA[base + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w1, k, tA[base + 32u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w2, k, tA[base + 64u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w3, k, tA[base + 96u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w4, k, tA[base + 128u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w5, k, tA[base + 160u + k]); }
    return a;
}
fn dotRed64(w0: u32, w1: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w0, k, red[k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w1, k, red[32u + k]); }
    return a;
}
fn dotRed32(w: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w, k, red[k]); }
    return a;
}
fn dotB128(w0: u32, w1: u32, w2: u32, w3: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w0, k, tB[base + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w1, k, tB[base + 32u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w2, k, tB[base + 64u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w3, k, tB[base + 96u + k]); }
    return a;
}
fn dotB32(w: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w, k, tB[base + k]); }
    return a;
}
// 6-word/192-wide variant added for lex-large: proj_out's input (2*DIM=192)
// and head_hidden's input (2*DIM=192) are both this width.
fn dotB192(w0: u32, w1: u32, w2: u32, w3: u32, w4: u32, w5: u32, base: u32) -> f32 {
    var a: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w0, k, tB[base + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w1, k, tB[base + 32u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w2, k, tB[base + 64u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w3, k, tB[base + 96u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w4, k, tB[base + 128u + k]); }
    for (var k: u32 = 0u; k < 32u; k = k + 1u) { a = a + m1(w5, k, tB[base + 160u + k]); }
    return a;
}

// Three words per token, not lex's two: lex-large's tokenizer (see
// lex-large/src/tokenizer.js) adds six structural fields with nowhere left to
// pack in lex's tw0/tw1 layout (tw0 is bit-for-bit full; tw1 has 2 spare bits
// out of 32), plus wider hash1/hash2 that no longer fit tw1's old field
// widths either -- so tw1 itself is repacked, not just extended.
fn tw0(o: u32, i: u32) -> u32 { return tokens[(o + i) * 3u]; }
fn tw1(o: u32, i: u32) -> u32 { return tokens[(o + i) * 3u + 1u]; }
fn tw2(o: u32, i: u32) -> u32 { return tokens[(o + i) * 3u + 2u]; }
fn tok_kind(o: u32, i: u32)  -> u32 { return  tw0(o, i)        & 3u; }
fn tok_flags(o: u32, i: u32) -> u32 { return (tw0(o, i) >> 19u) & 255u; }

fn field_id(o: u32, i: u32, f: u32) -> u32 {
    switch f {
        case 0u:  { return  tw0(o, i)        & 3u; }          // kind
        case 1u:  { return (tw0(o, i) >>  2u) & 7u; }          // len_bucket
        case 2u:  { return (tw0(o, i) >>  5u) & 127u; }        // first_char
        case 3u:  { return (tw0(o, i) >> 12u) & 127u; }        // last_char
        case 4u:  { return  tw1(o, i)        & 1023u; }        // hash1 (10 bits)
        case 5u:  { return (tw1(o, i) >> 10u) & 255u; }        // hash2 (8 bits, wider than lex's 7)
        case 6u:  { return (tw1(o, i) >> 18u) & 15u; }         // trans_prev
        case 7u:  { return (tw1(o, i) >> 22u) & 15u; }         // trans_next
        case 8u:  { return (tw1(o, i) >> 26u) & 31u; }         // sym_prev
        case 9u:  { return (tw0(o, i) >> 27u) & 31u; }         // sym_next
        case 10u: { return  tw2(o, i)        & 7u; }           // paren_depth
        case 11u: { return (tw2(o, i) >>  3u) & 7u; }          // brace_depth
        case 12u: { return (tw2(o, i) >>  6u) & 7u; }          // bracket_depth
        case 13u: { return (tw2(o, i) >>  9u) & 7u; }          // line_pos
        case 14u: { return (tw2(o, i) >> 12u) & 7u; }          // indent_bucket
        default:  { return (tw2(o, i) >> 15u) & 3u; }          // quote_state (f == 15)
    }
}

// Per-token inverse RMS for a tile, computed by the first TILE threads.
fn tile_rms(d: u32, n: u32) {
    if (d < n) {
        var s: f32 = 0.0;
        for (var k: u32 = 0u; k < DIM; k = k + 1u) {
            let v = tA[d * DIM + k];
            s = s + v * v;
        }
        red[d] = inverseSqrt(s / f32(DIM) + 1e-6);
    }
}
'''

EMBED = r'''
@compute @workgroup_size(96)
fn embed(@builtin(local_invocation_id) lid: vec3<u32>,
         @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let off = job.token_offset;
    let hbase = job.stride * wid.y;
    let r_xorig = NREG * job.stride * wid.y + 7u * job.stride;
    let base_t = wid.x * TILE;

    let up_w0 = planes[UP_P + d * 2u];
    let up_w1 = planes[UP_P + d * 2u + 1u];
    let up_s = fp[UP_S + d];
    let up_b = fp[UP_BI + d];

    for (var j: u32 = 0u; j < TILE; j = j + 1u) {
        let t = base_t + j;
        if (d < EDIM) {
            var acc: f32 = 0.0;
            if (t < T) {
                let kind = tok_kind(off, t);
                for (var f: u32 = 0u; f < NFIELD; f = f + 1u) {
                    // Word hashes are undefined on non-word tokens; feeding
                    // bucket 0 into every symbol would alias them together.
                    if ((f == 4u || f == 5u) && kind != 0u) { continue; }
                    let row = FIELD_ROW[f] + field_id(off, t, f);
                    acc = acc + wgt(EMB_P, EMB_W, EMB_B, row * EDIM + d, fp[EMB_S + f]);
                }
                let flags = tok_flags(off, t);
                for (var b: u32 = 0u; b < NFLAG; b = b + 1u) {
                    if (((flags >> b) & 1u) == 1u) {
                        acc = acc + wgt(EMB_P, EMB_W, EMB_B,
                                        (FLAG_ROW + b) * EDIM + d, fp[EMB_S + NFIELD]);
                    }
                }
            }
            tA[j * EDIM + d] = acc;
        }
    }
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {
        let t = base_t + j;
        if (t < T) {
            let v = up_b + up_s * dotA64(up_w0, up_w1, j * EDIM);
            hid[hbase + t * DIM + d] = v;
            scratch[r_xorig + t * DIM + d] = v;
        }
    }
}
'''

SIGNATURE = r"""
// ---- document signature and FiLM --------------------------------------------
// A linear probe recovers the language from this pooled signature at 84%
// accuracy, and the model was previously only able to use that at the very last
// layer. These two passes compute it once and turn it into a per-layer, per-
// channel scale and shift, so it reaches the depthwise conv and the recurrence.
@compute @workgroup_size(96)
fn sig_pool(@builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let hbase = job.stride * wid.y;
    let r_film = NREG * job.stride * wid.y + 6u * job.stride;

    var s: f32 = 0.0;
    var mx: f32 = -3.4e38;
    for (var t: u32 = 0u; t < T; t = t + 1u) {
        let v = hid[hbase + t * DIM + d];
        s = s + v;
        mx = max(mx, v);
    }
    scratch[r_film + d] = s / max(1.0, f32(T));
    scratch[r_film + DIM + d] = mx;
}

@compute @workgroup_size(96)
fn film(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x;
    let job = jobs[wid.y];
    let r_film = NREG * job.stride * wid.y + 6u * job.stride;

    // Signature into workgroup memory, RMS-normalized. Without the norm the
    // projection below saturates tanh for most units and stops distinguishing
    // documents at all.
    tA[d] = scratch[r_film + d];
    tA[DIM + d] = scratch[r_film + DIM + d];
    workgroupBarrier();
    var ss: f32 = 0.0;
    for (var k: u32 = 0u; k < 2u * DIM; k = k + 1u) { ss = ss + tA[k] * tA[k]; }
    let inv = inverseSqrt(ss / f32(2u * DIM) + 1e-6);
    let n0 = tA[d] * inv * fp[FN_F + d];
    let n1 = tA[DIM + d] * inv * fp[FN_F + DIM + d];
    workgroupBarrier();
    tA[d] = n0;
    tA[DIM + d] = n1;
    workgroupBarrier();
    for (var r = d; r < 192u; r = r + 96u) {
        red[r] = 0.0;
        if (r < FILM_RANK) {
            red[r] = tanh_s(fp[FD_BI + r] + fp[FD_S + r] * dotA192(
                planes[FD_P + r * 6u], planes[FD_P + r * 6u + 1u],
                planes[FD_P + r * 6u + 2u], planes[FD_P + r * 6u + 3u],
                planes[FD_P + r * 6u + 4u], planes[FD_P + r * 6u + 5u], 0u));
        }
    }
    workgroupBarrier();

    // 2 * DIM outputs per layer, spread across the 96 threads.
    let n_out = NLAYER * 2u * DIM;
    for (var o = d; o < n_out; o = o + 96u) {
        let raw = fp[FU_BI + o] + fp[FU_S + o] * __FILM_UP_DOT__;
        let l = o / (2u * DIM);
        let rem = o % (2u * DIM);
        let c = rem % DIM;
        let str = fp[FS_F + l * DIM + c];
        // k == 0 is the scale (identity at strength 0), k == 1 the shift.
        let v = select(str * tanh_s(raw), 1.0 + str * tanh_s(raw), rem < DIM);
        scratch[r_film + 2u * DIM + o] = v;
    }
}
"""

# One workgroup: the recurrence is 0.8% of the layer's arithmetic and is
# sequential in t, so spreading it would cost more in carries than it saves.
#
# Two variants: a static per-channel decay (the original design), and a
# dynamic one where the candidate's own magnitude at each token pushes the
# decay toward a hard reset -- a channel carrying a strong signal forgets
# faster, which is what lets a block comment's state clear the instant real
# code resumes. Chosen per layer at generation time from whether the
# checkpoint has reset_f/reset_b parameters, not branched at runtime, since
# the two forms need different tensor offsets baked in as literals.
SCAN_STATIC = r'''
@compute @workgroup_size(96)
fn l{L}_scan(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let sbase = NREG * job.stride * wid.y;
    let r_b = sbase + job.stride;
    let r_fw = sbase + 2u * job.stride;
    let r_bw = sbase + 3u * job.stride;

    let af = sigmoid_s(fp[{DF_F}u + d]);
    let ab = sigmoid_s(fp[{DB_F}u + d]);
    var hf: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        hf = af * hf + scratch[r_b + t * DIM + d];
        scratch[r_fw + t * DIM + d] = hf;
    }}
    var hb: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let ti = T - 1u - t;
        hb = ab * hb + scratch[r_b + ti * DIM + d];
        scratch[r_bw + ti * DIM + d] = hb;
    }}
}}
'''

SCAN_DYNAMIC = r'''
@compute @workgroup_size(96)
fn l{L}_scan(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let sbase = NREG * job.stride * wid.y;
    let r_b = sbase + job.stride;
    let r_fw = sbase + 2u * job.stride;
    let r_bw = sbase + 3u * job.stride;

    let decay_f = fp[{DF_F}u + d];
    let decay_b = fp[{DB_F}u + d];
    let rf = softplus_s(fp[{RF_F}u + d]);
    let rb = softplus_s(fp[{RB_F}u + d]);
    var hf: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let bt = scratch[r_b + t * DIM + d];
        let aft = sigmoid_s(decay_f - rf * abs(bt));
        hf = aft * hf + bt;
        scratch[r_fw + t * DIM + d] = hf;
    }}
    var hb: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let ti = T - 1u - t;
        let bt = scratch[r_b + ti * DIM + d];
        let abt = sigmoid_s(decay_b - rb * abs(bt));
        hb = abt * hb + bt;
        scratch[r_bw + ti * DIM + d] = hb;
    }}
}}
'''

SCAN_SELECTIVE = r'''
@compute @workgroup_size(96)
fn l{L}_scan(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let sbase = NREG * job.stride * wid.y;
    let r_b = sbase + job.stride;
    let r_fw = sbase + 2u * job.stride;
    let r_bw = sbase + 3u * job.stride;

    let base_f = sigmoid_s(fp[{DF_F}u + d]);
    let base_b = sigmoid_s(fp[{DB_F}u + d]);
    var hf: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let bt = scratch[r_b + t * DIM + d];
        let aft = base_f * (1.0 - scratch[r_fw + t * DIM + d]);
        hf = aft * hf + bt;
        scratch[r_fw + t * DIM + d] = hf;
    }}
    var hb: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let ti = T - 1u - t;
        let bt = scratch[r_b + ti * DIM + d];
        let abt = base_b * (1.0 - scratch[r_bw + ti * DIM + d]);
        hb = abt * hb + bt;
        scratch[r_bw + ti * DIM + d] = hb;
    }}
}}
'''


def _erase_body(tensors: dict, layer: int) -> str:
    down = tensors[f'layers.{layer}.erase_down']
    up = tensors[f'layers.{layer}.erase_up']
    rank = down['shape'][0]
    body = r'''
    // Low-rank input-conditioned erase. tB is zero-padded to 32 values so the
    // packed up projection remains word-aligned for ranks below 32.
    if (d < 32u) {
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            let t = base_t + j;
            var v: f32 = 0.0;
            if (t < T && d < __ERANK__u) {
                let raw = fp[__ED_BI__u + d] + fp[__ED_S__u + d] * dotA96(
                    planes[__ED_P__u + d * 3u], planes[__ED_P__u + d * 3u + 1u],
                    planes[__ED_P__u + d * 3u + 2u],
                    j * DIM);
                v = tanh_s(raw);
            }
            tB[j * 2u * DIM + d] = v;
        }
    }
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {
        let t = base_t + j;
        if (t < T) {
            let ef = fp[__EU_BI__u + d] + fp[__EU_S__u + d] * dotB32(
                planes[__EU_P__u + d], j * 2u * DIM);
            let eb = fp[__EU_BI__u + DIM + d] + fp[__EU_S__u + DIM + d] * dotB32(
                planes[__EU_P__u + DIM + d], j * 2u * DIM);
            scratch[r_fw + t * DIM + d] = sigmoid_s(ef);
            scratch[r_bw + t * DIM + d] = sigmoid_s(eb);
        }
    }
    workgroupBarrier();
'''
    replacements = {
        '__ERANK__': str(rank),
        '__ED_P__': str(down['plane_offset']),
        '__ED_S__': str(down['scale_offset']),
        '__ED_BI__': str(down['bias_offset']),
        '__EU_P__': str(up['plane_offset']),
        '__EU_S__': str(up['scale_offset']),
        '__EU_BI__': str(up['bias_offset']),
    }
    for key, value in replacements.items():
        body = body.replace(key, value)
    return body

LAYER_TEMPLATE = r'''
// ---- layer {L} -------------------------------------------------------------

@compute @workgroup_size(96)
fn l{L}_pre(@builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let hbase = job.stride * wid.y;
    let sbase = NREG * job.stride * wid.y;
    let r_nrm = sbase;
    let r_film = sbase + 6u * job.stride;
    let base_t = wid.x * TILE;

    let gain = fp[{NORM_F}u + d];
    // FiLM for this layer: scale and shift produced from the document signature.
    let gamma = scratch[r_film + 2u * DIM + ({L}u * 2u + 0u) * DIM + d];
    let beta = scratch[r_film + 2u * DIM + ({L}u * 2u + 1u) * DIM + d];
    // Normalize this tile plus its two-token halo, so the depthwise window has
    // the values it needs without a separate pass over the whole sequence.
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {{
        let t = base_t + j;
        tA[j * DIM + d] = select(0.0, hid[hbase + t * DIM + d], t < T);
    }}
    workgroupBarrier();
    tile_rms(d, TILE);
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {{
        let t = base_t + j;
        if (t < T) {{
            scratch[r_nrm + t * DIM + d] =
                (tA[j * DIM + d] * red[j] * gain) * gamma + beta;
        }}
    }}
}}

@compute @workgroup_size(96)
fn l{L}_conv(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let sbase = NREG * job.stride * wid.y;
    let r_nrm = sbase;
    let r_b = sbase + job.stride;
    let r_fw = sbase + 2u * job.stride;
    let r_bw = sbase + 3u * job.stride;
    let base_t = wid.x * TILE;

    let pic0 = planes[{PI_P}u + d * 3u];
    let pic1 = planes[{PI_P}u + d * 3u + 1u];
    let pic2 = planes[{PI_P}u + d * 3u + 2u];
    let pig0 = planes[{PI_P}u + (DIM + d) * 3u];
    let pig1 = planes[{PI_P}u + (DIM + d) * 3u + 1u];
    let pig2 = planes[{PI_P}u + (DIM + d) * 3u + 2u];
    let sc = fp[{PI_S}u + d];
    let sg = fp[{PI_S}u + DIM + d];
    let bc = fp[{PI_BI}u + d];
    let bg = fp[{PI_BI}u + DIM + d];
    let dbias = fp[{DW_BI}u + d];
    let dws = fp[{DW_S}u + d];
    var dwv: array<f32, KSIZE>;
    for (var k: u32 = 0u; k < KSIZE; k = k + 1u) {{
        dwv[k] = wgt({DW_P}u, {DW_W}u, {DW_B}u, d * KSIZE + k, dws);
    }}

    for (var j: u32 = 0u; j < TILE; j = j + 1u) {{
        let t = base_t + j;
        var acc: f32 = 0.0;
        if (t < T) {{
            acc = dbias;
            for (var k: u32 = 0u; k < KSIZE; k = k + 1u) {{
                let o = i32(t) + (i32(k) - i32(KSIZE / 2u)) * {DIL};
                if (o >= 0 && o < i32(T)) {{
                    acc = acc + dwv[k] * scratch[r_nrm + u32(o) * DIM + d];
                }}
            }}
        }}
        tA[j * DIM + d] = acc;
    }}
    workgroupBarrier();
{ERASE_BODY}
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {{
        let t = base_t + j;
        if (t < T) {{
            let cand = bc + sc * dotA96(pic0, pic1, pic2, j * DIM);
            let gate = bg + sg * dotA96(pig0, pig1, pig2, j * DIM);
            scratch[r_b + t * DIM + d] = tanh_s(cand) * sigmoid_s(gate);
        }}
    }}
}}

{SCAN_BODY}

@compute @workgroup_size(96)
fn l{L}_post(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let hbase = job.stride * wid.y;
    let sbase = NREG * job.stride * wid.y;
    let r_fw = sbase + 2u * job.stride;
    let r_bw = sbase + 3u * job.stride;
    let base_t = wid.x * TILE;

    let po0 = planes[{PO_P}u + d * 6u];
    let po1 = planes[{PO_P}u + d * 6u + 1u];
    let po2 = planes[{PO_P}u + d * 6u + 2u];
    let po3 = planes[{PO_P}u + d * 6u + 3u];
    let po4 = planes[{PO_P}u + d * 6u + 4u];
    let po5 = planes[{PO_P}u + d * 6u + 5u];
    let spo = fp[{PO_S}u + d];
    let bpo = fp[{PO_BI}u + d];
    let og = sigmoid_s(fp[{OG_F}u + d]);

    for (var j: u32 = 0u; j < TILE; j = j + 1u) {{
        let t = base_t + j;
        if (t < T) {{
            tB[j * 2u * DIM + d]       = scratch[r_fw + t * DIM + d];
            tB[j * 2u * DIM + DIM + d] = scratch[r_bw + t * DIM + d];
        }}
    }}
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {{
        let t = base_t + j;
        if (t < T) {{
            let y = bpo + spo * dotB192(po0, po1, po2, po3, po4, po5, j * 2u * DIM);
            hid[hbase + t * DIM + d] = hid[hbase + t * DIM + d] + y * og;
        }}
    }}
}}
'''

# Four pooled views: mean and max are constant across the sequence; the prefix
# and suffix means vary with position, which is what lets the model tell "this
# identifier was declared earlier" from "it appears out of nowhere". The running
# sums are sequential in t, so they share the single-workgroup pass.
#
# Two variants, chosen at generation time by whether the checkpoint has a
# global_ctx.decl_gate parameter: a plain running mean, or one gated per
# channel by the pooled value itself, so a declaration can outweigh
# incidental earlier mentions of the same shape rather than being averaged
# down by them.
POOL_REDUCE_STATIC = r'''
// ---- file-scale context ----------------------------------------------------
@compute @workgroup_size(96)
fn pool_reduce(@builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let hbase = job.stride * wid.y;
    let sbase = NREG * job.stride * wid.y;
    let r_pre = sbase;                       // reuses the normalized-input region
    let r_suf = sbase + job.stride;          // reuses the scan-input region
    let r_stat = sbase + 5u * job.stride;

    var s: f32 = 0.0;
    var mx: f32 = -3.4e38;
    for (var t: u32 = 0u; t < T; t = t + 1u) {
        let v = hid[hbase + t * DIM + d];
        s = s + v;
        mx = max(mx, v);
        scratch[r_pre + t * DIM + d] = s / f32(t + 1u);
    }
    var sb: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {
        let ti = T - 1u - t;
        sb = sb + hid[hbase + ti * DIM + d];
        scratch[r_suf + ti * DIM + d] = sb / f32(t + 1u);
    }
    scratch[r_stat + d] = s / max(1.0, f32(T));
    scratch[r_stat + DIM + d] = mx;
}
'''

POOL_REDUCE_DYNAMIC = r'''
// ---- file-scale context ----------------------------------------------------
@compute @workgroup_size(96)
fn pool_reduce(@builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let hbase = job.stride * wid.y;
    let sbase = NREG * job.stride * wid.y;
    let r_pre = sbase;                       // reuses the normalized-input region
    let r_suf = sbase + job.stride;          // reuses the scan-input region
    let r_stat = sbase + 5u * job.stride;

    let dgw = fp[{DG_F}u + d];
    var s: f32 = 0.0;    // plain sum, still feeds the sequence-mean stat below
    var sg: f32 = 0.0;   // decl-gated sum, feeds the prefix mean only
    var sdg: f32 = 0.0;
    var mx: f32 = -3.4e38;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let v = hid[hbase + t * DIM + d];
        s = s + v;
        mx = max(mx, v);
        let dg = sigmoid_s(v * dgw);
        sg = sg + v * dg;
        sdg = sdg + dg;
        scratch[r_pre + t * DIM + d] = sg / max(1.0, sdg);
    }}
    var sb: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let ti = T - 1u - t;
        sb = sb + hid[hbase + ti * DIM + d];
        scratch[r_suf + ti * DIM + d] = sb / f32(t + 1u);
    }}
    scratch[r_stat + d] = s / max(1.0, f32(T));
    scratch[r_stat + DIM + d] = mx;
}}
'''

POOL_HEAD = r'''
@compute @workgroup_size(96)
fn pool_apply(@builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let hbase = job.stride * wid.y;
    let sbase = NREG * job.stride * wid.y;
    let r_pre = sbase;
    let r_suf = sbase + job.stride;
    let r_ctx = sbase + 4u * job.stride;
    let r_stat = sbase + 5u * job.stride;
    let base_t = wid.x * TILE_POOL;

    let gs0 = planes[GS_P + d * 12u];
    let gs1 = planes[GS_P + d * 12u + 1u];
    let gs2 = planes[GS_P + d * 12u + 2u];
    let gs3 = planes[GS_P + d * 12u + 3u];
    let gs4 = planes[GS_P + d * 12u + 4u];
    let gs5 = planes[GS_P + d * 12u + 5u];
    let gs6 = planes[GS_P + d * 12u + 6u];
    let gs7 = planes[GS_P + d * 12u + 7u];
    let gs8 = planes[GS_P + d * 12u + 8u];
    let gs9 = planes[GS_P + d * 12u + 9u];
    let gs10 = planes[GS_P + d * 12u + 10u];
    let gs11 = planes[GS_P + d * 12u + 11u];
    let gss = fp[GS_S + d];
    let gsb = fp[GS_BI + d];
    let gg0 = planes[GG_P + d * 3u];
    let gg1 = planes[GG_P + d * 3u + 1u];
    let gg2 = planes[GG_P + d * 3u + 2u];
    let ggs = fp[GG_S + d];
    let ggb = fp[GG_BI + d];

    for (var j: u32 = 0u; j < TILE_POOL; j = j + 1u) {
        let t = base_t + j;
        let ok = t < T;
        tP[j * 4u * DIM + d]             = scratch[r_stat + d];
        tP[j * 4u * DIM + DIM + d]       = scratch[r_stat + DIM + d];
        tP[j * 4u * DIM + 2u * DIM + d]  = select(0.0, scratch[r_pre + t * DIM + d], ok);
        tP[j * 4u * DIM + 3u * DIM + d]  = select(0.0, scratch[r_suf + t * DIM + d], ok);
        tA[j * DIM + d] = select(0.0, hid[hbase + t * DIM + d], ok);
    }
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE_POOL; j = j + 1u) {
        let t = base_t + j;
        if (t < T) {
            let b = j * 4u * DIM;
            var acc: f32 = 0.0;
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs0, k, tP[b + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs1, k, tP[b + 32u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs2, k, tP[b + 64u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs3, k, tP[b + 96u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs4, k, tP[b + 128u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs5, k, tP[b + 160u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs6, k, tP[b + 192u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs7, k, tP[b + 224u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs8, k, tP[b + 256u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs9, k, tP[b + 288u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs10, k, tP[b + 320u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs11, k, tP[b + 352u + k]); }
            let ctx = tanh_s(gsb + gss * acc);
            let g = sigmoid_s(ggb + ggs * dotA96(gg0, gg1, gg2, j * DIM));
            scratch[r_ctx + t * DIM + d] = ctx * g;
        }
    }
}

// ---- classifier head -------------------------------------------------------
@compute @workgroup_size(96)
fn head(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let off = job.token_offset;
    let hbase = job.stride * wid.y;
    let sbase = NREG * job.stride * wid.y;
    let r_ctx = sbase + 4u * job.stride;
    let r_xorig = sbase + 7u * job.stride;
    let base_t = wid.x * TILE;

    let hgain = fp[HNORM_F + d];
    let o0 = d;
    let o1 = DIM + d;
    let h0a = planes[HH_P + o0 * 6u];
    let h0b = planes[HH_P + o0 * 6u + 1u];
    let h0c = planes[HH_P + o0 * 6u + 2u];
    let h0d = planes[HH_P + o0 * 6u + 3u];
    let h0e = planes[HH_P + o0 * 6u + 4u];
    let h0f = planes[HH_P + o0 * 6u + 5u];
    var h1a: u32 = 0u; var h1b: u32 = 0u; var h1c: u32 = 0u; var h1d: u32 = 0u;
    var h1e: u32 = 0u; var h1f: u32 = 0u;
    if (o1 < HEAD_HIDDEN) {
        h1a = planes[HH_P + o1 * 6u];
        h1b = planes[HH_P + o1 * 6u + 1u];
        h1c = planes[HH_P + o1 * 6u + 2u];
        h1d = planes[HH_P + o1 * 6u + 3u];
        h1e = planes[HH_P + o1 * 6u + 4u];
        h1f = planes[HH_P + o1 * 6u + 5u];
    }
    // head_out is 9 rows; thread c owns class c, so the final matmul is nine
    // parallel dots rather than one thread doing all of them.
    var ho0: u32 = 0u; var ho1: u32 = 0u; var ho2: u32 = 0u;
    var ho3: u32 = 0u; var ho4: u32 = 0u; var ho5: u32 = 0u;
    if (d < NCLASS) {
        ho0 = planes[HO_P + d * 6u];
        ho1 = planes[HO_P + d * 6u + 1u];
        ho2 = planes[HO_P + d * 6u + 2u];
        ho3 = planes[HO_P + d * 6u + 3u];
        ho4 = planes[HO_P + d * 6u + 4u];
        ho5 = planes[HO_P + d * 6u + 5u];
    }

    // The pre-layer embedding folded back in at strength __HW_TERM__ -- zero
    // unless the checkpoint has a highway_scale parameter, in which case it
    // gives the classifier a direct path to the raw token identity alongside
    // whatever the recurrent layers built on top of it.
    let hw: f32 = __HW_TERM__;
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {
        let t = base_t + j;
        let xv = select(0.0, hid[hbase + t * DIM + d], t < T);
        let xo = select(0.0, scratch[r_xorig + t * DIM + d], t < T);
        tA[j * DIM + d] = xv + hw * xo;
    }
    workgroupBarrier();
    tile_rms(d, TILE);
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {
        let t = base_t + j;
        tB[j * 2u * DIM + d] = tA[j * DIM + d] * red[j] * hgain;
        tB[j * 2u * DIM + DIM + d] = select(0.0, scratch[r_ctx + t * DIM + d], t < T);
    }
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE; j = j + 1u) {
        let a0 = fp[HH_BI + o0] + fp[HH_S + o0] * dotB192(h0a, h0b, h0c, h0d, h0e, h0f, j * 2u * DIM);
        let c0 = 0.7978845608 * (a0 + 0.044715 * a0 * a0 * a0);
        tA[j * HEAD_HIDDEN + o0] = 0.5 * a0 * (1.0 + tanh_s(c0));
        if (o1 < HEAD_HIDDEN) {
            let a1 = fp[HH_BI + o1] + fp[HH_S + o1] * dotB192(h1a, h1b, h1c, h1d, h1e, h1f, j * 2u * DIM);
            let c1 = 0.7978845608 * (a1 + 0.044715 * a1 * a1 * a1);
            tA[j * HEAD_HIDDEN + o1] = 0.5 * a1 * (1.0 + tanh_s(c1));
        }
    }
    workgroupBarrier();
    if (d < NCLASS) {
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            lg[j * NCLASS + d] = fp[HO_BI + d] + fp[HO_S + d]
                * dotA192(ho0, ho1, ho2, ho3, ho4, ho5, j * HEAD_HIDDEN);
        }
    }
    workgroupBarrier();
    if (d < TILE) {
        let t = base_t + d;
        if (t < T) {
            var best: u32 = 0u;
            var bestv: f32 = lg[d * NCLASS];
            for (var c: u32 = 1u; c < NCLASS; c = c + 1u) {
                let v = lg[d * NCLASS + c];
                if (v > bestv) { bestv = v; best = c; }
            }
            out_cls[off + t] = best;
        }
    }
}
'''


MIXED_DOTS = r'''
fn dotQA(base: u32, wpp: u32, bits: u32, row: u32, act: u32, width: u32) -> f32 {
    var value = 0.0;
    for (var k = 0u; k < width; k = k + 1u) {
        value = value + wgt(base, wpp, bits, row + k, 1.0) * tA[act + k];
    }
    return value;
}
fn dotQB(base: u32, wpp: u32, bits: u32, row: u32, act: u32, width: u32) -> f32 {
    var value = 0.0;
    for (var k = 0u; k < width; k = k + 1u) {
        value = value + wgt(base, wpp, bits, row + k, 1.0) * tB[act + k];
    }
    return value;
}
fn dotQRed(base: u32, wpp: u32, bits: u32, row: u32, width: u32) -> f32 {
    var value = 0.0;
    for (var k = 0u; k < width; k = k + 1u) {
        value = value + wgt(base, wpp, bits, row + k, 1.0) * red[k];
    }
    return value;
}
'''


def generate(meta: dict) -> str:
    t = meta['tensors']
    cfg = meta['config']
    if (cfg['dim'], cfg['embed_dim'], cfg['head_hidden']) != (96, 64, 192):
        raise ValueError('large shader requires dim=96, embed_dim=64, head_hidden=192')
    if not 0 < cfg['film_rank'] <= 192:
        raise ValueError('large shader requires 0 < film_rank <= 192')
    # These paths still use the optimized binary dot products below.
    for name, entry in t.items():
        if entry['kind'] == 'linear' and not name.endswith('.dw') and name not in (
                'embedding.up', 'head_hidden', 'head_out') and entry['bits'] != 1:
            raise ValueError(f'{name}: large shader currently requires binary projections')
    n = cfg['n_layers']
    dilations = cfg.get('dilations', (1, 2, 4))
    film_up = t['film.up']
    film_up_words = film_up['words_per_row']
    if film_up_words == 1:
        film_up_dot = 'dotRed32(planes[FU_P + o])'
    elif film_up_words == 2:
        film_up_dot = ('dotRed64(planes[FU_P + o * 2u], '
                       'planes[FU_P + o * 2u + 1u])')
    else:
        film_up_dot = f'dotQRed(FU_P, FU_W, FU_B, o * {film_up_words * 32}u, FILM_RANK)'
    signature = SIGNATURE.replace('__FILM_UP_DOT__', film_up_dot)

    # Keep the fast binary paths for existing artifacts; decode extra planes
    # only for the explicitly mixed-precision input and classifier tensors.
    embed = EMBED
    head = POOL_HEAD
    if 'group_sizes' not in t['embedding.table']:
        embed = embed.replace('fp[EMB_S + f]', 'fp[EMB_S + row]')
        embed = embed.replace('fp[EMB_S + NFIELD]', 'fp[EMB_S + FLAG_ROW + b]')
    if t['embedding.up']['bits'] > 1:
        embed = embed.replace('dotA64(up_w0, up_w1, j * EDIM)',
                              'dotQA(UP_P, UP_W, UP_B, d * EDIM, j * EDIM, EDIM)')
    if t['head_hidden']['bits'] > 1:
        for idx in (0, 1):
            head = head.replace(
                f'dotB192(h{idx}a, h{idx}b, h{idx}c, h{idx}d, h{idx}e, h{idx}f, j * 2u * DIM)',
                f'dotQB(HH_P, HH_W, HH_B, o{idx} * 2u * DIM, j * 2u * DIM, 2u * DIM)')
    if t['head_out']['bits'] > 1:
        head = head.replace('dotA192(ho0, ho1, ho2, ho3, ho4, ho5, j * HEAD_HIDDEN)',
                            'dotQA(HO_P, HO_W, HO_B, d * HEAD_HIDDEN, j * HEAD_HIDDEN, HEAD_HIDDEN)')

    parts = ['// Generated by wgsl.py -- do not edit.\n'
             '// Multi-entry-point pipeline; see the module docstring for the split.\n',
             _consts(meta), PRELUDE, MIXED_DOTS, embed, signature]
    for i in range(n):
        if f'layers.{i}.erase_down' in t:
            scan_body = SCAN_SELECTIVE.format(
                L=i,
                DF_F=t[f'layers.{i}.decay_f']['f16_offset'],
                DB_F=t[f'layers.{i}.decay_b']['f16_offset'],
            )
            erase_body = _erase_body(t, i)
        elif f'layers.{i}.reset_f' in t:
            scan_body = SCAN_DYNAMIC.format(
                L=i,
                DF_F=t[f'layers.{i}.decay_f']['f16_offset'],
                DB_F=t[f'layers.{i}.decay_b']['f16_offset'],
                RF_F=t[f'layers.{i}.reset_f']['f16_offset'],
                RB_F=t[f'layers.{i}.reset_b']['f16_offset'],
            )
            erase_body = ''
        else:
            scan_body = SCAN_STATIC.format(
                L=i,
                DF_F=t[f'layers.{i}.decay_f']['f16_offset'],
                DB_F=t[f'layers.{i}.decay_b']['f16_offset'],
            )
            erase_body = ''
        dil = dilations[i] if i < len(dilations) else 1
        parts.append(LAYER_TEMPLATE.format(
            L=i,
            DIL=dil,
            SCAN_BODY=scan_body,
            ERASE_BODY=erase_body,
            NORM_F=t[f'layers.{i}.norm.weight']['f16_offset'],
            OG_F=t[f'layers.{i}.out_gate']['f16_offset'],
            DW_P=t[f'layers.{i}.dw']['plane_offset'],
            DW_W=t[f'layers.{i}.dw']['words_per_plane'],
            DW_B=t[f'layers.{i}.dw']['bits'],
            DW_S=t[f'layers.{i}.dw']['scale_offset'],
            DW_BI=t[f'layers.{i}.dw']['bias_offset'],
            PI_P=t[f'layers.{i}.proj_in']['plane_offset'],
            PI_S=t[f'layers.{i}.proj_in']['scale_offset'],
            PI_BI=t[f'layers.{i}.proj_in']['bias_offset'],
            PO_P=t[f'layers.{i}.proj_out']['plane_offset'],
            PO_S=t[f'layers.{i}.proj_out']['scale_offset'],
            PO_BI=t[f'layers.{i}.proj_out']['bias_offset'],
        ))

    if 'global_ctx.decl_gate' in t:
        parts.append(POOL_REDUCE_DYNAMIC.format(DG_F=t['global_ctx.decl_gate']['f16_offset']))
    else:
        parts.append(POOL_REDUCE_STATIC)

    hw_term = 'fp[HW_F]' if 'highway_scale' in t else '0.0'
    parts.append(head.replace('__HW_TERM__', hw_term))
    return '\n'.join(parts)


def pipeline_order(n_layers: int) -> list[dict]:
    """Entry points in execution order, and how each one is dispatched.

    `tiled` passes get one workgroup per tile of tokens; the others need the
    whole sequence in one workgroup because they are sequential in t.
    """
    steps: list[dict] = [
        {'entry': 'embed', 'tile': TILE},
        {'entry': 'sig_pool', 'tile': 0},
        {'entry': 'film', 'tile': 0},
    ]
    for i in range(n_layers):
        steps += [
            {'entry': f'l{i}_pre', 'tile': TILE},
            {'entry': f'l{i}_conv', 'tile': TILE},
            {'entry': f'l{i}_scan', 'tile': 0},
            {'entry': f'l{i}_post', 'tile': TILE},
        ]
    steps += [
        {'entry': 'pool_reduce', 'tile': 0},
        {'entry': 'pool_apply', 'tile': TILE_POOL},
        {'entry': 'head', 'tile': TILE},
    ]
    return steps


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='./checkpoints/weights.meta.json')
    ap.add_argument('--out', default='./webgpu/shader.wgsl')
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    src = generate(meta)
    Path(args.out).write_text(src)
    print(f'wrote {args.out} ({len(src)} bytes, '
          f'{len(pipeline_order(meta["config"]["n_layers"]))} dispatches)')


if __name__ == '__main__':
    main()
