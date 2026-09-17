"""
Generates the WGSL compute shaders from the exported weight metadata.

The kernel is split into several entry points rather than one fused dispatch,
because one workgroup is 64 lanes of a GPU that has thousands. The split follows
where the work actually is. Per token per layer:

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
indirection to find a weight. The generator targets lex-lite (feature version 2:
three packed words per token, selective-erase recurrence, declaration-gated
prefix/suffix context); `minify` produces what ships in lex/src/shader.js.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TILE = 16        # tokens per workgroup for the token-parallel passes
TILE_POOL = 4    # smaller tile for the pooled-context pass; its rows are 2*DIM
MAX_JOBS = 64    # uniform job array needs a compile-time size
N_REGIONS = 8    # per job: nrm, b, fwd, bwd, ctx, (spare), film, x_orig.
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
    lines.append(f"const DG_F: u32 = {t['global_ctx.decl_gate']['f16_offset']}u;")
    lines.append(f"const HW_F: u32 = {t['highway_scale']['f16_offset']}u;")

    fo = meta['field_offsets']
    order = list(meta['field_sizes'].keys())
    lines.append('const FIELD_ROW: array<u32, %d> = array<u32, %d>(%s);' % (
        len(order), len(order), ', '.join(f'{fo[k]}u' for k in order)))
    lines.append(f"const FLAG_ROW: u32 = {meta['flag_offset']}u;")
    # Three packed u32 per token; the third holds gap/indent/line/depth.
    lines.append("const WPT: u32 = 3u;")
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

var<workgroup> tA  : array<f32, 1536>;   // TILE x DIM, reused as TILE x HEAD_HIDDEN
var<workgroup> tB  : array<f32, 2048>;   // TILE x 2*DIM
var<workgroup> tP  : array<f32, 1024>;   // TILE_POOL x 2*DIM (sized for 4*DIM)
var<workgroup> red : array<f32, 64>;
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

fn tw0(o: u32, i: u32) -> u32 { return tokens[(o + i) * WPT]; }
fn tw1(o: u32, i: u32) -> u32 { return tokens[(o + i) * WPT + 1u]; }
fn tw2(o: u32, i: u32) -> u32 { return tokens[(o + i) * WPT + 2u]; }
fn tok_kind(o: u32, i: u32)  -> u32 { return  tw0(o, i)        & 3u; }
fn tok_flags(o: u32, i: u32) -> u32 { return (tw0(o, i) >> 19u) & 255u; }

fn field_id(o: u32, i: u32, f: u32) -> u32 {
    switch f {
        case 0u:  { return  tw0(o, i)        & 3u; }
        case 1u:  { return (tw0(o, i) >>  2u) & 7u; }
        case 2u:  { return (tw0(o, i) >>  5u) & 127u; }
        case 3u:  { return (tw0(o, i) >> 12u) & 127u; }
        case 4u:  { return  tw1(o, i)        & 1023u; }
        case 5u:  { return (tw1(o, i) >> 10u) & 127u; }
        case 6u:  { return (tw1(o, i) >> 17u) & 15u; }
        case 7u:  { return (tw1(o, i) >> 21u) & 15u; }
        case 8u:  { return (tw1(o, i) >> 25u) & 31u; }
        case 9u:  { return (tw0(o, i) >> 27u) & 31u; }
        case 10u: { return  tw2(o, i)        & 3u; }
        case 11u: { return (tw2(o, i) >>  2u) & 3u; }
        case 12u: { return (tw2(o, i) >>  4u) & 7u; }
        case 13u: { return (tw2(o, i) >>  7u) & 31u; }
        case 14u: { return (tw2(o, i) >> 12u) & 3u; }
        default:  { return (tw2(o, i) >> 14u) & 3u; }
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
@compute @workgroup_size(64)
fn embed(@builtin(local_invocation_id) lid: vec3<u32>,
         @builtin(workgroup_id) wid: vec3<u32>) {
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let off = job.token_offset;
    let hbase = job.stride * wid.y;
    let r_xorig = NREG * job.stride * wid.y + 7u * job.stride;
    let base_t = wid.x * TILE;

    let up_w = planes[UP_P + d];
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
            let v = up_b + up_s * dotA32(up_w, j * EDIM);
            hid[hbase + t * DIM + d] = v;
            scratch[r_xorig + t * DIM + d] = v;
        }
    }
}
'''

SIGNATURE = r"""
// ---- document signature and FiLM --------------------------------------------
// The pooled signature is strongly language-separable. These two passes compute
// it once and turn it into a per-layer, per-channel scale and shift, so language
// identity reaches the depthwise conv and the recurrence.
@compute @workgroup_size(64)
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

@compute @workgroup_size(64)
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
    if (d < FILM_RANK) {
        red[d] = tanh_s(fp[FD_BI + d] + fp[FD_S + d] * dotA128(
            planes[FD_P + d * 4u], planes[FD_P + d * 4u + 1u],
            planes[FD_P + d * 4u + 2u], planes[FD_P + d * 4u + 3u], 0u));
    } else {
        red[d] = 0.0;
    }
    workgroupBarrier();

    // 2 * DIM outputs per layer, spread across the 64 threads.
    let n_out = NLAYER * 2u * DIM;
    for (var o = d; o < n_out; o = o + 64u) {
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
# The per-token decay is the learned per-channel base times (1 - erase), where
# erase comes from the layer's low-rank gate (computed in l{L}_conv's erase
# body), so closing delimiters can clear string/comment state.
SCAN_SELECTIVE = r'''
@compute @workgroup_size(64)
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
                let raw = fp[__ED_BI__u + d] + fp[__ED_S__u + d] * dotA64(
                    planes[__ED_P__u + d * 2u], planes[__ED_P__u + d * 2u + 1u],
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

@compute @workgroup_size(64)
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

@compute @workgroup_size(64)
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

    let pic0 = planes[{PI_P}u + d * 2u];
    let pic1 = planes[{PI_P}u + d * 2u + 1u];
    let pig0 = planes[{PI_P}u + (DIM + d) * 2u];
    let pig1 = planes[{PI_P}u + (DIM + d) * 2u + 1u];
    let sc = fp[{PI_S}u + d];
    let sg = fp[{PI_S}u + DIM + d];
    let bc = fp[{PI_BI}u + d];
    let bg = fp[{PI_BI}u + DIM + d];
    let dbias = fp[{DW_BI}u + d];
    let dws = fp[{DW_S}u + d];
    var dwv: array<f32, 5>;
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
            let cand = bc + sc * dotA64(pic0, pic1, j * DIM);
            let gate = bg + sg * dotA64(pig0, pig1, j * DIM);
            scratch[r_b + t * DIM + d] = tanh_s(cand) * sigmoid_s(gate);
        }}
    }}
}}

{SCAN_BODY}

@compute @workgroup_size(64)
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

    let po0 = planes[{PO_P}u + d * 4u];
    let po1 = planes[{PO_P}u + d * 4u + 1u];
    let po2 = planes[{PO_P}u + d * 4u + 2u];
    let po3 = planes[{PO_P}u + d * 4u + 3u];
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
            let y = bpo + spo * dotB128(po0, po1, po2, po3, j * 2u * DIM);
            hid[hbase + t * DIM + d] = hid[hbase + t * DIM + d] + y * og;
        }}
    }}
}}
'''

# Prefix and suffix means vary with position, which is what lets the model tell
# "this identifier was declared earlier" from "it appears out of nowhere". The
# prefix mean is gated per channel by the pooled value itself, so a declaration
# can outweigh incidental earlier mentions of the same shape. The running sums
# are sequential in t, so they share the single-workgroup pass.
POOL_REDUCE = r'''
// ---- file-scale context ----------------------------------------------------
@compute @workgroup_size(64)
fn pool_reduce(@builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let job = jobs[wid.y];
    let T = job.token_count;
    let hbase = job.stride * wid.y;
    let sbase = NREG * job.stride * wid.y;
    let r_pre = sbase;                       // reuses the normalized-input region
    let r_suf = sbase + job.stride;          // reuses the scan-input region

    let dgw = fp[{DG_F}u + d];
    var sg: f32 = 0.0;
    var sdg: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {{
        let v = hid[hbase + t * DIM + d];
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
}}
'''

POOL_HEAD = r'''
@compute @workgroup_size(64)
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
    let base_t = wid.x * TILE_POOL;

    let gs0 = planes[GS_P + d * 4u];
    let gs1 = planes[GS_P + d * 4u + 1u];
    let gs2 = planes[GS_P + d * 4u + 2u];
    let gs3 = planes[GS_P + d * 4u + 3u];
    let gss = fp[GS_S + d];
    let gsb = fp[GS_BI + d];
    let gg0 = planes[GG_P + d * 2u];
    let gg1 = planes[GG_P + d * 2u + 1u];
    let ggs = fp[GG_S + d];
    let ggb = fp[GG_BI + d];

    for (var j: u32 = 0u; j < TILE_POOL; j = j + 1u) {
        let t = base_t + j;
        let ok = t < T;
        tP[j * 2u * DIM + d]       = select(0.0, scratch[r_pre + t * DIM + d], ok);
        tP[j * 2u * DIM + DIM + d] = select(0.0, scratch[r_suf + t * DIM + d], ok);
        tA[j * DIM + d] = select(0.0, hid[hbase + t * DIM + d], ok);
    }
    workgroupBarrier();
    for (var j: u32 = 0u; j < TILE_POOL; j = j + 1u) {
        let t = base_t + j;
        if (t < T) {
            let b = j * 2u * DIM;
            var acc: f32 = 0.0;
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs0, k, tP[b + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs1, k, tP[b + 32u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs2, k, tP[b + 64u + k]); }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) { acc = acc + m1(gs3, k, tP[b + 96u + k]); }
            let ctx = tanh_s(gsb + gss * acc);
            let g = sigmoid_s(ggb + ggs * dotA64(gg0, gg1, j * DIM));
            scratch[r_ctx + t * DIM + d] = ctx * g;
        }
    }
}

// ---- classifier head -------------------------------------------------------
@compute @workgroup_size(64)
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
    let h0a = planes[HH_P + o0 * 4u];
    let h0b = planes[HH_P + o0 * 4u + 1u];
    let h0c = planes[HH_P + o0 * 4u + 2u];
    let h0d = planes[HH_P + o0 * 4u + 3u];
    var h1a: u32 = 0u; var h1b: u32 = 0u; var h1c: u32 = 0u; var h1d: u32 = 0u;
    if (o1 < HEAD_HIDDEN) {
        h1a = planes[HH_P + o1 * 4u];
        h1b = planes[HH_P + o1 * 4u + 1u];
        h1c = planes[HH_P + o1 * 4u + 2u];
        h1d = planes[HH_P + o1 * 4u + 3u];
    }
    // head_out is 9 rows; thread c owns class c, so the final matmul is nine
    // parallel dots rather than one thread doing all of them.
    var ho0: u32 = 0u; var ho1: u32 = 0u; var ho2: u32 = 0u;
    if (d < NCLASS) {
        ho0 = planes[HO_P + d * 3u];
        ho1 = planes[HO_P + d * 3u + 1u];
        ho2 = planes[HO_P + d * 3u + 2u];
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
        let a0 = fp[HH_BI + o0] + fp[HH_S + o0] * dotB128(h0a, h0b, h0c, h0d, j * 2u * DIM);
        let c0 = 0.7978845608 * (a0 + 0.044715 * a0 * a0 * a0);
        tA[j * HEAD_HIDDEN + o0] = 0.5 * a0 * (1.0 + tanh_s(c0));
        if (o1 < HEAD_HIDDEN) {
            let a1 = fp[HH_BI + o1] + fp[HH_S + o1] * dotB128(h1a, h1b, h1c, h1d, j * 2u * DIM);
            let c1 = 0.7978845608 * (a1 + 0.044715 * a1 * a1 * a1);
            tA[j * HEAD_HIDDEN + o1] = 0.5 * a1 * (1.0 + tanh_s(c1));
        }
    }
    workgroupBarrier();
    if (d < NCLASS) {
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            lg[j * NCLASS + d] = fp[HO_BI + d] + fp[HO_S + d]
                * dotA96(ho0, ho1, ho2, j * HEAD_HIDDEN);
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


def generate(meta: dict) -> str:
    t = meta['tensors']
    cfg = meta['config']
    if meta.get('feature_version') != 2 or t['global_ctx.summary']['shape'][1] != 2 * cfg['dim']:
        raise ValueError('wgsl.py generates lex-lite (feature version 2, prefix/suffix context)')
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
        raise ValueError(f'unsupported FiLM up row width: {film_up_words} words')
    signature = SIGNATURE.replace('__FILM_UP_DOT__', film_up_dot)

    parts = ['// Generated by wgsl.py -- do not edit.\n'
             '// Multi-entry-point pipeline; see the module docstring for the split.\n',
             _consts(meta), PRELUDE, EMBED, signature]
    for i in range(n):
        scan_body = SCAN_SELECTIVE.format(
            L=i,
            DF_F=t[f'layers.{i}.decay_f']['f16_offset'],
            DB_F=t[f'layers.{i}.decay_b']['f16_offset'],
        )
        erase_body = _erase_body(t, i)
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

    parts.append(POOL_REDUCE.format(DG_F=t['global_ctx.decl_gate']['f16_offset']))
    parts.append(POOL_HEAD.replace('__HW_TERM__', 'fp[HW_F]'))
    return '\n'.join(parts)


_WGSL_TOKEN = re.compile(
    r'\s+|//[^\n]*|/\*.*?\*/'
    r'|(?P<num>0[xX][0-9a-fA-F]+[iu]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fiuh]?)'
    r'|(?P<id>[A-Za-z_][A-Za-z0-9_]*)'
    r'|(?P<op>->|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|[-+*/%&|^]=|[^\s])',
    re.S)
# Names that must survive renaming even if they look declared (`default:`).
_WGSL_KEEP = frozenset('''
    alias array bitcast bool break case const const_assert continue continuing
    default diagnostic discard else enable f16 f32 false fn for function i32 if
    let loop mat2x2 mat3x3 mat4x4 override private ptr read read_write requires
    return storage struct switch true u32 uniform var vec2 vec3 vec4 while
    workgroup compute builtin group binding workgroup_size
'''.split())


def _wgsl_tokens(src: str) -> list[tuple[str, str]]:
    out = []
    pos = 0
    for m in _WGSL_TOKEN.finditer(src):
        if m.start() != pos:
            raise ValueError(f'unlexable WGSL at {pos}: {src[pos:pos + 20]!r}')
        pos = m.end()
        kind = m.lastgroup
        if kind is not None:
            out.append((kind, m.group()))
    return out


def _short_names():
    import itertools
    import string
    first = string.ascii_uppercase
    rest = string.ascii_letters + string.digits
    yield from first
    for n in itertools.count(1):
        for head in first:
            for tail in itertools.product(rest, repeat=n):
                yield head + ''.join(tail)


def minify(src: str, keep: set[str]) -> str:
    """Strip comments and whitespace and shorten the shader's own names.

    A name is renamed when it is declared here (`fn`/`const`/`let`/`var`/
    `struct`/`alias` or `name:` in a parameter, member or typed binding),
    is longer than two characters (so swizzles like `.x` and short locals are
    never touched), is not a WGSL keyword, and is not in `keep` (the entry
    points the runtime creates pipelines for). Renaming is purely lexical and
    applied to every occurrence, including `.member` accesses, so a declared
    name must not coincide with a builtin -- enforced by _WGSL_KEEP and the
    length rule.
    """
    toks = _wgsl_tokens(src)
    declared = {}
    for i, (kind, text) in enumerate(toks):
        if kind != 'id' or len(text) <= 2 or text in _WGSL_KEEP or text in keep:
            continue
        prev = toks[i - 1][1] if i else ''
        nxt = toks[i + 1][1] if i + 1 < len(toks) else ''
        if prev in ('fn', 'const', 'let', 'var', 'struct', 'alias') or nxt == ':' \
                or (prev == '>' and i >= 2 and any(t == 'var' for _, t in toks[max(0, i - 6):i])):
            declared[text] = declared.get(text, 0)
    counts = {}
    for kind, text in toks:
        if kind == 'id' and text in declared:
            counts[text] = counts.get(text, 0) + 1
    taken = {t for k, t in toks if k == 'id'}
    names = (n for n in _short_names() if n not in taken)
    rename = {old: next(names) for old in sorted(counts, key=lambda n: (-counts[n], n))}

    out: list[str] = []
    prev_text = ''
    for kind, text in toks:
        text = rename.get(text, text) if kind == 'id' else text
        if out:
            joined = prev_text + text
            words = (prev_text[-1:].isalnum() or prev_text[-1:] == '_') and \
                    (text[:1].isalnum() or text[:1] == '_')
            if words or len(_wgsl_tokens(joined)) != 2:
                out.append(' ')
        out.append(text)
        prev_text = text
    return ''.join(out)


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
    ap.add_argument('--meta', default='./checkpoints_v2/weights.meta.json')
    ap.add_argument('--out', default='../lex/src/shader.wgsl')
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    src = generate(meta)
    Path(args.out).write_text(src)
    print(f'wrote {args.out} ({len(src)} bytes, '
          f'{len(pipeline_order(meta["config"]["n_layers"]))} dispatches)')


if __name__ == '__main__':
    main()
