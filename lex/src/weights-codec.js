// Decodes the embedded weight blob. Zero dependencies and no network fetch: the
// model is ~28 KB, so inlining it costs less than the round-trip it saves and
// keeps `lex` usable from a single import.
//
// Two strings: the bit-packed weight planes as hex of their little-endian
// bytes, and everything else (fp16 scales/norms/headers followed by 8-bit
// scalar codes) as base64. Measured on the v2 model after Brotli, that pair was
// 26.82 KiB against 27.11 for base64+base64, 27.15 for a hex-per-code
// encoding, 27.77 for base85+base85, and 26.63 for the raw binary: 1-bit
// planes are near-random, so the encoding that keeps whole bytes on character
// boundaries compresses best.

const HEX = new Uint8Array(128);
for (let i = 0; i < 16; i++) {
  HEX['0123456789abcdef'.charCodeAt(i)] = i;
  HEX['0123456789ABCDEF'.charCodeAt(i)] = i;
}

/** hex of little-endian bytes -> u32 words (the packed bit-planes). */
function decodeWords(str, words) {
  const out = new Uint32Array(words);
  for (let i = 0; i < words; i++) {
    let w = 0;
    for (let b = 3; b >= 0; b--) {
      const k = (i * 4 + b) * 2;
      w = (w << 8) | (HEX[str.charCodeAt(k)] << 4) | HEX[str.charCodeAt(k + 1)];
    }
    out[i] = w >>> 0;
  }
  return out;
}

/** base64 -> bytes. */
export function decodeBase64(str) {
  const bin = atob(str);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * IEEE half -> float, for the fp16 stream (the 8-bit scalars' step/offset
 * headers). Runs once at load.
 */
export function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  const buf = new ArrayBuffer(4);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let man = h & 0x3ff;
    if (exp === 0) {
      if (man === 0) {
        u32[0] = sign;
      } else {
        // Subnormal: renormalize into a float32 exponent.
        exp = 1;
        while ((man & 0x400) === 0) {
          man <<= 1;
          exp--;
        }
        man &= 0x3ff;
        u32[0] = sign | ((exp + 127 - 15) << 23) | (man << 13);
      }
    } else if (exp === 0x1f) {
      u32[0] = sign | 0x7f800000 | (man << 13);
    } else {
      u32[0] = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    out[i] = f32[0];
  }
  return out;
}

/**
 * Reassemble the two GPU buffers the shader binds: the plane words, and the
 * float table (fp16 values and headers expanded, 8-bit scalars decoded).
 */
export function unpackWeights(planesHex, scalarsB64, meta) {
  const planes = decodeWords(planesHex, meta.plane_words);
  const f16Bytes = decodeBase64(scalarsB64);
  const halves = new Uint16Array(
    f16Bytes.buffer.slice(f16Bytes.byteOffset, f16Bytes.byteOffset + meta.f16_bytes));
  const floats = halfToFloat(halves);
  const codes = f16Bytes.subarray(meta.f16_bytes, meta.f16_bytes + meta.u8_count);
  return { planes, fp: decodeScalars(meta.scalar_segments, floats, codes, meta.f16_count) };
}

/**
 * 8-bit scalar streams -> the flat float table the shader indexes; mirrors
 * train/export.py's decode_scalars. Segments are in table order: `f16` reads
 * raw halves, `i8` reads one half step then signed codes, `log8` reads a half
 * (lo, step) pair then unsigned codes as exp(lo + code * step).
 */
export function decodeScalars(segments, halves, codes, total) {
  const out = new Float32Array(total);
  const signed = new Int8Array(codes.buffer, codes.byteOffset, codes.length);
  let o = 0;
  let fi = 0;
  let ui = 0;
  for (const seg of segments) {
    const n = seg.count;
    if (seg.enc === 'f16') {
      out.set(halves.subarray(fi, fi + n), o);
      fi += n;
    } else if (seg.enc === 'i8') {
      const step = halves[fi++];
      for (let k = 0; k < n; k++) out[o + k] = signed[ui + k] * step;
      ui += n;
    } else {
      const lo = halves[fi++];
      const step = halves[fi++];
      for (let k = 0; k < n; k++) out[o + k] = Math.fround(Math.exp(lo + codes[ui + k] * step));
      ui += n;
    }
    o += n;
  }
  return out;
}
