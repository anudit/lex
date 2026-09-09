// Decodes the embedded weight blob. Zero dependencies and no network fetch: the
// model is ~30 KB, so inlining it as base64 costs less than the round-trip it
// saves and keeps `lex` usable from a single import.

const B64 =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
const LOOKUP = new Uint8Array(128);
for (let i = 0; i < B64.length; i++) LOOKUP[B64.charCodeAt(i)] = i;

/** base64url -> bytes, without relying on atob (absent in some runtimes). */
export function decodeBase64(str) {
  const n = str.length;
  const out = new Uint8Array((n * 3) >> 2);
  let o = 0;
  for (let i = 0; i < n; i += 4) {
    const a = LOOKUP[str.charCodeAt(i)];
    const b = LOOKUP[str.charCodeAt(i + 1)];
    const c = i + 2 < n ? LOOKUP[str.charCodeAt(i + 2)] : 0;
    const d = i + 3 < n ? LOOKUP[str.charCodeAt(i + 3)] : 0;
    out[o++] = (a << 2) | (b >> 4);
    if (i + 2 < n) out[o++] = ((b & 15) << 4) | (c >> 2);
    if (i + 3 < n) out[o++] = ((c & 3) << 6) | d;
  }
  return out.subarray(0, o);
}

/**
 * IEEE half -> float. The model file stores scales, biases, norm gains and decay
 * logits as fp16; the shader reads f32, so this runs once at load over a couple
 * of thousand values.
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

/** Split the flat blob into the two GPU buffers the shader binds. */
export function unpackWeights(bytes, meta) {
  const planeBytes = meta.plane_bytes;
  const planes = new Uint32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + planeBytes));
  const halves = new Uint16Array(
    bytes.buffer.slice(bytes.byteOffset + planeBytes,
                       bytes.byteOffset + planeBytes + meta.f16_bytes));
  return { planes, fp: halfToFloat(halves) };
}
