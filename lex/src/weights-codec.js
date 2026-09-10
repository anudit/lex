// Decodes the embedded weight blob. Zero dependencies and no network fetch: the
// model is ~35 KB, so inlining it as base85 costs less than the round-trip it
// saves and keeps `lex` usable from a single import.

// 85 symbols packing 4 bytes into 5 characters (25% overhead) instead of
// base64's 3-into-4 (33%) -- the same alphabet and padding rule as Python's
// stdlib base64.b85encode/b85decode, so bundle_lex.py can just call that
// directly rather than shipping a second implementation to stay in sync with.
const B85 =
  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~';
const B85_LOOKUP = new Uint8Array(128);
for (let i = 0; i < B85.length; i++) B85_LOOKUP[B85.charCodeAt(i)] = i;

/**
 * base85 -> bytes. A short final group is padded with '~' (the alphabet's
 * highest-value character) up to 5 characters before decoding, then the
 * corresponding number of trailing bytes is dropped -- the exact inverse of
 * how b85encode pads with zero bytes and truncates output characters.
 */
export function decodeBase85(str) {
  const padding = (5 - (str.length % 5)) % 5;
  const n = str.length + padding;
  const out = new Uint8Array((n / 5) * 4);
  let o = 0;
  for (let i = 0; i < n; i += 5) {
    let acc = 0;
    for (let j = 0; j < 5; j++) {
      const k = i + j;
      acc = acc * 85 + B85_LOOKUP[k < str.length ? str.charCodeAt(k) : 126]; // 126 = '~'
    }
    out[o++] = (acc >>> 24) & 255;
    out[o++] = (acc >>> 16) & 255;
    out[o++] = (acc >>> 8) & 255;
    out[o++] = acc & 255;
  }
  return out.subarray(0, out.length - padding);
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
