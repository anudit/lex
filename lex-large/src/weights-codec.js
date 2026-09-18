// Decode the embedded packed planes and scalar stream. The scalar stream is
// fp16 headers followed by QAT-trained 8-bit values, matching train/export.py.

const HEX = new Uint8Array(128);
for (let i = 0; i < 16; i++) {
  HEX['0123456789abcdef'.charCodeAt(i)] = i;
  HEX['0123456789ABCDEF'.charCodeAt(i)] = i;
}

const B85 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~';
const B85_LOOKUP = new Uint8Array(128);
for (let i = 0; i < B85.length; i++) B85_LOOKUP[B85.charCodeAt(i)] = i;

function decodeBase85(str) {
  const padding = (5 - (str.length % 5)) % 5;
  const out = new Uint8Array(((str.length + padding) / 5) * 4);
  let offset = 0;
  for (let i = 0; i < str.length + padding; i += 5) {
    let value = 0;
    for (let j = 0; j < 5; j++) {
      const index = i + j;
      value = value * 85 + B85_LOOKUP[index < str.length ? str.charCodeAt(index) : 126];
    }
    out[offset++] = (value >>> 24) & 255;
    out[offset++] = (value >>> 16) & 255;
    out[offset++] = (value >>> 8) & 255;
    out[offset++] = value & 255;
  }
  return out.subarray(0, out.length - padding);
}

function packLegacyPlanes(symbols, meta) {
  const planes = new Uint32Array(meta.plane_words);
  for (const tensor of Object.values(meta.sym_tensors)) {
    const count = tensor.words * 32;
    for (let i = 0; i < count; i++) {
      const code = HEX[symbols.charCodeAt(tensor.offset + i)];
      const word = i >>> 5;
      const bit = i & 31;
      for (let plane = 0; plane < tensor.bits; plane++) {
        if ((code >>> plane) & 1) {
          planes[tensor.plane_offset + plane * tensor.words + word] |= (1 << bit);
        }
      }
    }
  }
  return planes;
}

function decodeWords(str, words) {
  const out = new Uint32Array(words);
  for (let i = 0; i < words; i++) {
    let word = 0;
    for (let byte = 3; byte >= 0; byte--) {
      const offset = (i * 4 + byte) * 2;
      word = (word << 8) | (HEX[str.charCodeAt(offset)] << 4)
        | HEX[str.charCodeAt(offset + 1)];
    }
    out[i] = word >>> 0;
  }
  return out;
}

export function decodeBase64(str) {
  const binary = atob(str);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

export function halfToFloat(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const h = values[i];
    const sign = (h & 0x8000) ? -1 : 1;
    const exponent = (h >>> 10) & 0x1f;
    const fraction = h & 0x3ff;
    if (exponent === 0) out[i] = sign * 2 ** -14 * (fraction / 1024);
    else if (exponent === 31) out[i] = fraction ? NaN : sign * Infinity;
    else out[i] = sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return out;
}

export function decodeScalars(segments, halves, codes, total) {
  const out = new Float32Array(total);
  const signed = new Int8Array(codes.buffer, codes.byteOffset, codes.length);
  let outputOffset = 0;
  let halfOffset = 0;
  let codeOffset = 0;
  for (const segment of segments) {
    const count = segment.count;
    if (segment.enc === 'f16') {
      out.set(halves.subarray(halfOffset, halfOffset + count), outputOffset);
      halfOffset += count;
    } else if (segment.enc === 'i8') {
      const step = halves[halfOffset++];
      for (let i = 0; i < count; i++) out[outputOffset + i] = signed[codeOffset + i] * step;
      codeOffset += count;
    } else if (segment.enc === 'log8') {
      const low = halves[halfOffset++];
      const step = halves[halfOffset++];
      for (let i = 0; i < count; i++) {
        out[outputOffset + i] = Math.fround(Math.exp(low + codes[codeOffset + i] * step));
      }
      codeOffset += count;
    } else {
      throw new Error(`unknown scalar encoding: ${segment.enc}`);
    }
    outputOffset += count;
  }
  return out;
}

export function unpackWeights(planesHex, scalarsBase64, meta) {
  if ((meta.scalar_bits ?? meta.config?.scalar_bits) !== 8) {
    const bytes = decodeBase85(scalarsBase64);
    const halves = new Uint16Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + meta.f16_bytes),
    );
    return { planes: packLegacyPlanes(planesHex, meta), fp: halfToFloat(halves) };
  }
  const planes = decodeWords(planesHex, meta.plane_words);
  const bytes = decodeBase64(scalarsBase64);
  const halves = new Uint16Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + meta.f16_bytes),
  );
  const floats = halfToFloat(halves);
  const codes = bytes.subarray(meta.f16_bytes, meta.f16_bytes + meta.u8_count);
  return {
    planes,
    fp: decodeScalars(meta.scalar_segments, floats, codes, meta.f16_count),
  };
}
