// WebGPU inference runtime. Zero dependencies.
//
// Two measurements shape this:
//
//   * ~96% of a highlight() call is the mapAsync readback round-trip, with a
//     flat floor independent of input size. So every call the page makes in one
//     microtask turn is coalesced into a single command buffer with a single
//     readback -- highlighting ten blocks pays the floor once, not ten times.
//   * The model's arithmetic is 99% token-parallel matmuls. Running it as one
//     workgroup (the first design here) used 64 lanes of a GPU with thousands
//     and was 2x slower than gpu-lexer. The model is now a sequence of entry
//     points, most of them dispatched one workgroup per 16-token tile, with the
//     sequential recurrence left on a single workgroup because it is 0.8% of
//     the work.
//
// All of those dispatches go into one command buffer, so the split costs no
// extra submits and no extra readbacks. Batched jobs ride the second dispatch
// dimension, so a batch is still one submit.

const CLASS_NAMES = [
  'plain', 'comment', 'string', 'number', 'keyword',
  'type', 'function', 'constant', 'operator',
];

const JOB_BYTES = 16;  // Job { token_count, token_offset, stage, stride }
// Must match MAX_JOBS in the shader: a uniform array needs a compile-time size,
// so a larger burst of calls is flushed in chunks rather than one dispatch.
const MAX_JOBS = 64;
// Scratch regions per job: normalized input, scan input, forward and backward
// scan results, gated context, pooled statistics, the FiLM parameters, and the
// pre-layer embedding kept around for the highway connection.
const SCRATCH_REGIONS = 8;

function align(n, a) {
  return Math.ceil(n / a) * a;
}

export class LexRuntime {
  #device;
  #pipelines = [];
  #steps = [];
  #layout;
  #buffers = {};
  #capacity = { tokens: 0, out: 0, jobs: 0 };
  #pending = [];
  #scheduled = false;
  #inflight = Promise.resolve();
  #bind = null;
  #compilation = null;

  static async create({ shader, planes, fp, steps, dim = 64, adapterOptions } = {}) {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('WebGPU is not available in this environment');
    }
    const adapter = await navigator.gpu.requestAdapter(adapterOptions);
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice();
    // WebGPU reports validation failures asynchronously and otherwise silently:
    // a bad pipeline produces a dispatch that writes nothing, which looks like a
    // model that predicts class 0 everywhere rather than like an error.
    device.addEventListener?.('uncapturederror', (e) => {
      console.error('[lex] WebGPU error:', e.error?.message ?? e.error);
    });
    const runtime = new LexRuntime(device, shader, planes, fp, steps, dim);
    await runtime.ready();
    return runtime;
  }

  constructor(device, shader, planes, fp, steps, dim) {
    this.#device = device;
    this.#steps = steps;
    this.dim = dim;
    const module = device.createShaderModule({ code: shader, label: 'lex' });
    this.#compilation = module.getCompilationInfo?.() ?? Promise.resolve({ messages: [] });

    this.#layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.#layout],
    });
    this.#pipelines = steps.map((step) => device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: step.entry },
      label: step.entry,
    }));

    this.#buffers.planes = this.#upload(planes, GPUBufferUsage.STORAGE, 'planes');
    this.#buffers.fp = this.#upload(fp, GPUBufferUsage.STORAGE, 'fp');
  }

  /** Resolves once the shader has compiled; rejects with the WGSL diagnostics. */
  async ready() {
    const info = await this.#compilation;
    const errors = (info.messages ?? []).filter((m) => m.type === 'error');
    if (errors.length) {
      throw new Error('WGSL compilation failed:\n' + errors
        .map((m) => `  ${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }
    for (const m of (info.messages ?? [])) {
      if (m.type === 'warning') console.warn(`[lex] WGSL ${m.lineNum}:${m.linePos} ${m.message}`);
    }
  }

  #upload(data, usage, label) {
    const buf = this.#device.createBuffer({
      size: align(data.byteLength, 4), usage: usage | GPUBufferUsage.COPY_DST, label,
    });
    this.#device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  #ensure(totalTokens, maxTokens, jobs) {
    const dev = this.#device;
    if (totalTokens > this.#capacity.tokens) {
      const n = Math.max(totalTokens, this.#capacity.tokens * 2, 4096);
      this.#buffers.tokens?.destroy();
      this.#buffers.out?.destroy();
      this.#buffers.staging?.destroy();
      this.#buffers.tokens = dev.createBuffer({
        size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'tokens',
      });
      this.#buffers.out = dev.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'out',
      });
      this.#buffers.staging = dev.createBuffer({
        size: n * 4 * 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'staging',
      });
      this.#capacity.tokens = n;
      this.#bind = null;
    }
    // Every job gets its own hidden-state slot so the whole batch can run as one
    // parallel dispatch rather than as N serialized passes.
    const need = jobs * maxTokens * this.dim * 4;
    if (need > (this.#capacity.hid || 0)) {
      const n = Math.max(need, (this.#capacity.hid || 0) * 2, 64 * 1024);
      this.#buffers.hid?.destroy();
      this.#buffers.scratch?.destroy();
      // COPY_SRC so the test page can read intermediate hidden state back.
      this.#buffers.hid = dev.createBuffer({
        size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: 'hid' });
      this.#buffers.scratch = dev.createBuffer({
        size: n * SCRATCH_REGIONS, usage: GPUBufferUsage.STORAGE, label: 'scratch' });
      this.#capacity.hid = n;
      this.#bind = null;
    }
    if (jobs > this.#capacity.jobs) {
      const n = MAX_JOBS;
      this.#buffers.jobs?.destroy();
      this.#buffers.jobs = dev.createBuffer({
        size: MAX_JOBS * JOB_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'jobs',
      });
      this.#capacity.jobs = n;
      this.#bind = null;
    }
    if (!this.#bind) {
      const b = this.#buffers;
      this.#bind = this.#device.createBindGroup({
        layout: this.#layout,
        entries: [
          { binding: 0, resource: { buffer: b.tokens } },
          { binding: 1, resource: { buffer: b.planes } },
          { binding: 2, resource: { buffer: b.fp } },
          { binding: 3, resource: { buffer: b.hid } },
          { binding: 4, resource: { buffer: b.scratch } },
          { binding: 5, resource: { buffer: b.out } },
          { binding: 6, resource: { buffer: b.jobs } },
        ],
      });
    }
  }

  /** Queue one classification. Resolves with a Uint32Array of class ids. */
  classify(packed, count, stage = 0) {
    return new Promise((resolve, reject) => {
      this.#pending.push({ packed, count, stage, resolve, reject });
      if (!this.#scheduled) {
        this.#scheduled = true;
        queueMicrotask(() => this.#flush());
      }
    });
  }

  async #flush() {
    const queued = this.#pending;
    this.#pending = [];
    this.#scheduled = false;
    if (!queued.length) return;
    // Flushes are chained rather than run concurrently. There is one staging
    // buffer, and a second mapAsync on it while the first is still pending
    // throws "buffer already has an outstanding map pending" -- which a caller
    // hits simply by awaiting highlight() in a loop.
    const run = this.#inflight.then(async () => {
      // The uniform job array is fixed-size, so a burst larger than MAX_JOBS is
      // split across dispatches rather than dropped.
      for (let i = 0; i < queued.length; i += MAX_JOBS) {
        await this.#runBatch(queued.slice(i, i + MAX_JOBS));
      }
    });
    // Keep the chain alive even if this batch rejects; each job already has its
    // own rejection delivered by #runBatch.
    this.#inflight = run.catch(() => {});
    return run;
  }

  async #runBatch(jobs) {

    try {
      const dev = this.#device;
      let total = 0;
      let maxTokens = 0;
      for (const j of jobs) {
        j.offset = total;
        total += j.count;
        maxTokens = Math.max(maxTokens, j.count);
      }
      this.#ensure(total, maxTokens, jobs.length);

      const stride = maxTokens * this.dim;
      const tokenData = new Uint32Array(total * 2);
      const jobData = new Uint32Array(MAX_JOBS * (JOB_BYTES / 4));
      for (let i = 0; i < jobs.length; i++) {
        const j = jobs[i];
        tokenData.set(j.packed.subarray(0, j.count * 2), j.offset * 2);
        const u = i * (JOB_BYTES / 4);
        jobData[u] = j.count;
        jobData[u + 1] = j.offset;
        jobData[u + 2] = j.stage || 0;
        jobData[u + 3] = stride;
      }
      dev.queue.writeBuffer(this.#buffers.tokens, 0, tokenData);
      dev.queue.writeBuffer(this.#buffers.jobs, 0, jobData);

      // A non-zero stage truncates the pipeline and reads back hidden state
      // instead of classes, which is how the tests localize a mismatch.
      const debugStage = jobs[0].stage || 0;
      const readBytes = debugStage
        ? jobs.length * maxTokens * this.dim * 4
        : total * 4;

      const enc = dev.createCommandEncoder();
      // Clear the output slice first. Without this a job whose result is never
      // written returns whatever the previous call left there, which reads as a
      // plausible-but-wrong highlight rather than an obvious failure.
      enc.clearBuffer(this.#buffers.out, 0, total * 4);
      // One compute pass per stage. Consecutive dispatches *within* a pass are
      // not guaranteed to see each other's storage writes, and this pipeline is
      // a strict dependency chain -- sharing a pass produced deterministically
      // wrong results for every job but the last. Separate passes still share
      // one command buffer and one submit, so the readback cost is unchanged.
      const nSteps = debugStage ? Math.min(debugStage, this.#steps.length)
                                : this.#steps.length;
      for (let i = 0; i < nSteps; i++) {
        const step = this.#steps[i];
        const x = step.tile ? Math.ceil(maxTokens / step.tile) : 1;
        const pass = enc.beginComputePass();
        pass.setPipeline(this.#pipelines[i]);
        pass.setBindGroup(0, this.#bind);
        pass.dispatchWorkgroups(x, jobs.length);
        pass.end();
      }
      enc.copyBufferToBuffer(
        debugStage ? this.#buffers.hid : this.#buffers.out, 0,
        this.#buffers.staging, 0, readBytes);
      dev.queue.submit([enc.finish()]);

      // The one readback the whole batch shares.
      await this.#buffers.staging.mapAsync(GPUMapMode.READ, 0, readBytes);
      const raw = this.#buffers.staging.getMappedRange(0, readBytes).slice(0);
      this.#buffers.staging.unmap();
      if (debugStage) {
        const all32 = new Float32Array(raw);
        const stride32 = maxTokens * this.dim;
        jobs.forEach((j, i) => j.resolve(all32.subarray(i * stride32,
                                                        i * stride32 + j.count * this.dim)));
        return;
      }
      const all = new Uint32Array(raw);
      for (const j of jobs) j.resolve(all.subarray(j.offset, j.offset + j.count));
    } catch (err) {
      for (const j of jobs) j.reject(err);
    }
  }

  stageNames() {
    return this.#steps.map((s) => s.entry);
  }

  destroy() {
    for (const b of Object.values(this.#buffers)) b?.destroy?.();
    this.#device.destroy?.();
  }
}

export { CLASS_NAMES };
