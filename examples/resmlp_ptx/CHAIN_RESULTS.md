# Residual/affine fusion on RTX 5090

This increment depends on the original PTX experiment at `5d802a0`.
For each residual, a single direct-PTX launch computes both `x + gamma * branch`
and the following Affine output. The residual remains available for the next
skip connection; Affine consumes the value in registers. The final Affine uses
the same path. PTX launches per full forward drop from 49 to 25.

`mul.rn.f32` and `add.rn.f32` preserve the two native residual rounding steps.
Only Affine uses `fma.rn.f32`, matching `torch.addcmul`. Both output tensors retain
token-contiguous layout, and contiguous spans use 128-bit loads/stores.

Fixed unsigned divisions use `mul.hi.u32` and shifts. For divisor 196, the
multiplier is `0x5397829d` with a total shift of 38; for 384, it is `0xaaaaaaab`
with a total shift of 40. Their reciprocal numerator errors are 52 and 128.
For every unsigned 32-bit input, these add less than `1/divisor` to the exact
quotient, so flooring preserves the integer result. Token and batch remainders
then use multiplication/subtraction. The final SASS has no reciprocal sequence
in either new kernel.

Scope is unchanged: the complete pretrained `timm/resmlp_12_224.fb_in1k`,
all 12 blocks plus stem/classifier, FP32, batch 2, 224×224 seeded inputs,
196 tokens, hidden width 384, identity drop-path, inference, SM120, TF32 off.
The metric is full-model CUDA Graph GPU time; preprocessing and serving overhead
are outside the measurement. Odd-batch kernel checks cover indexing tails,
not a performance claim for additional model configurations.

## Independent validation

Source was exported from `5b323f3c011796f7f1118505ad0a68df9f85a491` into a fresh
directory and rebuilt. All three arms share read-only checkpoint weights and
inputs, with distinct CUDA Graph objects and private pools. Each fresh process
runs a rotated three-arm order followed by its reverse, 20 warmups and 100
forwards per observation. Each arm's two observations are averaged per process;
reported aggregates are geometric means of five paired processes. The seeds
were held out from screening. Five processes are a small sample, so no p-value
or broad-model claim is made.

Every process checks eight standalone cases (both layouts, batches 1/3, zero
and scaled random residual inputs), all 24 fused model boundaries, and full
logits from every graph against native eager. Equality compares the FP32 bit
patterns through int32 views. The pinned checkpoint hash is enforced.

| Seed | Native ms | Previous PTX ms | Fused PTX ms | Speedup vs previous PTX | Speedup vs native |
|---:|---:|---:|---:|---:|---:|
| 2026092901 | 1.048250 | 0.999477 | 0.989617 | 1.0100× | 1.0592× |
| 2026092902 | 1.060416 | 1.006400 | 0.999456 | 1.0069× | 1.0610× |
| 2026092903 | 1.041750 | 0.995440 | 0.984160 | 1.0115× | 1.0585× |
| 2026092904 | 1.036517 | 0.992290 | 0.969681 | 1.0233× | 1.0689× |
| 2026092905 | 1.043029 | 0.996867 | 0.979350 | 1.0179× | 1.0650× |
| Geometric mean | 1.045961 | 0.998084 | 0.984403 | 1.0139× | 1.0625× |

The incremental throughput gain is **1.39%** (latency reduction 1.37%); the
combined gain over native is **6.25%** (latency reduction 5.89%). All five
processes improved over the previous PTX arm, with a range of 0.69%–2.33%.
These ratios use the three arms measured together here; historical results in
README.md used different sessions and must not be added to this increment.
Screening of the retained kernel measured 1.95% over the previous PTX path;
only the independent results above support the PR claim.

Compute Sanitizer memcheck completed with **0 errors**, using the same clean
build and `--validate-only`; its instrumented execution is excluded from timing.
[Raw results, input/output hashes, source/binary hashes, and telemetry](results/)
include the five `chain-clean-*.json` files, before/after CSV snapshots, and
`chain-memcheck.log`. The GPU was RTX 5090 UUID
`a015762a-f0d5-9065-109d-37898af80ecf`, driver 580.82.07, CUDA 13.0.88,
torch 2.13.0+cu130, timm 1.0.29. Clocks were not changed or locked.

Both new kernels compile to four FMUL, four FADD and four FFMA instructions,
with two STG.E.128 stores. The token kernel has two LDG.E.128 loads; the channel
kernel has one plus strided scalar branch loads. Neither contains MUFU.RCP.
The binary hashes are recorded with each run; use `cuobjdump -sass` to inspect.

The checkpoint revision and digest remain those in README.md. Both task-owned
61,417,494-byte checkpoint copies were removed after validation. The existing
main environment was reused read-only through a child environment; no process
was terminated, and no system/environment update was needed.

## Reproduce

Follow the checkpoint setup in README.md, then run the clean build and all three
arms with a fresh process for each seed:

```bash
bash examples/resmlp_ptx/build.sh
CUDA_VISIBLE_DEVICES=<GPU-UUID> python examples/resmlp_ptx/model_e2e.py \
  --weights /absolute/path/model.safetensors \
  --seed 2026092901 --output /absolute/path/result.json
CUDA_VISIBLE_DEVICES=<GPU-UUID> compute-sanitizer --tool memcheck --error-exitcode 99 \
  python examples/resmlp_ptx/model_e2e.py --validate-only \
  --weights /absolute/path/model.safetensors \
  --seed 2026092900 --output /absolute/path/memcheck.json
```
