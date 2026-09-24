# ResMLP-12 SM120 PTX experiment

Full pretrained `timm/resmlp_12_224.fb_in1k`, FP32, batch 2, 224×224 seeded
images, all 12 blocks and the classifier. A direct PTX implementation specializes
25 affine calls for the observed dense layouts and fuses 24 scale-plus-residual
pairs. Loads/stores use 128-bit vectors where layout permits. Affine preserves
native `addcmul` FMA rounding; residuals preserve separate multiply/add rounding.

Scope: inference on SM120, hidden width 384, 196 patch tokens, CUDA Graph replay,
identity drop-path. Both variants keep the same pretrained weights and inputs.
This is an explicit model experiment; its gains belong to the PTX consumer,
separate from the reforge workspace/snapshot/swap correctness changes.

The stacked residual/affine fusion is described in [CHAIN_RESULTS.md](CHAIN_RESULTS.md).
The current harness compares native, the original PTX path, and the fused path
in separate CUDA Graphs with the same weights and input.

## Original PTX results

Fresh source export at `d9927c2` followed by a clean CUDA/C++ build, then five
independent processes. Each process loads weights once, builds separate graph
objects, and runs ABBA or BAAB with 20 warmups and 100 forwards per arm.
Two arm observations are averaged within each process; the summary is the
geometric mean over five independent paired processes. Every process checks
**bit-exact full logits** against native eager and native graph execution.
Input seeds below were held out from pilot/screening. Five units are a small
sample; the observed per-process speedup range is reported without a p-value.

| Input seed | Native ms/forward | PTX ms/forward | Speedup | Throughput gain |
|---:|---:|---:|---:|---:|
| 2026092501 | 1.091428 | 1.030597 | 1.0590× | 5.90% |
| 2026092502 | 1.040839 | 0.988155 | 1.0533× | 5.33% |
| 2026092503 | 1.047658 | 0.998577 | 1.0492× | 4.92% |
| 2026092504 | 1.048322 | 0.998126 | 1.0503× | 5.03% |
| 2026092505 | 1.047309 | 0.998280 | 1.0491× | 4.91% |
| Geometric mean | 1.054954 | 1.002644 | 1.0522× | 5.22% |

Latency reduction: 4.96%. This measures the complete model's GPU
graph time; input preprocessing and server/request overhead are outside the
measurement. The initial five-process screening independently measured 1.0485×.
Affine specialization alone was approximately neutral in the pilot; the
residual fusion produced the useful signal. No SASS swap gain is claimed.

[Raw paired observations](results/) include GPU event and host wall times,
checkpoint/source/binary hashes, output hashes, versions and actual GPU UUID.
Hardware: RTX 5090 `a015762a-f0d5-9065-109d-37898af80ecf`, driver 580.82.07,
CUDA 13.0.88, torch 2.13.0+cu130, timm 1.0.29. TF32 was disabled. Each process
had its own compiled-graph state; no process was terminated for validation.

Checkpoint revision: `59a4112ca93db165fd828bef7d47d758ad84295b`.
Weight SHA-256: `2ed399f6a787a30925f49bd3e8040e6b0f0a620ed68d56758c703e98bb7c0fbe`.
Task-owned checkpoint copies were removed after validation (61,417,494 bytes per
copy); checkpoints and compiled binaries are excluded from this example.

## Reproduce

Use an environment containing the listed torch/timm versions and safetensors,
with a CUDA toolkit and C++ compiler. Download the pinned model checkpoint into
a directory owned by the experiment, then pass that file explicitly:

```bash
bash examples/resmlp_ptx/build.sh
CUDA_VISIBLE_DEVICES=<GPU-UUID> python examples/resmlp_ptx/model_e2e.py \
  --weights /absolute/path/model.safetensors \
  --seed 2026092501 --output /absolute/path/result.json
```

The script asserts supported layouts and PTX capture counts: 25 affine + 24
residual launches for the original path, or 1 affine + 24 fused launches for the
new path. It also checks eight kernel cases and all 24 model boundaries. Use
`--validate-only` to run correctness checks without timing. Changing model size,
dtype, training behavior or layout requires separate kernels and validation.
