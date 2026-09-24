# Swap legality validation (SM120)

Baseline: `913c13451285e07e189b8f89440bd848251a50e1`.
The supported set is deliberately small: two verified `MOV RZ, imm32` encodings
with identical recognized scheduling controls. RZ discards writes, avoiding
unmodeled dependencies with instructions outside the pair. Live-destination
moves and every memory/predicate/uniform/unknown form remain unsupported.
Load hoisting currently produces no candidates. This is not an accelerated
live-instruction scheduler, and it does not establish stall/NOP mutation safety.

| Check | Baseline | Patched |
|---|---|---|
| May-alias store/load without GPR overlap | Incorrectly allowed | Rejected |
| Unmodeled effects, modifiers and control fields | Can pass GPR-only checks | Rejected |
| Manually constructed unsafe or stale swap | Can bypass proposal gate | Revalidated before writes |
| Next block's first instruction | Included by inclusive end comparison | Excluded by owned CFG slice |
| CPU policy/integration tests | Regression failures | 65 passed |
| Audited GPU swap through both decoders | Not asserted | Bit-exact for zero and fixed random inputs |

The CFG already owns its instruction list. Reusing that list also removes a
whole-kernel scan and allocation from every lookup:

| Instructions in kernel | Original lookup | Owned CFG lookup | Lookup speedup |
|---:|---:|---:|---:|
| 128 | 5.528 µs | 0.096 µs | 57× |
| 4096 | 172.911 µs | 0.098 µs | 1770× |
| 65536 | 2944.253 µs | 0.098 µs | 29997× |

Five ABBA groups per size, 500 baseline calls and 50,000 patched calls per arm;
values are mean CPU wall time per call. This is **only block-lookup speedup**,
not search or GPU speedup. The old lookup also returns one extra boundary
instruction; the new result matches the exclusive CFG interval.

GPU oracle: 65,536 uint32 values, `output = input * 3 + 7` modulo 2^32,
256 × 256 launch, no dynamic shared memory. Two reachable reserved instructions
were replaced by audited discard moves without relocating branches or changing
ELF size. Swapping only those two words preserved all non-target sections and
produced matching cuobjdump/Cubit byte and text order. Zero and multiple nonzero
seeds passed full readback; unsafe negative cases were never submitted to GPU.
The combined lifecycle fixture passed 1/2/4/8 independent GPU workers.

Encoding/control references are pinned to Cubit
[`MOV_R_II`](https://github.com/kacper-daftcode/cubit/blob/1f7a5aa6cb0096223f4054930b58c9c0208251e5/tables/sm120.json)
and [`build_ctrl`](https://github.com/kacper-daftcode/cubit/blob/1f7a5aa6cb0096223f4054930b58c9c0208251e5/cubit/assembler.py).
Cubit trailing scheduling annotations must agree with the raw control word.

Environment: RTX 5090, CUDA 13.0.88, driver 580.82.07, Python 3.12.13.
The seven original fixture-dependent tests were skipped. Multi-iteration GPU
validation uses the integrated workspace/snapshot fixes and a test-only
numerical runner; this independent patch does not silently include those fixes.
