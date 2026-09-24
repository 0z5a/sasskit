# Reforge workspace validation (SM120)

Baseline: `kacper-daftcode/sasskit@913c13451285e07e189b8f89440bd848251a50e1`.
Python 3.12.13; tests ran in a dedicated environment under `/home/gongji/0z5a`.
Cubit was built from `1f7a5aa6cb0096223f4054930b58c9c0208251e5`;
pyelftools 0.32. No system environment packages were changed.

| Check | Baseline | Patched | Speedup |
|---|---|---|---|
| Barrier-synchronized 2-process shared-path regression | Cross-run token overwritten | Pass | N/A: correctness |
| Barrier-synchronized 8-process shared-path regression | Cross-run token overwritten | Pass | N/A: correctness |
| Workspace/CLI/publication suite | New output API absent | 11 passed | N/A |
| Existing suite | 7 skipped | Same 7 skipped | N/A |
| GPU workers sharing one workspace parent | Not run on unsafe baseline | 1 / 2 / 4 / 8 passed | No kernel speedup claimed |

The CPU baseline reproducer relocates the **two shared filenames only** into a
task-owned directory; it preserves the old shared-path behavior and forces all
writers to reach a barrier before reading. The fixed test uses the actual
`reforge` function, not a replacement lifecycle implementation.

Tests cover repeated calls, spaces, explicit output collisions, atomic
no-clobber publication, input symlink/hardlink aliases, benchmark exceptions,
KeyboardInterrupt, publication failure, durable returned paths and legacy CLI
arguments. Existing fixture-dependent tests remain skipped because their
original four-kernel test binary is unavailable; they are not counted as passes.

GPU checks used an explicit `(uint32_t* input, uint32_t* output, uint32_t n)` PTX
fixture, 65,536 elements, 256 blocks × 256 threads, no dynamic shared memory,
and full modulo-2^32 output comparison against `input * 3 + 7`. Each process
verified its actual device UUID against its assigned UUID. Published outputs
were reopened after cleanup and matched the input hash. These runs used
`max_iters=0` and the integrated A1/A2/A3 tree with a test-only numerical runner.
They establish file ownership and numerical survival, not search quality.

RTX 5090 (SM120), driver 580.82.07, CUDA 13.0.88. The existing fixed-ABI
`sass_test` harness was not treated as a generic numerical oracle. Validation
subprocesses exited naturally; their timeout/termination paths were disabled in
the test adapter. Production timeout behavior was not changed by this PR.
