# Reforge SM120 validation fixtures

This branch integrates the three independent correctness drafts. The driver
oracle has the explicit ABI `(uint32_t* input, uint32_t* output, uint32_t n)`;
it checks every output and prints the actual device UUID. Only the audited
RZ-discard swap is executed. Negative cases stay in CPU tests.

The scripts use a fresh run directory with `tmp/`, `evidence/`, `fixture.cubin`,
`harness/validate`, and the pinned Cubit source at `deps/cubit`. The Cubit CLI is
built at `target-cli/release/cubit`; its Python binding and pyelftools are in the
selected Python environment. Keep the CLI off PATH: the test explicitly selects
it for the Cubit half and otherwise uses cuobjdump.

```bash
mkdir -p tmp evidence
/usr/local/cuda/bin/ptxas -arch=sm_120 harness/fixture.ptx -o fixture.cubin
g++ -O3 -std=c++17 -I/usr/local/cuda/include harness/validate.cpp -lcuda -o harness/validate
export PYTHONPATH="$PWD/src"
export CUBIT_TABLE="$PWD/deps/cubit/tables/sm120.json"
export TMPDIR="$PWD/tmp"
CUDA_VISIBLE_DEVICES=0 python harness/gpu_integration.py prepare
for count in 1 2 4 8; do python harness/gpu_integration.py "$count"; done
python harness/lookup_bench.py > evidence/lookup-bench.json
```

The snapshot replay uses scripted acceptance timings while recording real GPU
event measurements separately. It verifies accept/reject/accept using fresh
analysis, non-target byte preservation, both disassemblers and full numerical
readback. Existing outputs are not overwritten by reforge; use a fresh run
rather than sharing an earlier run's output names. Test subprocess timeout
termination is disabled in this task adapter; the fixture has finite work and
all child processes are joined after natural exit.

`evidence/reforge/` contains the recorded unit-test summaries, decode/oracle
records and block-lookup observations. Full-model PTX performance belongs to
[the separate experiment draft](https://github.com/0z5a/sasskit/pull/2).
