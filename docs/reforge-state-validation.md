# Reforge snapshot validation (SM120)

Incremental baseline: the `fix/reforge-workspace` branch. This change keeps
`instructions`, `blocks` and `current_time_ms` together as current-search state;
`best_time_ms` and the published cubin describe the independently retained best.

| Check | Original search | Patched search | Speedup |
|---|---|---|---|
| Accept A/B; inspect next A/C dependency | Uses old A/B/C analysis | Uses B/A/C analysis; rejects dependent pair | N/A: correctness |
| Accepted candidate slower than best | Analysis remains old | Current refreshed, best preserved | N/A |
| Reject / invalid timing / apply failure / exception | Rollback covers bytes only | Known-good current remains intact | N/A |
| Snapshot unit tests | Regression failures | 17 passed | N/A |
| Combined workspace + snapshot suite | — | 28 passed, 7 existing skips | N/A |
| Both real decoders and GPU readback | Not asserted | cuobjdump + Cubit passed | No kernel speedup claimed |

Candidates are saved before GPU validation and timing. **Only accepted
candidates are reloaded and decoded**, before promotion. Rejected candidates
therefore incur no extra disassembler process; a regression asserts this.
This preserves the annealing formula while avoiding unnecessary analysis work.
Decode exceptions abort before replacing current bytes or publishing a result.

The real fixture replays accept / reject / accept with fresh instruction/CFG
objects. Its acceptance timings are intentionally scripted (10, 9, 11, 8),
not reported as measured GPU performance. Every candidate is separately loaded
and checked against a full integer oracle. Both decoder backends produce the
same swapped binary hash. Non-target sections and the sentinel kernel remain
unchanged. Separate CPU cases cover accepted-non-best state, raw control-word
changes, NaN/Inf/nonpositive timing, and failure before promotion.

The fixture's initial complete decode + CFG construction measured approximately
0.69 s with cuobjdump and 0.71 s with Cubit on this host. These single observations
include external process/table loading; they are not kernel timings or a
statistical decoder comparison. GPU event measurements are stored separately.

Validation: Python 3.12.13, Cubit `1f7a5aa6cb0096223f4054930b58c9c0208251e5`,
CUDA 13.0.88, driver 580.82.07, RTX 5090. Missing original fixture tests remain
skipped. The integration adapter uses a known numerical ABI and does not invoke
subprocess timeout termination. This PR does not change the legacy harness ABI
or claim safety for other mutation families.
