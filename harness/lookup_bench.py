import json
import statistics
import time
from sasskit.analysis.cfg import BasicBlock
from sasskit.core.decoder import Instruction
from sasskit.schedule.reforge import _get_block_instructions


def old_lookup(instructions, block):
    return [i for i in instructions if block.start_offset <= i.code_offset <= block.end_offset]


rows = []
for n in (128, 4096, 65536):
    instructions = [Instruction(i * 16, 0, 0, '', '') for i in range(n)]
    lo, hi = n // 4, n // 2
    block = BasicBlock(0, lo * 16, hi * 16, instructions[lo:hi])
    groups = []
    for repeat in range(5):
        times = {}
        for name, fn in [('baseline', old_lookup), ('patched', _get_block_instructions),
                         ('patched', _get_block_instructions), ('baseline', old_lookup)]:
            count = 500 if name == 'baseline' else 50000
            start = time.perf_counter_ns()
            for _ in range(count):
                result = fn(instructions, block)
            elapsed = (time.perf_counter_ns() - start) / count
            times.setdefault(name, []).append(elapsed)
        groups.append({k: statistics.mean(v) for k,v in times.items()})
    rows.append({'instructions': n, 'unit': 'ns/call', 'groups': groups,
                 'baseline_count': len(old_lookup(instructions, block)),
                 'patched_count': len(_get_block_instructions(instructions, block))})
print(json.dumps(rows, indent=2))
