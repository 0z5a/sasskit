"""CLI for the forge (AI-driven hot loop optimization) module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract a hot loop specification from a cubin."""
    from sasskit.forge.hotloop import extract_hot_loop

    spec = extract_hot_loop(
        cubin_path=args.input,
        kernel_name=args.kernel,
        start_offset=args.start,
        end_offset=args.end,
        name=args.name or "hotloop",
        description=args.description or "",
        max_registers=args.max_regs,
    )

    output = args.output or f"{spec.name}_spec.json"
    spec.save(output)
    print(f"Saved hot loop spec: {output}")
    print(f"  Instructions: {spec.max_instructions}")
    print(f"  Inputs: {spec.n_inputs}, Outputs: {spec.n_outputs}, Scratch: {spec.n_scratch}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the AI-driven optimization loop."""
    from sasskit.forge.hotloop import HotLoopSpec
    from sasskit.forge.engine import ForgeEngine

    spec = HotLoopSpec.load(args.spec)

    # Override model if specified
    llm_fn = None
    if args.model:
        from sasskit.forge.engine import call_anthropic
        model = args.model
        def llm_fn(prompt, system=None):
            return call_anthropic(prompt, system=system, model=model)

    engine = ForgeEngine(
        spec=spec,
        llm_fn=llm_fn,
        sass_test_bin=args.sass_test_bin,
        bench_bin=args.bench_bin,
        log_path=args.log,
    )
    engine.run(max_iterations=args.iterations, extra_instructions=args.extra or "")
    return 0


def cmd_manual(args: argparse.Namespace) -> int:
    """Test a manually written SASS variant."""
    from sasskit.forge.hotloop import HotLoopSpec
    from sasskit.forge.engine import ForgeEngine

    spec = HotLoopSpec.load(args.spec)
    engine = ForgeEngine(spec=spec, sass_test_bin=args.sass_test_bin)

    sass_lines = Path(args.sass_file).read_text().strip().split("\n")
    sass_lines = [l.strip() for l in sass_lines if l.strip() and not l.strip().startswith("//")]

    result = engine.run_manual(sass_lines)
    return 0 if result.status == "PASS" else 1


def add_forge_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add forge subcommands to a parent subparser group."""

    p_extract = subparsers.add_parser(
        'forge-extract',
        help='Extract a hot loop spec from a cubin',
    )
    p_extract.add_argument('input', help='Input cubin file')
    p_extract.add_argument('-k', '--kernel', required=True, help='Kernel name')
    p_extract.add_argument('--start', type=lambda x: int(x, 0), required=True,
                           help='Hot loop start offset (hex)')
    p_extract.add_argument('--end', type=lambda x: int(x, 0), required=True,
                           help='Hot loop end offset (hex)')
    p_extract.add_argument('--name', help='Name for the spec')
    p_extract.add_argument('--description', help='Description')
    p_extract.add_argument('--max-regs', type=int, default=64,
                           help='Max register budget (default: 64)')
    p_extract.add_argument('-o', '--output', help='Output spec JSON file')

    p_run = subparsers.add_parser(
        'forge-run',
        help='Run AI-driven hot loop optimization',
    )
    p_run.add_argument('spec', help='Hot loop spec JSON file')
    p_run.add_argument('-n', '--iterations', type=int, default=0,
                       help='Max iterations (0=unlimited, default: 0)')
    p_run.add_argument('--sass-test-bin', help='Path to sass_test binary')
    p_run.add_argument('--bench-bin', help='Path to bench_harness binary (for timing+correctness)')
    p_run.add_argument('--model', help='LLM model name (e.g. claude-opus-4-5)')
    p_run.add_argument('--log', help='Log file path')
    p_run.add_argument('--extra', help='Extra instructions for the LLM prompt')

    p_manual = subparsers.add_parser(
        'forge-manual',
        help='Test a manually written SASS variant',
    )
    p_manual.add_argument('spec', help='Hot loop spec JSON file')
    p_manual.add_argument('sass_file', help='SASS file with instructions')
    p_manual.add_argument('--sass-test-bin', help='Path to sass_test binary')


def dispatch_forge(command: str, args: argparse.Namespace) -> int:
    """Dispatch a forge subcommand."""
    if command == 'forge-extract':
        return cmd_extract(args)
    elif command == 'forge-run':
        return cmd_run(args)
    elif command == 'forge-manual':
        return cmd_manual(args)
    return 1
