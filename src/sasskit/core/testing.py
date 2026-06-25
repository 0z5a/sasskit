"""GPU test runner — subprocess wrapper around the sass_test C harness.

Provides a single entry point for testing cubin files on the GPU,
used by recolor (fuzz mode), forge (variant testing), and cuasm_qa (L2 tests).
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def find_sass_test_binary() -> str:
    """Locate the sass_test binary."""
    # Check env var first
    env_bin = os.environ.get("SASS_TEST_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    # Check relative to package
    pkg_harness = Path(__file__).parent.parent.parent.parent / "harnesses" / "sass_test"
    if pkg_harness.is_file():
        return str(pkg_harness)
    # Fallback
    return "sass_test"


def run_sass_test(
    cubin_path: str,
    kernel_name: str = "KernelA",
    blocks: int = 1,
    threads: int = 256,
    smem: int = 0,
    timeout: int = 30,
    sass_test_bin: Optional[str] = None,
) -> tuple[str, str]:
    """Run sass_test on a cubin file.

    Returns (status, detail) where status is one of:
        PASS, FAIL, CRASH, TIMEOUT
    """
    if sass_test_bin is None:
        sass_test_bin = find_sass_test_binary()

    try:
        result = subprocess.run(
            [sass_test_bin, str(cubin_path), kernel_name,
             str(blocks), str(threads), str(smem)],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0 and "PASS" in stdout:
            m = re.search(r'per_iter=([\d.]+)ms', stdout)
            detail = f"per_iter={m.group(1)}ms" if m else ""
            return ("PASS", detail)
        elif result.returncode < 0:
            import signal
            sig = -result.returncode
            return ("CRASH", f"signal={sig}")
        else:
            detail = ""
            for line in (stderr + "\n" + stdout).split("\n"):
                if line.startswith("FAIL:"):
                    detail = line
                    break
            return ("FAIL", detail or f"rc={result.returncode}")

    except subprocess.TimeoutExpired:
        return ("TIMEOUT", f">{timeout}s")


def test_cubin_bytes(
    cubin_bytes: bytes,
    kernel_name: str = "KernelA",
    timeout: int = 30,
    sass_test_bin: Optional[str] = None,
) -> tuple[str, str]:
    """Test cubin from bytes (writes to temp file, runs, cleans up)."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        tmp_path = f.name
    try:
        return run_sass_test(tmp_path, kernel_name, timeout=timeout,
                             sass_test_bin=sass_test_bin)
    finally:
        os.unlink(tmp_path)
