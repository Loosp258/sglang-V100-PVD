"""Run isolated PVD CPU modules without importing the CUDA inference frontend.

Usage: python test/registered/disaggregation/run_pvd_cpu_tests.py [pytest args]
Only package initializers are bypassed; PVD code, torch and HTTP are real.
"""

import sys
import types
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    for name in ("sglang", "sglang.srt", "sglang.srt.disaggregation"):
        module = types.ModuleType(name)
        module.__path__ = [str(root / "python" / Path(*name.split(".")))]
        sys.modules[name] = module
    import pytest

    raise SystemExit(
        pytest.main(
            sys.argv[1:] or [str(Path(__file__).with_name("test_pvd3.py")), "-q"]
        )
    )
