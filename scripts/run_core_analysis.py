"""Run the core ODE + EM analysis and write selected outputs.

Usage
-----
python scripts/run_core_analysis.py
python scripts/run_core_analysis.py --output-dir analysis_outputs_rebuilt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from athlete_recovery.pipeline import run_core_analysis, write_core_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs",
        help="Directory where selected tables and figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_core_analysis(REPO_ROOT)
    write_core_outputs(result, args.output_dir)

    print(f"Wrote core outputs to: {args.output_dir}")
    print(f"Selected EM profile count: {result.em_result.primary_k}")
    print(result.em_result.cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
