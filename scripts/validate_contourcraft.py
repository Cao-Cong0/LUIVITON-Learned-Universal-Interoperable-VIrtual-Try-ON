from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luiviton.contourcraft import validate_contourcraft


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ContourCraft checkout for Luiviton.")
    parser.add_argument("--contourcraft-root", required=True, type=Path)
    parser.add_argument("--contourcraft-data", "--hood-data", dest="contourcraft_data", type=Path)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Require the complete upstream training, examples, model, SMPL, and SMPL-X layout.",
    )
    args = parser.parse_args()

    project_root = args.contourcraft_root.expanduser().resolve()
    data_root = args.contourcraft_data or REPO_ROOT / "third_party" / "contourcraft_data"
    result = validate_contourcraft(project_root, data_root)
    report = asdict(result)
    if not args.full_data:
        report.pop("missing_upstream_data")
    print(json.dumps(report, indent=2, default=str))
    if not result.ok or (args.full_data and not result.full_data_ok):
        print("ContourCraft validation failed.", file=sys.stderr)
        return 1
    print("ContourCraft validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
