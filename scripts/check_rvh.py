#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = REPO_ROOT / "integrations" / "rvh" / "overlay"


def overlay_files() -> list[Path]:
    return sorted(
        path.relative_to(OVERLAY_ROOT)
        for path in OVERLAY_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Luiviton RVH registration integration.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    errors = []
    for relative in overlay_files():
        source = OVERLAY_ROOT / relative
        target = root / relative
        if not target.is_file():
            errors.append(f"missing file: {target}")
            continue
        if not filecmp.cmp(source, target, shallow=False):
            errors.append(f"overlay mismatch: {target}")
            continue
        if target.suffix == ".py":
            py_compile.compile(str(target), doraise=True)

    if errors:
        raise SystemExit("Incompatible RVH checkout:\n- " + "\n- ".join(errors))

    env = os.environ.copy()
    python_paths = [str(root), str(REPO_ROOT / "third_party" / "psbody_mesh")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    with tempfile.TemporaryDirectory(prefix="luiviton-matplotlib-") as matplotlib_dir:
        env["MPLCONFIGDIR"] = matplotlib_dir
        for entrypoint in ("fit_SMPLHD_body.py", "fit_SMPLH_cloth.py"):
            subprocess.run(
                [sys.executable, str(root / "smpl_registration" / entrypoint), "--help"],
                cwd=root,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )

    print(f"RVH registration integration is compatible: {root}")


if __name__ == "__main__":
    main()
