from __future__ import annotations

import argparse
import filecmp
import py_compile
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from luiviton.contourcraft import git_commit


COMPAT_ROOT = REPO_ROOT / "integrations" / "contourcraft"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the Luiviton ContourCraft integration into a compatible checkout."
    )
    parser.add_argument("--contourcraft-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_root = args.contourcraft_root.expanduser().resolve()
    if not target_root.is_dir():
        raise SystemExit(f"ContourCraft directory not found: {target_root}")

    head = git_commit(target_root)
    if head is None:
        print("Warning: target is not a Git checkout; compatibility cannot be verified.")

    files = sorted(
        path
        for path in COMPAT_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for source in files:
        relative = source.relative_to(COMPAT_ROOT)
        if relative == Path("LICENSE"):
            continue
        destination = target_root / relative
        print(f"{'Would copy' if args.dry_run else 'Copying'} {relative}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    if not args.dry_run:
        mismatches = []
        for source in files:
            relative = source.relative_to(COMPAT_ROOT)
            if relative == Path("LICENSE"):
                continue
            destination = target_root / relative
            if not filecmp.cmp(source, destination, shallow=False):
                mismatches.append(str(relative))
            elif destination.suffix == ".py":
                py_compile.compile(str(destination), doraise=True)
        if mismatches:
            raise SystemExit("ContourCraft overlay validation failed: " + ", ".join(mismatches))
        print("Luiviton ContourCraft integration installed.")
        print("Keep HOOD_PROJECT and HOOD_DATA pointed at this checkout and its data root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
