#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import py_compile
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = REPO_ROOT / "integrations" / "diffusion_net" / "overlay"
OVERLAY_FILES = (
    Path("src/diffusion_net/geometry.py"),
    Path("src/diffusion_net/layers.py"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Luiviton DiffusionNet integration.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models/cloth_correspondence.pth")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    errors = []
    for relative in OVERLAY_FILES:
        source = OVERLAY_ROOT / relative
        target = root / relative
        if not target.is_file():
            errors.append(f"missing file: {target}")
            continue
        if not filecmp.cmp(source, target, shallow=False):
            errors.append(f"overlay mismatch: {target}")
            continue
        py_compile.compile(str(target), doraise=True)

    if errors:
        raise SystemExit("Incompatible DiffusionNet checkout:\n- " + "\n- ".join(errors))

    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        sys.path.insert(0, str(root / "src"))
        import torch
        from diffusion_net.layers import DiffusionNet

        model = DiffusionNet(
            C_in=3,
            C_out=2,
            C_width=128,
            N_block=7,
            last_activation=torch.tanh,
            outputs_at="vertices",
            dropout=True,
            with_gradient_rotations=True,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    else:
        print(f"Warning: checkpoint not found; skipped model compatibility check: {checkpoint}")

    print(f"DiffusionNet integration is compatible: {root}")


if __name__ == "__main__":
    main()
