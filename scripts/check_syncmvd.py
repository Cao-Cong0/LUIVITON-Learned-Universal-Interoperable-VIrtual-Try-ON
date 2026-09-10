#!/usr/bin/env python3
import argparse
import py_compile
from pathlib import Path


REQUIRED_SNIPPETS = {
    "run_experiment.py": ("CustomUNet2DConditionModel.from_config",),
    "src/pipeline.py": (
        "mesh_transform.npy",
        "verts.npy",
        "f_maps.npz",
        "extract_layer=extract_layer",
        "render_geometry(image_size=render_rgb_size)",
        "i+base,",
    ),
    "src/custom_unet_2d_condition.py": (
        "CustomUNet2DConditionModel = UNet2DConditionModel",
        "extract_layer",
        "f_map.append(sample.clone())",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Luiviton SyncMVD integration.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    errors = []
    for relative_path, snippets in REQUIRED_SNIPPETS.items():
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        contents = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in contents:
                errors.append(f"{relative_path} is missing {snippet!r}")

    for path in (root / "run_experiment.py", root / "src/pipeline.py", root / "src/custom_unet_2d_condition.py"):
        if path.is_file():
            py_compile.compile(str(path), doraise=True)

    if errors:
        raise SystemExit("Incompatible SyncMVD checkout:\n- " + "\n- ".join(errors))

    print(f"SyncMVD integration is compatible: {root}")


if __name__ == "__main__":
    main()
