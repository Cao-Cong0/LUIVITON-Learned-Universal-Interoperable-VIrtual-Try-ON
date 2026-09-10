from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import os
import pickle
import shlex
import subprocess
import sys
import types
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .contourcraft import format_validation_error, validate_contourcraft
from .motion import DEFAULT_TRANSITION_FRAMES


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PipelineConfig:
    """Path and runtime configuration for the non-UI pipeline."""

    repo_root: Path = REPO_ROOT
    output_root: Path = REPO_ROOT / "outputs" / "garment"
    contourcraft_root: Optional[Path] = None
    contourcraft_data: Optional[Path | str] = None
    correspondence_dir: Optional[Path] = None
    registered_cloth_dir: Optional[Path] = None
    registered_body_dir: Optional[Path] = None
    runtime_root: Optional[Path] = None
    smpl_config_path: Optional[Path] = None
    smpl_model_root: Optional[Path] = None
    smpl_template_body: Optional[Path] = None
    correspondence_checkpoint: Optional[Path] = None
    python_executable: str = sys.executable
    device: str = "cuda:0"

    @property
    def cloth_correspondence_dir(self) -> Path:
        return _resolve(self.correspondence_dir, self.output_root / "correspondence" / "cloth")

    @property
    def cloth_registration_dir(self) -> Path:
        return _resolve(self.registered_cloth_dir, self.output_root / "cloth_registered_smpl")

    @property
    def body_registration_dir(self) -> Path:
        return _resolve(self.registered_body_dir, self.output_root / "body_registered_smpl")

    @property
    def smpl_config(self) -> Path:
        return _resolve(self.smpl_config_path, self.repo_root / "configs" / "smpl.yml")

    @property
    def template_body_obj(self) -> Path:
        return _resolve(self.smpl_template_body, self.repo_root / "models" / "smpl_uv_free.obj")

    @property
    def correspondence_model_checkpoint(self) -> Path:
        return _resolve(
            self.correspondence_checkpoint,
            self.repo_root / "models" / "cloth_correspondence.pth",
        )

    @property
    def correspondence_script(self) -> Path:
        return self.repo_root / "correspondence" / "clothing" / "diffusionnet_model.py"

    @property
    def registration_script(self) -> Path:
        return self.repo_root / "smpl_registration" / "smpl_registration" / "fit_SMPLH_cloth.py"

    @property
    def contourcraft_project(self) -> Path:
        if self.contourcraft_root is not None:
            return Path(self.contourcraft_root).expanduser().resolve()
        env_project = os.environ.get("HOOD_PROJECT")
        if env_project:
            return Path(env_project).expanduser().resolve()
        return (self.repo_root / "third_party" / "ContourCraft").resolve()

    @property
    def contourcraft_data_root(self) -> Path:
        data_root = Path(
            self.contourcraft_data
            or os.environ.get("HOOD_DATA")
            or self.repo_root / "third_party" / "contourcraft_data"
        ).expanduser()
        if data_root.is_absolute():
            return data_root.resolve()
        return (self.repo_root / data_root).resolve()

    @property
    def fromanypose_runtime_root(self) -> Path:
        return _resolve(self.runtime_root, self.output_root / "fromanypose")

    def cloth_template_path(self, cloth_name: str) -> Path:
        return self.fromanypose_runtime_root / "cloth_templates" / f"{cloth_name}.pkl"

    def motion_sequence_path(self, name: str) -> Path:
        return self.fromanypose_runtime_root / "motion_sequence" / f"{name}.pkl"

    def transferred_cloth_path(self, body_name: str, cloth_name: str) -> Path:
        return self.fromanypose_runtime_root / "transferred_cloth" / body_name / f"{cloth_name}.obj"


@dataclass(frozen=True)
class CorrespondenceResult:
    cloth_obj: Path
    correspondence_npz: Path
    reused_existing: bool = False


@dataclass(frozen=True)
class RegistrationResult:
    cloth_obj: Path
    correspondence_npz: Path
    registered_obj: Path
    registered_pkl: Path
    reused_existing: bool = False


@dataclass(frozen=True)
class TransferResult:
    cloth_obj: Path
    target_body_obj: Path
    source_body_pkl: Path
    target_body_pkl: Path
    body_transform_path: Path
    cloth_template_pkl: Path
    motion_sequence_pkl: Path
    transferred_obj: Path
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class FullPipelineResult:
    correspondence: CorrespondenceResult
    registration: RegistrationResult
    transfer: TransferResult


def predict_clothing_correspondence(
    cloth_obj: str | Path,
    *,
    config: Optional[PipelineConfig] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> CorrespondenceResult:
    """Predict clothing-to-SMPL vertex correspondence for one clothing OBJ."""

    config = config or PipelineConfig()
    cloth_obj = _require_file(cloth_obj, "clothing OBJ")
    output_dir = config.cloth_correspondence_dir
    output_npz = output_dir / f"top_1_indices_{cloth_obj.stem}.npz"

    if output_npz.exists() and not overwrite:
        return CorrespondenceResult(cloth_obj, output_npz, reused_existing=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.python_executable,
        config.correspondence_script,
        "--evaluate",
        "--mesh_type=cloth",
        f"--test_file={cloth_obj}",
        f"--output_dir={output_dir}",
        f"--body-template={config.template_body_obj}",
        f"--device={config.device}",
    ]
    checkpoint = config.correspondence_model_checkpoint
    if not dry_run:
        _require_file(checkpoint, "clothing correspondence checkpoint")
    cmd.append(f"--checkpoint={checkpoint}")

    _run(cmd, cwd=config.repo_root, dry_run=dry_run)

    if dry_run:
        return CorrespondenceResult(cloth_obj, output_npz)

    _require_file(output_npz, "correspondence output")
    return CorrespondenceResult(cloth_obj, output_npz)


def register_clothing_smpl(
    cloth_obj: str | Path,
    *,
    correspondence_npz: Optional[str | Path] = None,
    config: Optional[PipelineConfig] = None,
    gender: str = "female",
    overwrite: bool = False,
    dry_run: bool = False,
) -> RegistrationResult:
    """Fit a SMPL body to the clothing mesh using predicted correspondences."""

    config = config or PipelineConfig()
    cloth_obj = _require_file(cloth_obj, "clothing OBJ")
    if not dry_run:
        _require_file(config.smpl_config, "SMPL registration config")
        _require_file(config.template_body_obj, "SMPL template body OBJ")
    correspondence_npz = _resolve(
        correspondence_npz,
        config.cloth_correspondence_dir / f"top_1_indices_{cloth_obj.stem}.npz",
    )
    correspondence_npz = _require_file(correspondence_npz, "correspondence NPZ")

    output_dir = config.cloth_registration_dir
    registered_obj = output_dir / f"{cloth_obj.stem}_smpl.obj"
    registered_pkl = output_dir / f"{cloth_obj.stem}_smpl.pkl"

    if registered_obj.exists() and registered_pkl.exists() and not overwrite:
        return RegistrationResult(
            cloth_obj,
            correspondence_npz,
            registered_obj,
            registered_pkl,
            reused_existing=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.python_executable,
        config.registration_script,
        cloth_obj,
        correspondence_npz,
        output_dir,
        "-gender",
        gender,
        "--config-path",
        config.smpl_config,
        "--template-body-path",
        config.template_body_obj,
        "--device",
        config.device,
    ]
    if config.smpl_model_root is not None:
        cmd.extend(["--smpl-model-root", config.smpl_model_root])

    _run(cmd, cwd=config.repo_root, dry_run=dry_run)

    if dry_run:
        return RegistrationResult(cloth_obj, correspondence_npz, registered_obj, registered_pkl)

    _require_file(registered_obj, "registered SMPL OBJ")
    _require_file(registered_pkl, "registered SMPL PKL")
    return RegistrationResult(
        cloth_obj,
        correspondence_npz,
        registered_obj,
        registered_pkl,
    )


def transfer_clothing(
    cloth_obj: str | Path,
    target_body_obj: str | Path,
    *,
    target_body_pkl: str | Path,
    source_body_pkl: Optional[str | Path] = None,
    body_transform_path: str | Path,
    output_obj: Optional[str | Path] = None,
    config: Optional[PipelineConfig] = None,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    auto_scale: bool = False,
    num_frames: int = DEFAULT_TRANSITION_FRAMES,
    collision_frames: int = 10,
    material: Optional[dict[str, float]] = None,
) -> TransferResult:
    """Transfer the clothing mesh to a registered target body."""

    config = config or PipelineConfig()
    cloth_obj = _require_file(cloth_obj, "clothing OBJ")
    target_body_obj = _require_file(target_body_obj, "target body OBJ")
    target_body_pkl = _require_file(target_body_pkl, "target body SMPL PKL")
    if num_frames < 1 or collision_frames < 1:
        raise ValueError("num_frames and collision_frames must be positive")
    if any(float(value) <= 0 for value in scale):
        raise ValueError("scale values must be positive")

    cloth_name = cloth_obj.stem
    body_name = target_body_obj.stem
    source_body_pkl = _resolve(source_body_pkl, config.cloth_registration_dir / f"{cloth_name}_smpl.pkl")
    source_body_pkl = _require_file(source_body_pkl, "source clothing SMPL PKL")

    body_transform = _require_file(body_transform_path, "body transform")
    cloth_template_pkl = config.cloth_template_path(cloth_name)
    motion_sequence_pkl = config.motion_sequence_path(f"{cloth_name}_to_{body_name}")
    output_obj = _resolve(output_obj, config.transferred_cloth_path(body_name, cloth_name))

    contourcraft = _load_contourcraft(config)
    runner = _load_contourcraft_runner(contourcraft, material or _default_material(), config.device)

    cloth_template_pkl.parent.mkdir(parents=True, exist_ok=True)
    motion_sequence_pkl.parent.mkdir(parents=True, exist_ok=True)
    output_obj.parent.mkdir(parents=True, exist_ok=True)

    template_dict = contourcraft["obj2template"](str(cloth_obj), verbose=True)
    contourcraft["pickle_dump"](template_dict, cloth_template_pkl)

    final_scale = contourcraft["generate_pose_sequence"](
        source_pkl=str(source_body_pkl),
        target_pkl=str(target_body_pkl),
        transformation_path=str(body_transform) if body_transform else None,
        output_path=str(motion_sequence_pkl),
        num_frames=num_frames,
    )
    rollout_scale = _resolve_rollout_scale(scale, final_scale, auto_scale)

    trajectories = _rollout_transfer(
        contourcraft,
        runner,
        pose_sequence_path=motion_sequence_pkl,
        garment_template_path=cloth_template_pkl,
        scale=rollout_scale,
        num_frames=num_frames,
        process_motion=True,
    )

    _update_template_rest_pose(contourcraft, cloth_template_pkl, trajectories)

    self_motion_sequence = config.motion_sequence_path(f"{body_name}_to_{body_name}")
    contourcraft["generate_complex_mesh_sequence_single_obj"](
        str(target_body_obj),
        str(self_motion_sequence),
        collision_frames,
    )
    trajectories = _rollout_transfer(
        contourcraft,
        runner,
        pose_sequence_path=self_motion_sequence,
        garment_template_path=cloth_template_pkl,
        scale=(1.0, 1.0, 1.0),
        num_frames=collision_frames,
        process_motion=False,
    )

    _write_transferred_obj(output_obj, trajectories["pred"][-1], trajectories["cloth_faces"], cloth_obj)

    return TransferResult(
        cloth_obj=cloth_obj,
        target_body_obj=target_body_obj,
        source_body_pkl=source_body_pkl,
        target_body_pkl=target_body_pkl,
        body_transform_path=body_transform,
        cloth_template_pkl=cloth_template_pkl,
        motion_sequence_pkl=motion_sequence_pkl,
        transferred_obj=output_obj,
        scale=rollout_scale,
    )


def run_full_pipeline(
    cloth_obj: str | Path,
    target_body_obj: str | Path,
    *,
    target_body_pkl: str | Path,
    config: Optional[PipelineConfig] = None,
    gender: str = "female",
    body_transform_path: str | Path,
    output_obj: Optional[str | Path] = None,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    auto_scale: bool = False,
    overwrite: bool = False,
    num_frames: int = DEFAULT_TRANSITION_FRAMES,
    collision_frames: int = 10,
) -> FullPipelineResult:
    """Run correspondence prediction, clothing SMPL registration, and transfer."""

    config = config or PipelineConfig()
    correspondence = predict_clothing_correspondence(
        cloth_obj,
        config=config,
        overwrite=overwrite,
    )
    registration = register_clothing_smpl(
        cloth_obj,
        correspondence_npz=correspondence.correspondence_npz,
        config=config,
        gender=gender,
        overwrite=overwrite,
    )
    transfer = transfer_clothing(
        cloth_obj,
        target_body_obj,
        target_body_pkl=target_body_pkl,
        source_body_pkl=registration.registered_pkl,
        body_transform_path=body_transform_path,
        output_obj=output_obj,
        config=config,
        scale=scale,
        auto_scale=auto_scale,
        num_frames=num_frames,
        collision_frames=collision_frames,
    )
    return FullPipelineResult(correspondence, registration, transfer)


def _load_contourcraft(config: PipelineConfig) -> dict[str, Any]:
    project = config.contourcraft_project
    data_root = config.contourcraft_data_root
    utils_root = project / "utils"

    validation = validate_contourcraft(project, data_root)
    if not validation.ok:
        raise RuntimeError(format_validation_error(validation))

    os.environ["HOOD_PROJECT"] = str(project)
    os.environ["HOOD_DATA"] = str(data_root)

    _move_to_front(sys.path, str(project))
    _install_contourcraft_utils_package(utils_root)

    return {
        "obj2template": importlib.import_module("utils.mesh_creation").obj2template,
        "pickle_dump": importlib.import_module("utils.io").pickle_dump,
        "load_obj": importlib.import_module("utils.io").load_obj,
        "DEFAULTS": importlib.import_module("utils.defaults").DEFAULTS,
        "load_params": importlib.import_module("utils.arguments").load_params,
        "create_runner": importlib.import_module("utils.arguments").create_runner,
        "generate_pose_sequence": importlib.import_module("utils.Luviton").generate_pose_sequence,
        "process_sample": importlib.import_module("utils.Luviton").process_sample,
        "generate_complex_mesh_sequence_single_obj": importlib.import_module(
            "utils.Luviton"
        ).generate_complex_mesh_sequence_single_obj,
        "make_fromanypose_resize_dataloader": importlib.import_module(
            "utils.datasets"
        ).make_fromanypose_resize_dataloader,
        "apply_material_params": _apply_contourcraft_material_params,
    }


def _apply_contourcraft_material_params(config: Any, material: dict[str, float]) -> Any:
    runner_name = next(iter(config.runner))
    runner_material = config.runner[runner_name].material
    runner_material.density_override = material["density"]
    runner_material.lame_mu_override = material["lame_mu"]
    runner_material.lame_lambda_override = material["lame_lambda"]
    runner_material.bending_coeff_override = material["bending_coeff"]
    return config


def _move_to_front(paths: list[str], path: str) -> None:
    normalized = str(Path(path).resolve())
    paths[:] = [item for item in paths if str(Path(item or ".").resolve()) != normalized]
    paths.insert(0, path)


def _install_contourcraft_utils_package(utils_root: Path) -> None:
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]

    package = types.ModuleType("utils")
    package.__path__ = [str(utils_root)]
    package.__package__ = "utils"
    package.__spec__ = importlib.machinery.ModuleSpec("utils", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["utils"] = package


def _load_contourcraft_runner(contourcraft: dict[str, Any], material: dict[str, float], device: str) -> Any:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"ContourCraft transfer requested device '{device}', but no CUDA GPU is available. "
            "Run on a GPU machine, or use --device cpu after making the ContourCraft runner CPU-safe."
        )

    modules, config = contourcraft["load_params"]("aux/from_any_pose_resize")
    config["device"] = device
    config = contourcraft["apply_material_params"](config, material)
    checkpoint_path = Path(contourcraft["DEFAULTS"].data_root) / "trained_models" / "contourcraft.pth"
    runner_module, runner, aux_modules = contourcraft["create_runner"](modules, config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    runner.load_state_dict(state_dict["training_module"])
    return runner


def _rollout_transfer(
    contourcraft: dict[str, Any],
    runner: Any,
    *,
    pose_sequence_path: Path,
    garment_template_path: Path,
    scale: tuple[float, float, float],
    num_frames: int,
    process_motion: bool,
) -> dict[str, Any]:
    dataloader = contourcraft["make_fromanypose_resize_dataloader"](
        pose_sequence_type="mesh",
        pose_sequence_path=str(pose_sequence_path),
        garment_template_path=str(garment_template_path),
        scale_x=scale[0],
        scale_y=scale[1],
        scale_z=scale[2],
        num_frames=num_frames,
        restpos_scale=True,
    )
    sample = next(iter(dataloader))
    if process_motion:
        sample = contourcraft["process_sample"](sample, str(pose_sequence_path))
    return runner.valid_rollout(sample)


def _update_template_rest_pose(
    contourcraft: dict[str, Any],
    template_path: Path,
    trajectories: dict[str, Any],
) -> None:
    import trimesh

    with open(template_path, "rb") as fp:
        cloth_data = pickle.load(fp)
    cloth_mesh = trimesh.Trimesh(
        vertices=_to_numpy(trajectories["pred"][-1]),
        faces=_to_numpy(trajectories["cloth_faces"]),
        process=True,
    )
    cloth_data["rest_pos"] = cloth_mesh.vertices
    contourcraft["pickle_dump"](cloth_data, template_path)


def _write_transferred_obj(
    output_path: Path,
    vertices: Any,
    faces: Any,
    source_obj_for_uvs: Path,
) -> None:
    vertices = _to_numpy(vertices)
    faces = _to_numpy(faces).astype(np.int64)
    _, _, uvs, faces_uv = _load_obj(source_obj_for_uvs, tex_coords=True)
    use_uvs = len(uvs) > 0 and faces_uv.shape == faces.shape and faces_uv.max(initial=-1) < len(uvs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        for vertex in vertices:
            fp.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        if use_uvs:
            for uv in uvs:
                fp.write(f"vt {uv[0]} {uv[1]}\n")
        for face_index, face in enumerate(faces):
            v_idx = face + 1
            if use_uvs:
                vt_idx = faces_uv[face_index] + 1
                fp.write(
                    f"f {v_idx[0]}/{vt_idx[0]} {v_idx[1]}/{vt_idx[1]} {v_idx[2]}/{vt_idx[2]}\n"
                )
            else:
                fp.write(f"f {v_idx[0]} {v_idx[1]} {v_idx[2]}\n")


def _load_obj(path: Path, tex_coords: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    uvs: list[list[float]] = []
    faces_uv: list[list[int]] = []

    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tex_coords and parts[0] == "vt":
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f":
                face = []
                face_uv = []
                for token in parts[1:4]:
                    indices = token.split("/")
                    face.append(int(indices[0]) - 1)
                    if tex_coords and len(indices) > 1 and indices[1]:
                        face_uv.append(int(indices[1]) - 1)
                faces.append(face)
                if tex_coords and len(face_uv) == 3:
                    faces_uv.append(face_uv)

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(faces_uv, dtype=np.int64),
    )


def _resolve_rollout_scale(
    scale: tuple[float, float, float],
    final_scale: Sequence[float],
    auto_scale: bool,
) -> tuple[float, float, float]:
    if not auto_scale:
        return tuple(float(v) for v in scale)
    body_scale = tuple(float(value) for value in final_scale)
    if len(body_scale) != 3:
        raise ValueError(f"Expected three body scale values, got {len(body_scale)}")
    return body_scale


def _default_material() -> dict[str, float]:
    return {
        "density": 7e-1,
        "lame_mu": 1.5 * (63636 - 15909) / 4 + 15909,
        "lame_lambda": 1.5 * (93333.73508005822 - 3535.414406069427) / 4 + 3535.414406069427,
        "bending_coeff": 1.5 * (0.0013139737991266374 - 6.370782056371576e-08) / 4
        + 6.370782056371576e-08,
    }


def _run(cmd: Sequence[Any], *, cwd: Path, dry_run: bool = False) -> None:
    cmd = [str(part) for part in cmd]
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _require_file(path: str | Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve(path: str | Path | None, default: str | Path) -> Path:
    return Path(default if path is None else path).expanduser().resolve()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    return value


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        repo_root=Path(args.repo_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        contourcraft_root=Path(args.contourcraft_root).expanduser().resolve()
        if args.contourcraft_root
        else None,
        contourcraft_data=args.contourcraft_data,
        correspondence_dir=Path(args.correspondence_dir).expanduser().resolve()
        if args.correspondence_dir
        else None,
        registered_cloth_dir=Path(args.registered_cloth_dir).expanduser().resolve()
        if args.registered_cloth_dir
        else None,
        registered_body_dir=Path(args.registered_body_dir).expanduser().resolve()
        if args.registered_body_dir
        else None,
        runtime_root=Path(args.runtime_root).expanduser().resolve() if args.runtime_root else None,
        smpl_config_path=Path(args.smpl_config).expanduser().resolve() if args.smpl_config else None,
        smpl_model_root=Path(args.smpl_model_root).expanduser().resolve()
        if args.smpl_model_root
        else None,
        smpl_template_body=Path(args.smpl_template_body).expanduser().resolve()
        if args.smpl_template_body
        else None,
        correspondence_checkpoint=Path(args.correspondence_checkpoint).expanduser().resolve()
        if args.correspondence_checkpoint
        else None,
        python_executable=args.python_executable,
        device=args.device,
    )


def _add_config_arguments(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--repo-root", default=argparse.SUPPRESS if suppress_defaults else str(REPO_ROOT))
    parser.add_argument(
        "--output-root",
        default=argparse.SUPPRESS if suppress_defaults else str(REPO_ROOT / "outputs" / "garment"),
    )
    parser.add_argument("--contourcraft-root", default=default)
    parser.add_argument(
        "--contourcraft-data",
        "--hood-data",
        dest="contourcraft_data",
        default=default,
    )
    parser.add_argument("--correspondence-dir", default=default)
    parser.add_argument("--registered-cloth-dir", default=default)
    parser.add_argument("--registered-body-dir", default=default)
    parser.add_argument("--runtime-root", default=default)
    parser.add_argument("--smpl-config", default=default)
    parser.add_argument("--smpl-model-root", default=default)
    parser.add_argument("--smpl-template-body", default=default)
    parser.add_argument("--correspondence-checkpoint", default=default)
    parser.add_argument("--python-executable", default=argparse.SUPPRESS if suppress_defaults else sys.executable)
    parser.add_argument("--device", default=argparse.SUPPRESS if suppress_defaults else "cuda:0")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Luiviton clothing pipeline without a UI.",
        allow_abbrev=False,
    )
    _add_config_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    corr = subparsers.add_parser(
        "correspondence",
        help="predict clothing correspondence",
        allow_abbrev=False,
    )
    _add_config_arguments(corr, suppress_defaults=True)
    corr.add_argument("--cloth", required=True)
    corr.add_argument("--overwrite", action="store_true")
    corr.add_argument("--dry-run", action="store_true")

    reg = subparsers.add_parser(
        "register",
        help="register clothing SMPL",
        allow_abbrev=False,
    )
    _add_config_arguments(reg, suppress_defaults=True)
    reg.add_argument("--cloth", required=True)
    reg.add_argument("--correspondence", default=None)
    reg.add_argument("--gender", default="female")
    reg.add_argument("--overwrite", action="store_true")
    reg.add_argument("--dry-run", action="store_true")

    transfer = subparsers.add_parser(
        "transfer",
        help="transfer clothing to a body",
        allow_abbrev=False,
    )
    _add_config_arguments(transfer, suppress_defaults=True)
    transfer.add_argument("--cloth", required=True)
    transfer.add_argument("--body", required=True)
    transfer.add_argument("--target-body-pkl", required=True)
    transfer.add_argument("--source-body-pkl", default=None)
    transfer.add_argument("--body-transform", required=True)
    transfer.add_argument("--output", default=None)
    transfer.add_argument("--scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    transfer.add_argument("--auto-scale", action="store_true")
    transfer.add_argument("--num-frames", type=int, default=DEFAULT_TRANSITION_FRAMES)
    transfer.add_argument("--collision-frames", type=int, default=10)

    all_cmd = subparsers.add_parser(
        "all",
        help="run correspondence, registration, and transfer",
        allow_abbrev=False,
    )
    _add_config_arguments(all_cmd, suppress_defaults=True)
    all_cmd.add_argument("--cloth", required=True)
    all_cmd.add_argument("--body", required=True)
    all_cmd.add_argument("--target-body-pkl", required=True)
    all_cmd.add_argument("--body-transform", required=True)
    all_cmd.add_argument("--output", default=None)
    all_cmd.add_argument("--gender", default="female")
    all_cmd.add_argument("--scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    all_cmd.add_argument("--auto-scale", action="store_true")
    all_cmd.add_argument("--num-frames", type=int, default=DEFAULT_TRANSITION_FRAMES)
    all_cmd.add_argument("--collision-frames", type=int, default=10)
    all_cmd.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    config = _build_config(args)

    if args.command == "correspondence":
        result = predict_clothing_correspondence(
            args.cloth,
            config=config,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    elif args.command == "register":
        result = register_clothing_smpl(
            args.cloth,
            correspondence_npz=args.correspondence,
            config=config,
            gender=args.gender,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    elif args.command == "transfer":
        result = transfer_clothing(
            args.cloth,
            args.body,
            target_body_pkl=args.target_body_pkl,
            source_body_pkl=args.source_body_pkl,
            body_transform_path=args.body_transform,
            output_obj=args.output,
            config=config,
            scale=tuple(args.scale),
            auto_scale=args.auto_scale,
            num_frames=args.num_frames,
            collision_frames=args.collision_frames,
        )
    else:
        result = run_full_pipeline(
            args.cloth,
            args.body,
            target_body_pkl=args.target_body_pkl,
            config=config,
            gender=args.gender,
            body_transform_path=args.body_transform,
            output_obj=args.output,
            scale=tuple(args.scale),
            auto_scale=args.auto_scale,
            num_frames=args.num_frames,
            collision_frames=args.collision_frames,
            overwrite=args.overwrite,
        )

    print(json.dumps(_json_ready(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
