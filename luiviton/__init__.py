"""Public Python API for the Luiviton clothing transfer pipeline."""

from .pipeline import (
    CorrespondenceResult,
    FullPipelineResult,
    PipelineConfig,
    RegistrationResult,
    TransferResult,
    predict_clothing_correspondence,
    register_clothing_smpl,
    run_full_pipeline,
    transfer_clothing,
)

__all__ = [
    "CorrespondenceResult",
    "FullPipelineResult",
    "PipelineConfig",
    "RegistrationResult",
    "TransferResult",
    "predict_clothing_correspondence",
    "register_clothing_smpl",
    "run_full_pipeline",
    "transfer_clothing",
]
