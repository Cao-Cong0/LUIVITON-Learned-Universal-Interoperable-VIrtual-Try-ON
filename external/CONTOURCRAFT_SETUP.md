# ContourCraft Setup

Follow [ContourCraft](https://github.com/Dolorousrtur/ContourCraft) to create the `ccraft`
environment with its CUDA dependencies. Install LUIVITON from the repository root:

```bash
conda activate ccraft
pip install -e .
```

## Source

The main installation's `scripts/setup_third_party.sh` installs ContourCraft and the LUIVITON patches.
To use a separate checkout instead:

```bash
git clone https://github.com/Dolorousrtur/ContourCraft.git /path/to/ContourCraft
python scripts/install_contourcraft_compat.py --contourcraft-root /path/to/ContourCraft
```

## Data

Download the checkpoint and auxiliary data from ContourCraft's instructions, and obtain SMPL
under its model license. Place the minimal runtime files here:

```text
third_party/contourcraft_data/
├── aux_data/
│   ├── body_models/smpl/SMPL_FEMALE.pkl
│   └── smpl_aux.pkl
└── trained_models/contourcraft.pth
```

Link the SMPL models for registration:

```bash
conda run -n Luiviton bash scripts/setup_rvh.sh
```

RVH supplies the pose prior at `third_party/RVH_Mesh_Registration/assets/priors/body_prior.pkl`.
Additional ContourCraft training and example data are not required.

For custom paths, use `--contourcraft-root` and `--contourcraft-data` with the transfer CLI.
