# AENIN

## Adaptive Edge-Node Interaction Network

This repository contains the code release for the **accepted AENIN research
article**. AENIN is a graph-learning framework developed for ASD-vs-TD
classification from resting-state fMRI brain-connectivity data.

The repository was prepared from the two accepted-study notebooks supplied by
the authors:

- `AENIN_lasso(2).ipynb`
- `AENIN_baseline_comparison.ipynb`

The public release removes machine-specific paths and notebook outputs while
retaining the research implementation.

## What is included

The release contains:

- AENIN preprocessing and graph construction;
- adaptive phase synchronization / PLV connectivity processing;
- ROI-level feature extraction;
- correlation-feature augmentation used in the supplied implementation;
- the Adaptive Edge-Node Interaction layer and AENIN classifier;
- stratified 5-fold model evaluation;
- SCRM analyses (CIS, ITS, and PCS);
- group statistical analysis and visualization;
- ablation-study code;
- dataset diagnostics;
- the eight supplied baseline model implementations;
- 5-fold and site-wise baseline comparison utilities; and
- cleaned copies of both research notebooks.

## Main AENIN data flow

The executable implementation in the accepted-study notebook follows the
general sequence:

```text
rs-fMRI NIfTI
     │
     ▼
ROI time-series extraction
     │
     ▼
PLV connectivity
     │
     ▼
Adaptive thresholding / graph construction
     │
     ├──────────────► edge_index + edge_attr
     │
     ▼
ROI node features
     │
     ▼
Correlation-row augmentation
     │
     ▼
AENIN
     │
     ▼
Graph-level ASD / TD prediction
```

### Feature-count note

One comment/docstring in the original AENIN notebook describes the node vector
as having 13 features. The executable function, however, allocates and returns
**18 base node features (indices 0–17)** and the subject-processing code then
concatenates each ROI's correlation row. This release follows the executable
implementation. The stale feature-count wording was corrected in the packaged
Python module and the change is documented in `CHANGELOG.md`.

## Repository structure

```text
AENIN_Final_GitHub/
├── README.md
├── CODE_AVAILABILITY.md
├── REPRODUCIBILITY.md
├── DATA.md
├── SECURITY.md
├── CHANGELOG.md
├── CITATION.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── src/
│   ├── aenin/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   └── run.py
│   └── aenin_baselines/
│       ├── __init__.py
│       ├── feature_pipeline.py
│       ├── models.py
│       ├── dataset_bridge.py
│       └── train_compare.py
│
├── scripts/
│   ├── run_aenin.py
│   └── run_baselines.py
│
├── notebooks/
│   ├── AENIN_accepted_release.ipynb
│   └── AENIN_baseline_comparison_release.ipynb
│
├── data/
│   └── README.md
└── results/
    └── .gitkeep
```

## Installation

Python 3.10 or 3.11 is recommended for a reproducible scientific environment.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install a PyTorch build appropriate for your CPU/CUDA environment, then:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

Place the NIfTI files in:

```text
data/
├── ASD/
│   ├── subject_001.nii.gz
│   └── ...
└── TD/
    ├── subject_001.nii.gz
    └── ...
```

The original neuroimaging data are **not distributed** in this repository.
Obtain the data from the corresponding official source and comply with its
terms of use and acknowledgement requirements.

## Run AENIN

Example:

```bash
python scripts/run_aenin.py \
    --data-root ./data \
    --atlas aal \
    --epochs 200 \
    --hidden 64 \
    --batch-size 4 \
    --aps-alpha 0.5 \
    --output-dir ./results/aenin
```

To use a balanced subset:

```bash
python scripts/run_aenin.py \
    --data-root ./data \
    --balance-per-class 400 \
    --output-dir ./results/aenin_400_400
```

To save the processed graph dataset:

```bash
python scripts/run_aenin.py \
    --data-root ./data \
    --save-dataset \
    --output-dir ./results/aenin
```

To reuse a processed dataset:

```bash
python scripts/run_aenin.py \
    --dataset-pt ./results/aenin/dataset.pt \
    --output-dir ./results/aenin_reuse
```

## Run the baseline comparison

The supplied baseline notebook contains compact implementations of:

1. GroupINN
2. BrainGNN
3. EV-GCN
4. AL-NEGAT
5. Ex-NEGAT
6. DeepASD
7. GNN-LSTM
8. MCDGLN

Five-fold comparison:

```bash
python scripts/run_baselines.py \
    --data-root ./data \
    --mode 5fold \
    --epochs 100 \
    --balance-per-class 400
```

Site-wise comparison:

```bash
python scripts/run_baselines.py \
    --data-root ./data \
    --mode site \
    --epochs 100 \
    --balance-per-class 400
```

The baseline notebook itself states that these are **faithful-but-compact
re-implementations rather than the original authors' repositories**. That
qualification is preserved in this release.

### Dynamic-baseline note

`GNN-LSTM` and `MCDGLN` use the ROI time series. If a cached dataset was
created without `time_series`, the supplied dataset bridge falls back to a
zero temporal placeholder. For a meaningful dynamic-baseline comparison,
rebuild the baseline dataset with the time series retained.

## Reproducibility

The main experiment uses seeded NumPy/PyTorch execution and stratified
cross-validation. Numerical values can still vary across PyTorch, CUDA,
PyTorch Geometric, Nilearn, atlas, operating-system, and hardware versions.

See `REPRODUCIBILITY.md` before attempting an exact replication.

## Code availability

The associated article has been accepted. The implementation and supporting
release material are therefore provided here for reproducibility.

See `CODE_AVAILABILITY.md`.

## Citation

Add the definitive article citation and DOI to `CITATION.md` once the final
publisher metadata are available.

## License

A license is intentionally **not guessed or assigned** in this generated
release. Add the software license approved by the authors, institution, and
publisher before public distribution.
