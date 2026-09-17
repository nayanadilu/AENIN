# Reproducibility Notes

## Source

This release was prepared from the supplied accepted-study notebooks:

- `AENIN_lasso(2).ipynb`
- `AENIN_baseline_comparison.ipynb`

Cleaned notebook copies are retained under `notebooks/`.

## Recommended environment

Use an isolated Python environment and record exact package versions for the
archived run:

```bash
pip freeze > environment_frozen.txt
```

PyTorch and PyTorch Geometric builds should be selected to match the machine's
CUDA/driver stack when GPU execution is used.

## Main AENIN settings represented in the source notebook

The current `Config` in the supplied AENIN notebook contains, among other
settings:

- AAL as the default atlas;
- APS alpha = 0.5;
- minimum edge count = 30;
- hidden dimension = 64;
- one AENIN layer;
- dropout = 0.5;
- four attention heads;
- learning rate = 1e-4;
- weight decay = 1e-4;
- batch size = 4;
- 200 epochs;
- five folds; and
- seed = 42.

The command-line runner exposes the most commonly changed settings and retains
the remaining defaults in `aenin.Config`.

## Feature representation

The executable AENIN notebook creates 18 base node features and concatenates
the ROI correlation row. For AAL with 116 ROIs this produces a graph node
feature width of 134, and the training workflow infers the dimension directly
from the processed graph rather than relying on the class-level placeholder.

## Baseline comparison

The separate baseline notebook defines its own feature-pipeline module and
eight compact baseline model re-implementations. It uses 18 base node features
and the same correlation-row augmentation described in that notebook.

The baseline notebook explicitly describes those models as compact
re-implementations, not as the original authors' repositories.

## Potential sources of numerical variation

Exact values can vary with:

- source neuroimaging files and their preprocessing history;
- Nilearn/Nibabel versions;
- atlas files downloaded by Nilearn;
- SciPy/NumPy implementations;
- PyTorch/PyTorch Geometric versions;
- CUDA/cuDNN versions and GPU hardware; and
- nondeterministic third-party operations.

Keep package versions, hardware information, and the generated result files
with every archival experiment.
