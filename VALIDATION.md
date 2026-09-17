# Release Validation

- Python files checked: 10
- Python syntax errors: 0
- Cleaned notebooks checked: 2
- Notebook JSON errors: 0
- Known private absolute paths remaining: 0

## Runtime validation limitation

Static syntax validation was completed in the packaging environment.
Full neuroimaging runtime execution was not performed because this
environment does not contain the user's ABIDE NIfTI dataset and does not
currently have all neuroimaging/PyTorch-Geometric dependencies installed.
