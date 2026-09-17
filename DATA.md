# Data

The original neuroimaging dataset is not included in this repository.

For the main AENIN runner, organize subject files as:

```text
data/
├── ASD/
│   ├── subject_001.nii.gz
│   └── ...
└── TD/
    ├── subject_001.nii.gz
    └── ...
```

The supplied code labels files under `ASD/` as class `1` and files under
`TD/` as class `0`.

Do not commit participant data, restricted files, cached processed datasets,
or derived files when redistribution is not explicitly permitted.
