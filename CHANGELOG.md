# Release Cleanup Notes

## Version 1.0.0 — accepted-article GitHub release

The final public package was prepared from the two supplied research notebooks.

Changes made for repository release:

1. Removed notebook execution outputs and execution counters from the release
   notebook copies.
2. Replaced workstation/Colab-specific absolute paths with repository-relative
   paths such as `./data`.
3. Extracted the latest AENIN definition cells into `src/aenin/core.py`.
4. Removed one notebook-only line consisting solely of periods that prevented
   the extracted source from parsing as standalone Python.
5. Updated a stale AENIN feature-count docstring from 13 to 18 because the
   executable function allocates and returns 18 feature columns and uses
   indices 0 through 17.
6. Added command-line wrappers around the supplied AENIN and baseline workflows.
7. Converted notebook-global baseline definitions into importable modules.
8. Added repository documentation, `.gitignore`, packaging metadata, and
   reproducibility notes.
9. No research dataset, local cached graph dataset, credential, trained
   checkpoint, or notebook result output is included.

These changes are packaging/documentation changes; no intentional redesign of
the research algorithms was introduced.
10. Added the accepted article DOI: `10.1016/j.health.2026.100493` to the release documentation.
