# Remaining release checks

1. Supply A4 epoch-7 checkpoint download URL and SHA-256, and auxiliary weight provenance/checksums.
2. Confirm authorship wording and insert original paper authors, year, DOI, and upstream code citations.
3. Repair the malformed flake8/timm line in source requirements and validate the declared Python environment. Desktop and Nano environments must be documented separately.
4. Verify portable auxiliary checkpoint paths; metadata embeds the original workstation path.
5. Provide the Jetson benchmark summary and power mode, clocks, precision, warm-up count, measured pair count, and included timing stages.
6. Commit the desired inference script and documentation/assets to a suitable branch. The supplied revision predates these new documentation files.
7. Run setup and the full evaluation in a fresh environment with weights/data before describing reproduction as independently validated.

Validation performed for this package: inspected source wrappers and CLI flags; recomputed unaligned position ATE and endpoint error from supplied pose files; generated plots from those files. No neural-network training or inference was executed.
