# Reproducing an inference run

Prepare local annotations and images as described in [Data and models](../DATA_AND_MODELS.md).
Record the checkpoint revision, question file, image directory, and inference
command for each run. For a benchmark, pass the exact subset via
`--question-path` and omit `--no-evaluate` when the task's answer annotations
are available.

The runner uses `short_question` for attention when present and the full
question for answer generation. It saves configuration and input fingerprints
alongside each run. `--resume` checks these fingerprints before reusing
predictions. The locked setting runs with or without the original calibration
manifest. Calibration runs require that manifest.

The task-specific scoring code is in
[scoring_contract.py](../code/runtime/scoring_contract.py). DocVQA uses ANLS.
VSTAR and HRBench use multiple-choice scoring. For paired comparisons, use
identical examples and scoring protocols across methods.

See the [ablation definitions](ablation_experiments.md) for the inference
controls. The historical campaign and aggregation scripts assume access to
the original benchmark preparation and run manifests. For a fresh checkout,
use the single-task entry point in the [README](../README.md) with locally
prepared inputs.
