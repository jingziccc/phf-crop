# Supplementary experiments

Install the [project dependencies](../README.md#installation) first. Use your
own benchmark inputs and write generated artifacts under the ignored
`outputs/` directory.

## One-factor sensitivity

Use the same LLaVA runner for the default and for each single-factor variant.
For example, run `rho045` on the same VSTAR question file as the default:

```bash
python code/scripts/run_top1_hotspot.py --model llava --task vstar --gpu 0 \
  --question-path data/vstar/data.json --image-root data/vstar/images \
  --sensitivity-variant rho045 --no-evaluate --output-dir outputs/sensitivity/vstar/rho045
```

Omit `--sensitivity-variant` for the default. The presets cover minimum side
(`min224`, `min448`), initial area (`initial025`, `initial075`), area decay
(`decay025`, `decay075`), top-mass (`rho045`, `rho050`, `rho060`), and the
component score (`score_mass`, `score_no_area`). For TextVQA, pass the same
held-out question file to all runs and exclude any calibration examples.
To score matching prediction files, use the paired script below with
`--task vstar` or `--task textvqa --group-key image_path`. These options use
the strict VSTAR multiple-choice and TextVQA consensus scorers, respectively.
Use the appropriate image identifier if your predictions use a different
field. The original subset membership and per-example scores are not bundled.

## Same-scale random positions

First produce a Qwen2.5-VL PHF prediction file on VSTAR. The control reads the
saved crop sizes and stopping depth from that file. For each of the three
seeds, it samples a valid whole-image position at each saved size and
generates an answer from the original image plus the final random crop. The
script also runs original-image and saved-PHF-crop answering with the same
model and strict multiple-choice scorer.

```bash
python code/experiments/random_position.py \
  --reference outputs/qwen-vstar/phf.json --image-root data/vstar/images \
  --output outputs/random-position.json
```

Replace the reference path with the prediction file written by the PHF
runner. This is a location control conditioned on the saved scale schedule.

## Grouped paired intervals

Combine matching per-example scores in JSON arrays with an `id` and a score
in `[0, 1]`. Use `--group-key` with an image or base-question field when
multiple rows share a sampling unit. For independent questions, the script
groups by sample ID. Compute scores using the same questions and scorer for
both inputs. The comparison is candidate minus baseline in percentage points.

```bash
python code/experiments/paired_bootstrap.py \
  outputs/random-position.json outputs/random-position.json \
  --baseline-score random_mean_score --candidate-score phf_score
```

The defaults are 5,000 resamples and seed 4108.
