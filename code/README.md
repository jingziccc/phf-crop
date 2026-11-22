# Implementation

- [run_top1_hotspot.py](scripts/run_top1_hotspot.py) runs PHF-Crop and the iterative-local control.
- [run_first_stage_from_trace.py](scripts/run_first_stage_from_trace.py) evaluates the first-stage crop from a saved trace.
- [run_original_attention_baseline.py](scripts/run_original_attention_baseline.py) runs the ViCrop Rel-Att, Grad-Att, and Pure-Grad baselines.
- [run_final_freeze_campaign.py](scripts/run_final_freeze_campaign.py) is the historical campaign wrapper.
- [validate_frozen_phf_contract.py](tools/validate_frozen_phf_contract.py) checks the implementation with synthetic heatmaps.

For installation and a single-sample inference example, see the [project README](../README.md).
