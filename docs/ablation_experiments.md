# Ablation controls

| Arm | Inference policy |
| --- | --- |
| NoCrop | Answer from the original image. |
| First-stage only | Use the first crop from a matching PHF trace. |
| Iterative Local | Use the same progressive crop schedule and select the next crop from the latest local heatmap. |
| PHF-Crop | Align and fuse the heatmap history on the original image coordinates before selecting the next crop. |

The [PHF runner](../code/scripts/run_top1_hotspot.py) records both original-image
and crop-assisted answers. Use `--localization-strategy iterative_local` for
the latest-heatmap control. The [first-stage runner](../code/scripts/run_first_stage_from_trace.py)
takes a saved trace from the matching run. Compare Full PHF with Iterative Local
to isolate the contribution of fusion under the matching schedule.
