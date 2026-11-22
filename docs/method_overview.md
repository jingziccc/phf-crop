# PHF-Crop method overview

PHF-Crop progressively refines a single crop sequence. It starts from a
question-relevant full-image heatmap (tiled at 1024 px when needed), thresholds
by cumulative topmass, finds 8-connected components, and selects the top
component with `peak * mean * log1p(area)`.  The component's heatmap-weighted
centroid defines a square crop.

The crop begins at 50% image area and halves its area at each stage until the
336 px ViT floor.  Each non-stopped local heatmap is sum-normalized, mapped
back to original-image coordinates, and coverage-averaged with prior evidence.
The next hotspot is selected from that fused canvas.

The frozen configuration fixes `topmass=0.55`, `tile1024`, 8-connectivity,
coverage-average fusion, and adaptive stop at a component-to-candidate short-
side ratio of `>= 1.0`.  LLaVA and Qwen use direct question attention and
original-plus-crop answering; InstructBLIP uses positive
question-minus-general attention and crop-only answering.

The exact configuration is [../code/top1_hotspot/frozen_config.json](../code/top1_hotspot/frozen_config.json).
