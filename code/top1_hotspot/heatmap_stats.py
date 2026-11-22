from __future__ import annotations

import numpy as np


def compute_normalized_heatmap_entropy(att_map, eps: float = 1e-8) -> float:
    heatmap = np.asarray(att_map, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError("att_map must be a 2D array")

    positive = np.clip(heatmap.reshape(-1), a_min=0.0, a_max=None)
    total = float(positive.sum())
    if total <= eps:
        return 0.0

    probs = np.clip(positive / total, eps, None)
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = float(np.log(float(probs.size)))
    if max_entropy <= eps:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))
