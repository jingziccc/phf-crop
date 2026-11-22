"""Grouped paired percentile bootstrap over matched per-example scores."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_scores(rows, get_score, group_key):
    scores = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in scores:
            raise ValueError(f"duplicate sample ID {sample_id}")
        score = float(get_score(row))
        if not 0 <= score <= 1:
            raise ValueError(f"expected a score in [0, 1] for sample {sample_id}")
        group = str(row[group_key]) if group_key else sample_id
        scores[sample_id] = (group, score)
    return scores


def paired_interval(baseline, candidate, replicates=5000, seed=4108):
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("scores must have the same nonempty sample ID set")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    groups, deltas = [], []
    for sample_id, (group, score) in baseline.items():
        candidate_group, candidate_score = candidate[sample_id]
        if group != candidate_group:
            raise ValueError(f"mismatched group for sample {sample_id}")
        groups.append(group)
        deltas.append(candidate_score - score)

    _, indices = np.unique(groups, return_inverse=True)
    counts = np.bincount(indices)
    sums = np.bincount(indices, weights=deltas)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates)
    for start in range(0, replicates, 128):
        stop = min(start + 128, replicates)
        picks = rng.integers(len(counts), size=(stop - start, len(counts)))
        bootstrap[start:stop] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lower, upper = np.quantile(100 * bootstrap, [0.025, 0.975])
    return {
        "samples": len(deltas),
        "groups": len(counts),
        "delta_pp": 100 * sums.sum() / counts.sum(),
        "ci95_pp": [float(lower), float(upper)],
        "replicates": replicates,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-score", default="score")
    parser.add_argument("--candidate-score", default="score")
    parser.add_argument("--group-key", help="Use the image or base-question ID for grouped sampling")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=4108)
    parser.add_argument("--task", choices=("vstar", "textvqa"),
                        help="Score crop_answer from raw prediction files")
    args = parser.parse_args()
    baseline_rows = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_rows = json.loads(args.candidate.read_text(encoding="utf-8"))
    if args.task:
        def questions(rows):
            return {str(row["id"]): (row["question"], row["labels"]) for row in rows}
        if questions(baseline_rows) != questions(candidate_rows):
            parser.error("raw prediction files must match on question IDs, prompts and labels")

        if args.task == "vstar":
            from random_position import score_vstar

            def score(row):
                return score_vstar(row["crop_answer"], row)
        else:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
            from get_score import get_acc

            def score(row):
                return get_acc(row["crop_answer"], row["labels"]) / 100

        baseline = read_scores(baseline_rows, score, args.group_key)
        candidate = read_scores(candidate_rows, score, args.group_key)
    else:
        baseline = read_scores(baseline_rows, lambda row: row[args.baseline_score], args.group_key)
        candidate = read_scores(candidate_rows, lambda row: row[args.candidate_score], args.group_key)
    report = paired_interval(baseline, candidate, args.replicates, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
