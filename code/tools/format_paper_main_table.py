#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/tables/final_freeze_20260713/main_table.csv"
PAPER_TABLES = ROOT / "results/tables/paper"
SCORE_COLUMNS = ("NoCrop", "Rel-Att", "Grad-Att", "Pure-Grad", "PHF-Crop")
MODEL_ORDER = {"LLaVA": 0, "InstructBLIP": 1, "Qwen2.5-VL": 2}
TASK_ORDER = {
    "Standard": {"AOKVQA": 0, "GQA": 1, "POPE": 2, "TextVQA": 3, "VQAv2": 4},
    "High": {"DocVQA": 0, "VSTAR": 1, "HRBench-4K": 2, "HRBench-8K": 3},
}


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 27:
        raise RuntimeError(f"expected 27 main-table rows, got {len(rows)}")
    return rows


def group_rows(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["Resolution"] == group]
    return sorted(selected, key=lambda row: (MODEL_ORDER[row["Model"]], TASK_ORDER[group][row["Task"]]))


def maxima(row: dict[str, str]) -> set[str]:
    best = max(float(row[column]) for column in SCORE_COLUMNS)
    return {column for column in SCORE_COLUMNS if abs(float(row[column]) - best) < 1e-9}


def markdown_value(row: dict[str, str], column: str) -> str:
    value = row[column]
    return f"**{value}**" if column in maxima(row) else value


def markdown_table(rows: list[dict[str, str]]) -> str:
    columns = ("Model", "Task", "N", *SCORE_COLUMNS, "Gain vs best ViCrop")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [row["Model"], row["Task"], row["N"]]
        values.extend(markdown_value(row, column) for column in SCORE_COLUMNS)
        values.append(row["Gain vs best ViCrop"])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def latex_value(row: dict[str, str], column: str) -> str:
    value = latex_escape(row[column])
    return rf"\textbf{{{value}}}" if column in maxima(row) else value


def latex_table(rows: list[dict[str, str]], *, caption: str, label: str) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & Dataset & $N$ & NoCrop & Rel-Att & Grad-Att & Pure-Grad & PHF-Crop & $\Delta$ \\",
        r"\midrule",
    ]
    previous_model = None
    for row in rows:
        if previous_model is not None and row["Model"] != previous_model:
            lines.append(r"\midrule")
        values = [latex_escape(row["Model"]), latex_escape(row["Task"]), row["N"]]
        values.extend(latex_value(row, column) for column in SCORE_COLUMNS)
        values.append(latex_escape(row["Gain vs best ViCrop"]))
        lines.append(" & ".join(values) + r" \\")
        previous_model = row["Model"]
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{1mm}",
        r"\parbox{\textwidth}{\footnotesize \textbf{Notes.} Bold denotes the best result in each model--dataset row. $\Delta$ is PHF-Crop minus the strongest ViCrop variant. Each ViCrop column uses the stronger available standard/high-resolution implementation for that exact variant.}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    standard = group_rows(rows, "Standard")
    high = group_rows(rows, "High")
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    markdown = "\n".join([
        "# Final Main Results",
        "",
        "Bold denotes the maximum score in each model-dataset row. TextVQA reports all 5,000 examples after ID-safe calibration500 + heldout4500 merging. DocVQA uses ANLS. Each ViCrop column uses its stronger available standard/high-resolution implementation.",
        "",
        "## Standard-resolution benchmarks",
        "",
        markdown_table(standard),
        "",
        "## High-resolution benchmarks",
        "",
        markdown_table(high),
        "",
        "PHF-Crop beats the strongest ViCrop variant on 20/27 rows and on all 12 high-resolution rows. The mean gain is +7.19 points on high-resolution benchmarks and +0.68 points on standard-resolution benchmarks.",
        "",
        "Machine-readable source: [`../final_freeze_20260713/main_table.csv`](../final_freeze_20260713/main_table.csv). Full provenance: [`../final_freeze_20260713/final_result_source_manifest.json`](../final_freeze_20260713/final_result_source_manifest.json).",
        "",
    ])
    (PAPER_TABLES / "main_results.md").write_text(markdown, encoding="utf-8")
    (PAPER_TABLES / "main_results_standard.tex").write_text(
        latex_table(
            standard,
            caption="Results on standard-resolution benchmarks. TextVQA uses the complete 5,000-example evaluation set.",
            label="tab:main-standard",
        ),
        encoding="utf-8",
    )
    (PAPER_TABLES / "main_results_highres.tex").write_text(
        latex_table(
            high,
            caption="Results on high-resolution benchmarks. DocVQA is evaluated with ANLS; the remaining datasets use accuracy.",
            label="tab:main-highres",
        ),
        encoding="utf-8",
    )
    print(f"[complete] paper main tables written to {PAPER_TABLES}")


if __name__ == "__main__":
    main()
