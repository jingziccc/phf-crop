"""Stable, explicit task-level scoring contract metadata."""

SCORER_CONTRACT_VERSION = "task_contract_v2_20260725"

SCORER_CONTRACT = {
    "version": SCORER_CONTRACT_VERSION,
    "vqa_soft_tasks": ["textvqa", "vqav2", "aokvqa"],
    "gqa": "normalized_substring_match",
    "mcq_tasks": ["vstar", "hrbench_4k", "hrbench_8k"],
    "mcq_policy": "explicit_letter_first_else_candidate_sequence_similarity",
    "pope": "yes_no_candidate_match",
    "docvqa_runtime": "legacy_substring_containment_with_empty_guard",
    "docvqa_paper": "anls_threshold_0.5",
    "paper_boundary": "DocVQA paper tables use ANLS; legacy containment is historical only.",
}
