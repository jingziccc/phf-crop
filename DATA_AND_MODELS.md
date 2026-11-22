# Data and models

The inference runner takes a JSON array of records. Each record contains an
`id`, a `question`, and either an `image_path` or an `image_id`. A
relative `image_path` is resolved under the image directory. An `image_id`
is resolved as `<image_id>.jpg`. An optional `short_question` is used for
heatmap extraction. Answer generation uses the full question. Benchmark
scoring also requires the task's answer annotations.

Pass `--question-path` and `--image-root` explicitly, or use the default
layout:

```text
data/<task>/data.json
data/<task>/images/
```

Supported tasks are `aokvqa`, `docvqa`, `gqa`, `pope`, `textvqa`,
`vqav2`, `vstar`, `hrbench_4k`, and `hrbench_8k`. The historical campaign
wrapper looks for `data.subset5000.json` for its DocVQA, GQA, POPE, and VQAv2
subsets. Use benchmark files from their respective providers under each
dataset's terms. See [reproducibility](docs/reproducibility.md) for the scoring
protocol.

| Runner option | Model identifier | Optional directory under `models/` |
| --- | --- | --- |
| `llava` | `llava-hf/llava-1.5-7b-hf` | `llava-1.5-7b-hf` |
| `blip` | `Salesforce/instructblip-vicuna-7b` | `instructblip-vicuna-7b` |
| `qwen2_5` | `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen2.5-VL-3B-Instruct` |

A model directory or symlink takes precedence over the hosted identifier.
Without a local directory, Transformers resolves the identifier via Hugging
Face Hub. Consult the model cards for their respective terms.
