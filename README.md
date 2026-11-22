# PHF-Crop

**Progressive Coarse-to-Fine Heatmap Fusion Cropping for Visual Question Answering in MLLMs**

Zhenghai Chen, Chungang Yan, Junli Wang, Zhanpeng Ni, Ruiyuan Duan, and Yang Ruihua

Findings of the Association for Computational Linguistics: EACL 2027

PHF-Crop is a training-free inference method for visual question answering when the evidence occupies a small part of an image. It follows a single sequence of progressively smaller crops, recomputes question-conditioned attention on each local view, and aligns and fuses the heatmaps in the original image coordinates. The selected crop is then used to answer the question. The implementation supports LLaVA-1.5, InstructBLIP, and Qwen2.5-VL.

## Installation

Clone the repository and create a Python 3.10 environment. Install a CUDA-compatible PyTorch build for model inference. The [environment notes](ENVIRONMENT.md) give the versions used in the paper.

```bash
git clone https://github.com/jingziccc/phf-crop-release.git
cd phf-crop-release
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

Create a JSON list of questions and place your images in a local folder. Each record needs an `id`, a `question`, and an `image_path` relative to the image folder (or an absolute path). For example, save this as `questions.json` and place your image at `images/example.jpg`:

```json
[
  {
    "id": "example-1",
    "image_path": "example.jpg",
    "question": "What color is the sign?"
  }
]
```

Run inference on one sample:

```bash
python code/scripts/run_top1_hotspot.py \
  --model llava --task vstar --gpu 0 --cap 1 \
  --question-path questions.json --image-root images \
  --output-dir outputs/demo --no-evaluate
```

The runner saves the original-image answer, crop-assisted answer, final bounding box, and crop trajectory. To run a benchmark with evaluation, use the corresponding task and annotations in [Data and models](DATA_AND_MODELS.md). The `--model` options are `llava`, `blip`, and `qwen2_5`.

To check the crop and stopping implementation with synthetic heatmaps, run:

```bash
python code/tools/validate_frozen_phf_contract.py
```

## Repository layout

| Path | Contents |
| --- | --- |
| [code/top1_hotspot/](code/top1_hotspot/) | Heatmap fusion, crop selection, stopping, and model adapters |
| [code/scripts/](code/scripts/) | Inference, ViCrop baselines, and ablation runners |
| [code/runtime/](code/runtime/) | Model utilities and task-specific scoring |
| [code/tools/](code/tools/) | Scoring, analysis, and implementation checks |
| [code/experiments/](code/experiments/) | Same-scale random-position control and grouped paired intervals |
| [visualization/](visualization/) | Notebook source and the interactive viewer template |
| [docs/](docs/) | Method, evaluation protocol, and ablations |

See [Supplementary experiments](docs/supplementary_experiments.md) for one-factor sensitivity, same-scale random crops, and paired intervals. Run outputs, precomputed predictions, result tables, visualization exports, benchmark files, and model weights are kept outside the release. See the [reproduction guide](docs/reproducibility.md) for input and scoring details.

## Citation

```bibtex
@inproceedings{chen2027phfcrop,
  title     = {{PHF-Crop}: Progressive Coarse-to-Fine Heatmap Fusion Cropping
               for Visual Question Answering in {MLLMs}},
  author    = {Zhenghai Chen and Chungang Yan and Junli Wang and
               Zhanpeng Ni and Ruiyuan Duan and Yang Ruihua},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```
