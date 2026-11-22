from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE_ROOT / "data"
IMAGE_ROOT = PACKAGE_ROOT / "images"
MODEL_ROOT = PACKAGE_ROOT / "models"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "baseline_attention_results"

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LOCAL_LLAVA_MODEL_PATH = MODEL_ROOT / "llava-1.5-7b-hf"
LLAVA_MODEL_PATH = str(LOCAL_LLAVA_MODEL_PATH if LOCAL_LLAVA_MODEL_PATH.exists() else LLAVA_MODEL_ID)
BLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"
LOCAL_BLIP_MODEL_PATH = MODEL_ROOT / "instructblip-vicuna-7b"
BLIP_MODEL_PATH = str(LOCAL_BLIP_MODEL_PATH if LOCAL_BLIP_MODEL_PATH.exists() else BLIP_MODEL_ID)
QWEN_VL_MODEL_ID = "Qwen/Qwen-VL"
QWEN2_5_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_QWEN2_5_MODEL_PATH = MODEL_ROOT / "Qwen2.5-VL-3B-Instruct"
QWEN2_5_MODEL_PATH = str(
    LOCAL_QWEN2_5_MODEL_PATH
    if LOCAL_QWEN2_5_MODEL_PATH.exists()
    else QWEN2_5_MODEL_ID
)

task_to_image_path = {
    "aokvqa": str(DATA_ROOT / "aokvqa" / "images"),
    "gqa": str(DATA_ROOT / "gqa" / "images"),
    "textvqa": str(DATA_ROOT / "textvqa" / "images"),
    "docvqa": str(DATA_ROOT / "docvqa" / "images"),
    "pope": str(DATA_ROOT / "pope" / "images"),
    "vstar": str(DATA_ROOT / "vstar" / "images"),
    "vqav2": str(DATA_ROOT / "vqav2" / "images"),
    "hrbench_4k": str(DATA_ROOT / "hrbench_4k" / "images"),
    "hrbench_8k": str(DATA_ROOT / "hrbench_8k" / "images"),
}

task_to_question_path = {
    "aokvqa": str(DATA_ROOT / "aokvqa" / "data.json"),
    "gqa": str(DATA_ROOT / "gqa" / "data.json"),
    "textvqa": str(DATA_ROOT / "textvqa" / "data.json"),
    "docvqa": str(DATA_ROOT / "docvqa" / "data.json"),
    "pope": str(DATA_ROOT / "pope" / "data.json"),
    "vstar": str(DATA_ROOT / "vstar" / "data.json"),
    "vqav2": str(DATA_ROOT / "vqav2" / "data.json"),
    "hrbench_4k": str(DATA_ROOT / "hrbench_4k" / "data.json"),
    "hrbench_8k": str(DATA_ROOT / "hrbench_8k" / "data.json"),
}

task_to_subset_question_path = {
    "docvqa": str(DATA_ROOT / "docvqa" / "data.subset5000.json"),
    "pope": str(DATA_ROOT / "pope" / "data.subset5000.json"),
    "gqa": str(DATA_ROOT / "gqa" / "data.subset5000.json"),
    "vqav2": str(DATA_ROOT / "vqav2" / "data.subset5000.json"),
}

model_to_fullname = {
    "llava": LLAVA_MODEL_PATH,
    "blip": BLIP_MODEL_PATH,
    "qwen2_5": QWEN2_5_MODEL_PATH,
}

notebook_model_to_fullname = {
    **model_to_fullname,
    "qwen_vl": QWEN_VL_MODEL_ID,
}

demo_image_paths = {
    "demo1": str(IMAGE_ROOT / "demo1.png"),
    "demo2": str(IMAGE_ROOT / "demo2.png"),
    "demo3": str(IMAGE_ROOT / "demo3.jpg"),
}
