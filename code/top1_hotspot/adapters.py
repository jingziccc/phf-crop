from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from .config import COMMON_MIN_CROP_SIDE, QWEN_VISUAL_TOKENS


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    model: Any
    processor: Any
    min_crop_side: int
    heatmap_mode: str
    attention_prompt: Callable[[str], str]
    answer_prompt: Callable[[str], str]
    general_prompt: str
    heatmap_fn: Callable[[Image.Image, str, str, Any, Any], Any]
    original_answer_fn: Callable[[Image.Image, str, Any, Any, int], str]
    crop_answer_fn: Callable[[Image.Image, Image.Image, str, Any, Any, int], str]

    @property
    def vit_input_size(self) -> int:
        """Compatibility alias for older analysis and visualization helpers."""
        return self.min_crop_side


def _decode_llava(processor: Any, generate_ids: Any) -> str:
    text = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if "ASSISTANT: " in text:
        text = text.split("ASSISTANT: ", 1)[1]
    return text.strip()


def llava_single_answer(
    image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(
        model.device,
        torch.bfloat16,
    )
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer = _decode_llava(processor, generate_ids)
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def llava_original_plus_crop_answer(
    image: Image.Image,
    crop_image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    return llava_original_plus_crops_answer(
        image,
        [crop_image],
        prompt,
        model,
        processor,
        max_new_tokens,
    )


def llava_original_plus_crops_answer(
    image: Image.Image,
    crop_images: list[Image.Image],
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    if not crop_images:
        raise ValueError("at least one crop image is required")
    images = [image, *crop_images]
    image_tokens = "<image>" * len(images)
    multi_prompt = prompt.replace("<image>\n", f"{image_tokens}\n", 1)
    inputs = processor(images=images, text=multi_prompt, return_tensors="pt", padding=True).to(
        model.device,
        torch.bfloat16,
    )
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer = _decode_llava(processor, generate_ids)
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def blip_single_answer(
    image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    from blip_decode import decode_blip_short_answer

    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(
        model.device,
        torch.bfloat16,
    )
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer = decode_blip_short_answer(processor, generate_ids)
    if not answer.strip() and hasattr(inputs, "input_ids"):
        prompt_len = int(inputs.input_ids.shape[-1])
        trimmed = generate_ids[:, prompt_len:]
        answer = decode_blip_short_answer(processor, trimmed)
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def blip_crop_only_answer(
    image: Image.Image,
    crop_image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    return blip_single_answer(
        crop_image,
        prompt,
        model,
        processor,
        max_new_tokens,
    )


def blip_qformer_token_concat_answer(
    image: Image.Image,
    crop_image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
    image_order: str = "original_crop",
) -> str:
    import torch

    from blip_decode import decode_blip_short_answer

    if image_order == "original_crop":
        images = [image, crop_image]
    elif image_order == "crop_original":
        images = [crop_image, image]
    elif image_order == "crop_crop":
        images = [crop_image, crop_image]
    else:
        raise ValueError(f"unsupported BLIP qformer image order: {image_order}")
    inputs = processor(
        images=images,
        text=prompt,
        return_tensors="pt",
        padding=True,
    ).to(model.device, torch.bfloat16)
    with torch.no_grad():
        batch_size = int(inputs.input_ids.shape[0])
        num_images = int(inputs.pixel_values.shape[0] // batch_size)
        image_embeds = model.vision_model(
            inputs.pixel_values,
            return_dict=True,
        ).last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(
            query_tokens.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        qformer_attention_mask = getattr(inputs, "qformer_attention_mask", None)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(inputs.qformer_input_ids)
        qformer_input_ids = inputs.qformer_input_ids.repeat_interleave(num_images, dim=0)
        qformer_attention_mask = qformer_attention_mask.repeat_interleave(num_images, dim=0)
        qformer_attention_mask = torch.cat(
            [query_attention_mask, qformer_attention_mask],
            dim=1,
        )
        query_outputs = model.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]
        language_model_inputs = model.language_projection(query_output)
        language_model_inputs = language_model_inputs.reshape(
            batch_size,
            num_images * int(model.config.num_query_tokens),
            -1,
        )

        image_token_index = getattr(model.config, "image_token_index", None)
        if image_token_index is None:
            text_input_ids = inputs.input_ids[0:1]
            text_attention_mask = inputs.attention_mask[0:1]
        else:
            first_input_ids = inputs.input_ids[0]
            first_attention_mask = inputs.attention_mask[0]
            text_mask = first_input_ids != int(image_token_index)
            image_tokens = torch.full(
                (language_model_inputs.shape[1],),
                int(image_token_index),
                dtype=first_input_ids.dtype,
                device=first_input_ids.device,
            )
            image_attention = torch.ones(
                (language_model_inputs.shape[1],),
                dtype=first_attention_mask.dtype,
                device=first_attention_mask.device,
            )
            text_input_ids = torch.cat(
                [image_tokens, first_input_ids[text_mask]],
                dim=0,
            ).unsqueeze(0)
            text_attention_mask = torch.cat(
                [image_attention, first_attention_mask[text_mask]],
                dim=0,
            ).unsqueeze(0)

        inputs_embeds = model.get_input_embeddings()(text_input_ids)
        if image_token_index is None:
            language_attention_mask = torch.ones(
                language_model_inputs.size()[:-1],
                dtype=torch.long,
                device=language_model_inputs.device,
            )
            inputs_embeds = torch.cat(
                [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
                dim=1,
            )
            text_attention_mask = torch.cat(
                [language_attention_mask, text_attention_mask.to(language_attention_mask.device)],
                dim=1,
            )
        else:
            special_image_mask = (
                text_input_ids == int(image_token_index)
            ).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds[special_image_mask] = language_model_inputs.flatten().to(inputs_embeds.device)

        generate_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=text_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer = decode_blip_short_answer(processor, generate_ids)
    if not answer.strip():
        trimmed = generate_ids[:, int(text_input_ids.shape[-1]) :]
        answer = decode_blip_short_answer(processor, trimmed)
    del inputs, image_embeds, query_outputs, language_model_inputs, inputs_embeds, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def qwen_single_answer(
    image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    from utils import encode_base64, prepare_qwen2_5_input

    image_str = encode_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def qwen_original_plus_crop_answer(
    image: Image.Image,
    crop_image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    from utils import encode_base64, prepare_qwen2_5_input

    image_str = encode_base64(image)
    crop_str = encode_base64(crop_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "image", "image": f"data:image;base64,{crop_str}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def load_model_adapter(
    model_name: str,
    device: str,
    *,
    min_crop_side: int = COMMON_MIN_CROP_SIDE,
    vit_input_size: int | None = None,
    blip_bg_sub_alpha: float = 1.0,
) -> ModelAdapter:
    import torch
    from transformers import (
        AutoProcessor,
        InstructBlipForConditionalGeneration,
        InstructBlipProcessor,
        LlavaForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    )

    from info import model_to_fullname

    local_files_only = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )

    if vit_input_size is not None:
        if int(min_crop_side) != COMMON_MIN_CROP_SIDE and int(min_crop_side) != int(vit_input_size):
            raise ValueError("min_crop_side and legacy vit_input_size disagree")
        min_crop_side = int(vit_input_size)

    if model_name == "llava":
        from llava_methods import direct_attention_llava

        model = LlavaForConditionalGeneration.from_pretrained(
            model_to_fullname["llava"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            local_files_only=local_files_only,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_to_fullname["llava"],
            local_files_only=local_files_only,
        )
        model.eval()
        return ModelAdapter(
            name="llava",
            model=model,
            processor=processor,
            min_crop_side=int(min_crop_side),
            heatmap_mode="llava_direct_question_attention",
            attention_prompt=lambda q: f"<image>\nUSER: {q} Answer the question using a single word or phrase.\nASSISTANT:",
            answer_prompt=lambda q: f"<image>\nUSER: {q} Answer the question using a single word or phrase.\nASSISTANT:",
            general_prompt="<image>\nUSER: Write a general description of the image. Answer the question using a single word or phrase.\nASSISTANT:",
            heatmap_fn=direct_attention_llava,
            original_answer_fn=llava_single_answer,
            crop_answer_fn=llava_original_plus_crop_answer,
        )

    if model_name == "blip":
        os.environ["QPES_BLIP_BG_SUB_ALPHA"] = str(blip_bg_sub_alpha)
        from blip_methods import (
            LM_LAYER,
            QFORMER_LAYER,
            register_removed_attention_blip,
        )

        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_to_fullname["blip"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        ).to(device)
        processor = InstructBlipProcessor.from_pretrained(
            model_to_fullname["blip"],
            local_files_only=local_files_only,
        )
        model.eval()
        return ModelAdapter(
            name="blip",
            model=model,
            processor=processor,
            min_crop_side=int(min_crop_side),
            heatmap_mode=(
                f"instructblip_register_removed_rawsub_no_norm_alpha{blip_bg_sub_alpha:g}"
                f"_qformer{int(QFORMER_LAYER)}_lm{int(LM_LAYER)}"
            ),
            attention_prompt=lambda q: f"Question: {q} Short answer:",
            answer_prompt=lambda q: f"Question: {q} Short answer:",
            general_prompt="Question: Write a general description of the image. Short answer:",
            heatmap_fn=register_removed_attention_blip,
            original_answer_fn=blip_single_answer,
            crop_answer_fn=blip_crop_only_answer,
        )

    if model_name == "qwen2_5":
        from qwen2_5_methods import (
            ATT_LAYER,
            direct_attention_qwen2_5,
        )
        max_pixels = QWEN_VISUAL_TOKENS * 28 * 28
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_to_fullname["qwen2_5"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            local_files_only=local_files_only,
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_to_fullname["qwen2_5"],
            max_pixels=max_pixels,
            local_files_only=local_files_only,
        )
        processor.image_processor.size["longest_edge"] = max_pixels
        model.eval()
        return ModelAdapter(
            name="qwen2_5",
            model=model,
            processor=processor,
            min_crop_side=int(min_crop_side),
            heatmap_mode=(
                f"qwen2_5_direct_question_attention_layer{int(ATT_LAYER)}"
                f"_vis{QWEN_VISUAL_TOKENS}"
            ),
            attention_prompt=lambda q: f"{q} Answer the question using a single word or phrase.",
            answer_prompt=lambda q: f"{q} Answer the question using a single word or phrase.",
            general_prompt=(
                "Write a general description of the image. "
                "Answer the question using a single word or phrase."
            ),
            heatmap_fn=direct_attention_qwen2_5,
            original_answer_fn=qwen_single_answer,
            crop_answer_fn=qwen_original_plus_crop_answer,
        )

    raise ValueError(f"unsupported model: {model_name}")
