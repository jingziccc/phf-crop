import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *

# currently select 22 but feel free to try other layers
ATT_LAYER = 22
QWEN_GRAD_EPS = 1e-8


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return float(default)
    return float(value)


QPES_QWEN_BG_SUB_ALPHA = env_float("QPES_QWEN_BG_SUB_ALPHA", 0.0)


def _normalize_positive_map(att_map):
    arr = np.asarray(att_map, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / total


def _qwen_vision_span(inputs, processor):
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    return pos, pos_end


def _qwen_merged_heatmap_shape(inputs):
    return (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()


def _next_token_self_loss(outputs):
    ce = nn.CrossEntropyLoss()
    next_logits = outputs.logits[:, -1, :]
    next_class = torch.argmax(next_logits, dim=1)
    return -ce(next_logits, next_class)

def rel_attention_qwen2_5(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)

    att_shape = _qwen_merged_heatmap_shape(inputs)
    pos, pos_end = _qwen_vision_span(inputs, processor)

    outputs = model(**inputs, output_attentions=True)
    general_outputs = model(**general_inputs, output_attentions=True)

    att = outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
    general_att = general_outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

    if QPES_QWEN_BG_SUB_ALPHA > 0:
        prompt_att = _normalize_positive_map(att)
        background_att = _normalize_positive_map(general_att)
        att_map = np.maximum(
            prompt_att - float(QPES_QWEN_BG_SUB_ALPHA) * background_att,
            0.0,
        )
        if float(att_map.sum()) <= 0.0:
            att_map = prompt_att
    else:
        att_map = att / general_att

    att_map = att_map.reshape(att_shape)

    return att_map


def direct_attention_qwen2_5(image, prompt, general_prompt, model, processor):
    """Return question-conditioned visual-token attention without a baseline pass."""
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
    inputs = prepare_qwen2_5_input(messages, processor).to(
        model.device, torch.bfloat16
    )
    att_shape = _qwen_merged_heatmap_shape(inputs)
    pos, pos_end = _qwen_vision_span(inputs, processor)
    outputs = model(**inputs, output_attentions=True)
    att_map = outputs["attentions"][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0)
    return (
        att_map.to(torch.float32)
        .detach()
        .cpu()
        .numpy()
        .reshape(att_shape)
    )


def gradient_attention_qwen2_5(image, prompt, general_prompt, model, processor):
    image_str = encode_base64(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    att_shape = _qwen_merged_heatmap_shape(inputs)
    pos, pos_end = _qwen_vision_span(inputs, processor)

    model.zero_grad(set_to_none=True)
    outputs = model(**inputs, output_attentions=True)
    loss = _next_token_self_loss(outputs)
    attention = outputs['attentions'][ATT_LAYER]
    grads = torch.autograd.grad(loss, attention, retain_graph=False)[0]
    grad_att = attention * F.relu(grads)
    att_map = (
        grad_att[0, :, -1, pos:pos_end]
        .mean(dim=0)
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
        .reshape(att_shape)
    )
    return att_map


def _pixel_gradient_heatmap_qwen2_5(image, prompt, model, processor):
    image_str = encode_base64(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    raw_height, raw_width = inputs['image_grid_thw'][0, 1:].cpu().numpy().astype(int).tolist()
    merged_height, merged_width = _qwen_merged_heatmap_shape(inputs)
    inputs['pixel_values'].requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(**inputs)
    loss = _next_token_self_loss(outputs)
    grads = torch.autograd.grad(loss, inputs['pixel_values'], retain_graph=False)[0]
    grad_map = (
        grads.to(torch.float32)
        .detach()
        .norm(dim=-1)
        .cpu()
        .numpy()
        .reshape(raw_height, raw_width)
    )
    grad_map = grad_map[: merged_height * 2, : merged_width * 2]
    return block_reduce(grad_map, block_size=(2, 2), func=np.sum)


def pure_gradient_qwen2_5(image, prompt, general_prompt, model, processor):
    grad_map = _pixel_gradient_heatmap_qwen2_5(image, prompt, model, processor)
    general_grad_map = _pixel_gradient_heatmap_qwen2_5(image, general_prompt, model, processor)
    return grad_map / np.maximum(general_grad_map, QWEN_GRAD_EPS)
