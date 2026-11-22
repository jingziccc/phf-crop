from __future__ import annotations

from typing import Any


def decode_blip_short_answer(processor: Any, generate_ids: Any) -> str:
    text = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if "Question:" in text:
        text = text.rsplit("Question:", 1)[0]
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[-1]
    return text.strip()
