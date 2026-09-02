"""LiteLLM pre-call hook that bounds replayed image history for sparkDV4.

OpenAI-compatible clients resend their complete conversation on every turn.
This hook retains the newest eight image parts and replaces older image bytes
with a stable text marker while preserving all text and assistant analysis.
"""
from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

MAX_IMAGES = 8
OMITTED_IMAGE_MARKER = (
    "[Older image omitted by the gateway; subsequent text and assistant "
    "analysis remain available.]"
)
SUPPORTED_MODELS = frozenset(
    {
        "sparkDV4",
        "spark-active",
        "deepseek-v4-flash-vision-exp",
        "openai/deepseek-v4-flash-vision-exp",
    }
)
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_LOG = logging.getLogger("spark_litellm.image_history_pruner")


def _is_image_part(part: object) -> bool:
    return isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES


def prune_image_history(
    items: object,
    *,
    max_images: int = MAX_IMAGES,
    marker_type: str = "text",
) -> tuple[object, int]:
    """Return ``(items, pruned_count)`` without mutating the input.

    ``items`` is either Chat Completions ``messages`` or Responses API ``input``.
    Unknown shapes are returned unchanged. Image recency follows request order,
    so the last ``max_images`` image parts survive.
    """
    if not isinstance(items, list) or max_images < 0:
        return items, 0

    positions: list[tuple[int, int]] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            return items, 0
        content = item.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            return items, 0
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                return items, 0
            if _is_image_part(part):
                positions.append((item_index, part_index))

    pruned_count = max(0, len(positions) - max_images)
    if pruned_count == 0:
        return items, 0

    dropped = set(positions[:pruned_count])
    changed_items = list(items)
    changed_item_indexes = {item_index for item_index, _ in dropped}

    for item_index in changed_item_indexes:
        item = items[item_index]
        assert isinstance(item, dict)
        content = item.get("content")
        assert isinstance(content, list)
        changed_content: list[object] = []
        marker_inserted = False
        for part_index, part in enumerate(content):
            if (item_index, part_index) not in dropped:
                changed_content.append(part)
                continue
            if not marker_inserted:
                changed_content.append(
                    {"type": marker_type, "text": OMITTED_IMAGE_MARKER}
                )
                marker_inserted = True
        changed_items[item_index] = {**item, "content": changed_content}

    return changed_items, pruned_count


class ImageHistoryPruner(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: Any,
    ) -> dict[str, Any]:
        model = data.get("model")
        if model not in SUPPORTED_MODELS:
            return data

        if "messages" in data:
            key = "messages"
            marker_type = "text"
        elif "input" in data:
            key = "input"
            marker_type = "input_text"
        else:
            return data

        items, pruned_count = prune_image_history(
            data.get(key), max_images=MAX_IMAGES, marker_type=marker_type
        )
        if pruned_count == 0:
            return data

        _LOG.info(
            "pruned %d older image(s); retained=%d model=%s call_type=%s",
            pruned_count,
            MAX_IMAGES,
            model,
            call_type,
        )
        return {**data, key: items}


proxy_handler_instance = ImageHistoryPruner()
