#!/usr/bin/env python3
"""CPU-only contract tests for the LiteLLM image-history callback."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from typing import Any


class _CustomLogger:
    pass


litellm_module = types.ModuleType("litellm")
integrations_module = types.ModuleType("litellm.integrations")
custom_logger_module = types.ModuleType("litellm.integrations.custom_logger")
custom_logger_module.CustomLogger = _CustomLogger
sys.modules.setdefault("litellm", litellm_module)
sys.modules.setdefault("litellm.integrations", integrations_module)
sys.modules.setdefault("litellm.integrations.custom_logger", custom_logger_module)

MODULE_PATH = Path(__file__).resolve().parents[1] / "litellm" / "image_history_pruner.py"
SPEC = importlib.util.spec_from_file_location("image_history_pruner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PRUNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRUNER)


def image_part(value: str, *, part_type: str = "image_url") -> dict[str, Any]:
    if part_type == "input_image":
        return {"type": part_type, "image_url": value}
    return {"type": part_type, "image_url": {"url": value}}


def retained_image_values(items: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in items:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {
                "image_url",
                "input_image",
            }:
                continue
            image = part.get("image_url")
            values.append(image.get("url") if isinstance(image, dict) else image)
    return values


class ImageHistoryPrunerTests(unittest.TestCase):
    def test_eight_images_are_returned_by_identity(self) -> None:
        messages = [
            {"role": "user", "content": [image_part(f"image-{index}")]}
            for index in range(8)
        ]

        result, pruned = PRUNER.prune_image_history(messages)

        self.assertIs(result, messages)
        self.assertEqual(pruned, 0)

    def test_oldest_image_is_replaced_and_newest_eight_survive(self) -> None:
        messages = [
            {"role": "user", "content": [image_part(f"image-{index}")]}
            for index in range(9)
        ]
        original = copy.deepcopy(messages)

        result, pruned = PRUNER.prune_image_history(messages)

        self.assertEqual(pruned, 1)
        self.assertEqual(retained_image_values(result), [f"image-{i}" for i in range(1, 9)])
        self.assertEqual(messages, original)
        self.assertEqual(
            result[0]["content"],
            [{"type": "text", "text": PRUNER.OMITTED_IMAGE_MARKER}],
        )

    def test_text_order_and_assistant_analysis_are_preserved(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    image_part("old"),
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": "analysis already produced"},
            {
                "role": "user",
                "content": [image_part(f"new-{index}") for index in range(8)],
            },
        ]

        result, pruned = PRUNER.prune_image_history(messages)

        self.assertEqual(pruned, 1)
        self.assertEqual(
            result[0]["content"],
            [
                {"type": "text", "text": "before"},
                {"type": "text", "text": PRUNER.OMITTED_IMAGE_MARKER},
                {"type": "text", "text": "after"},
            ],
        )
        self.assertIs(result[1], messages[1])
        self.assertEqual(result[1]["content"], "analysis already produced")

    def test_multiple_old_images_in_one_message_use_one_stable_marker(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    image_part("duplicate"),
                    {"type": "text", "text": "between"},
                    image_part("duplicate"),
                ],
            },
            {
                "role": "user",
                "content": [image_part(f"new-{index}") for index in range(8)],
            },
        ]

        result, pruned = PRUNER.prune_image_history(messages)

        self.assertEqual(pruned, 2)
        self.assertEqual(
            result[0]["content"],
            [
                {"type": "text", "text": PRUNER.OMITTED_IMAGE_MARKER},
                {"type": "text", "text": "between"},
            ],
        )
        self.assertEqual(retained_image_values(result), [f"new-{i}" for i in range(8)])

    def test_unknown_shapes_are_unchanged(self) -> None:
        malformed_with_images = [
            {
                "role": "user",
                "content": [image_part(f"image-{index}") for index in range(9)]
                + ["not-a-content-part"],
            }
        ]
        for value in (
            None,
            "text",
            {"content": []},
            [None, {"content": "text"}],
            malformed_with_images,
        ):
            with self.subTest(value=value):
                result, pruned = PRUNER.prune_image_history(value)
                self.assertIs(result, value)
                self.assertEqual(pruned, 0)

    def test_responses_input_uses_input_text_marker(self) -> None:
        items = [
            {
                "role": "user",
                "content": [image_part(f"image-{index}", part_type="input_image")],
            }
            for index in range(9)
        ]
        data = {"model": "sparkDV4", "input": items}

        result = asyncio.run(
            PRUNER.proxy_handler_instance.async_pre_call_hook(None, None, data, "responses")
        )

        self.assertEqual(result["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(retained_image_values(result["input"]), [f"image-{i}" for i in range(1, 9)])

    def test_hook_guard_and_noop_preserve_request_identity(self) -> None:
        messages = [{"role": "user", "content": [image_part("one")]}]
        supported = {"model": "sparkDV4", "messages": messages}
        unsupported = {"model": "other", "messages": messages * 9}

        supported_result = asyncio.run(
            PRUNER.proxy_handler_instance.async_pre_call_hook(
                None, None, supported, "acompletion"
            )
        )
        unsupported_result = asyncio.run(
            PRUNER.proxy_handler_instance.async_pre_call_hook(
                None, None, unsupported, "acompletion"
            )
        )

        self.assertIs(supported_result, supported)
        self.assertIs(unsupported_result, unsupported)

    def test_log_contains_counts_but_not_image_payload(self) -> None:
        secret_payload = "data:image/png;base64,SECRET_IMAGE_BYTES"
        messages = [
            {"role": "user", "content": [image_part(secret_payload)]},
            *[
                {"role": "user", "content": [image_part(f"new-{index}")]}
                for index in range(8)
            ],
        ]
        data = {"model": "sparkDV4", "messages": messages}

        with self.assertLogs("spark_litellm.image_history_pruner", logging.INFO) as logs:
            asyncio.run(
                PRUNER.proxy_handler_instance.async_pre_call_hook(
                    None, None, data, "acompletion"
                )
            )

        rendered = "\n".join(logs.output)
        self.assertIn("pruned 1 older image", rendered)
        self.assertIn("retained=8", rendered)
        self.assertNotIn(secret_payload, rendered)
        self.assertNotIn("SECRET_IMAGE_BYTES", rendered)


if __name__ == "__main__":
    unittest.main()
