#!/usr/bin/env python3
"""CPU source-lock tests and an optional live-vLLM parser regression.

Run recipe integrity tests on the host:
  python3 scripts/test-dsv4-arg-streaming.py

Run the parser contract inside the patched vLLM container:
  python3 scripts/test-dsv4-arg-streaming.py --live
"""
from __future__ import annotations

import importlib.util
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-arg-streaming.py"

STOCK_ENGINE = '''import json

logger = init_logger(__name__)


class ToolCallSlot:
    __slots__ = (
        "id",
        "name",
        "_args_parts",
        "_args_joined",
        "name_sent",
        "string_keys",
        "streamed_json",
    )

    def __init__(self) -> None:
        self.id: str = ""
        self.name: str = ""
        self._args_parts: list[str] = []
        self._args_joined: str | None = ""
        self.name_sent: bool = False
        self.string_keys: set[str] | None = None
        self.streamed_json: str = ""

    def append_args(self, value: str) -> None:
        self._args_parts.append(value)
        self._args_joined = None


class ParserEngine:
    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        converter = self._arg_converter
        if converter is None:
            return raw_delta

        if not self._stream_arg_deltas:
            return None

        structural = self._arg_structural_chars
        if structural is not None and structural.isdisjoint(raw_delta):
            return None

        slot = self._tool_slots[idx]
        try:
            current_json = converter(slot.args, True)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
'''


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("hotfix_dsv4_arg_streaming", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArgStreamingHotfixTest(unittest.TestCase):
    def setUp(self):
        self.hotfix = _load_hotfix()

    def _tree(self, tmp: str) -> Path:
        root = Path(tmp) / "vllm"
        path = root / self.hotfix.PARSER_ENGINE
        path.parent.mkdir(parents=True)
        path.write_text(STOCK_ENGINE, encoding="utf-8")
        path.chmod(0o640)
        return root

    def test_apply_is_atomic_idempotent_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            path = root / self.hotfix.PARSER_ENGINE
            changed, result = self.hotfix._apply(path)
            self.assertTrue(changed)
            self.assertEqual(result, "applied")
            first = path.read_bytes()
            compile(first, str(path), "exec")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
            self.assertTrue(self.hotfix._is_applied(first.decode()))

            changed, result = self.hotfix._apply(path)
            self.assertFalse(changed)
            self.assertEqual(result, "already applied")
            self.assertEqual(path.read_bytes(), first)

    def test_patch_adds_bounded_pending_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            path = root / self.hotfix.PARSER_ENGINE
            self.hotfix._apply(path)
            source = path.read_text(encoding="utf-8")
            self.assertIn("_ARG_STREAM_MAX_PENDING_CHARS = 512", source)
            self.assertIn("self._arg_stream_pending_chars += len(value)", source)
            self.assertIn(
                "slot._arg_stream_pending_chars < _ARG_STREAM_MAX_PENDING_CHARS",
                source,
            )
            self.assertIn("slot._arg_stream_pending_chars = 0", source)

    def test_missing_anchor_refuses_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            path = root / self.hotfix.PARSER_ENGINE
            original = path.read_text(encoding="utf-8").replace(
                "structural.isdisjoint(raw_delta)", "structural.missing(raw_delta)"
            )
            path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "bounded conversion gate"):
                self.hotfix._apply(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


def live_probe() -> int:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
    )
    from vllm.tool_parsers.deepseekv4_engine_tool_parser import (
        DeepSeekV4EngineToolParser,
    )

    tc_start = "<｜DSML｜tool_calls>"
    tc_end = "</｜DSML｜tool_calls>"
    invoke_start = '<｜DSML｜invoke name="'
    invoke_end = "</｜DSML｜invoke>"
    param_start = '<｜DSML｜parameter name="'
    param_end = "</｜DSML｜parameter>"
    html_prefix = '''<!DOCTYPE html>
<html lang="en">
<head>
<style>
/* design tokens */
'''
    css = "".join(
        f".card-{i} {{ color: white; background: black; padding: {i % 32}px; }}\n"
        for i in range(1000)
    )
    html_suffix = "</style>\n</head>\n<body>Orbit</body>\n</html>"
    content = html_prefix + css + html_suffix
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    )
    full_text = (
        f'{tc_start}\n{invoke_start}write">\n'
        f'{param_start}path" string="true">/private/tmp/orbit-compute.html{param_end}\n'
        f'{param_start}content" string="true">{content}{param_end}\n'
        f'{invoke_end}\n{tc_end}'
    )

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    parser = DeepSeekV4EngineToolParser(tokenizer, tools=[tool])
    request = ChatCompletionRequest(model="m", messages=[], tools=[tool], stream=True)
    previous = ""
    events: list[tuple[int, str]] = []
    for start in range(0, len(full_text), 4):
        delta_text = full_text[start : start + 4]
        current = previous + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous, current, delta_text, [], [], [1], request
        )
        previous = current
        if delta is None:
            continue
        for call in delta.tool_calls or []:
            if call.function and call.function.arguments is not None:
                events.append((start + len(delta_text), call.function.arguments))

    parsed = json.loads("".join(chunk for _, chunk in events))
    css_start = full_text.index(css)
    css_end = css_start + len(css)
    css_positions = [position for position, _ in events if css_start <= position <= css_end]
    gaps = [
        right - left
        for left, right in zip(
            [css_start, *css_positions], [*css_positions, css_end], strict=True
        )
    ]
    max_gap = max(gaps)
    if parsed["content"] != content:
        raise AssertionError("streamed argument JSON did not round-trip")
    if len(css_positions) < 100:
        raise AssertionError(f"only {len(css_positions)} argument deltas inside CSS")
    if max_gap > 640:
        raise AssertionError(f"maximum silent parser span is {max_gap} characters")
    print(
        json.dumps(
            {
                "content_chars": len(content),
                "css_argument_deltas": len(css_positions),
                "max_silent_source_chars": max_gap,
                "round_trip_ok": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    if "--live" in sys.argv:
        raise SystemExit(live_probe())
    unittest.main()
