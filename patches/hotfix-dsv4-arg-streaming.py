#!/usr/bin/env python3
"""Bound DeepSeek V4 streaming tool-argument silence in vLLM.

The parser engine normally runs an argument converter only when a configured
structural character is present. DeepSeek V4 configures only ``>``. A long
string parameter such as CSS therefore emits no OpenAI ``function.arguments``
deltas between ``<style>`` and ``</style>``, even while the model is decoding.

This source-locked hotfix forces a converter attempt after at most 512 raw
argument characters without a structural trigger. The converter's existing
prefix-safety and schema checks still decide whether a JSON delta is safe to
emit. Structural triggers retain their stock behavior.

Usage inside the container:
  python3 hotfix-dsv4-arg-streaming.py
  python3 hotfix-dsv4-arg-streaming.py /path/to/vllm
  python3 hotfix-dsv4-arg-streaming.py --status [/path/to/vllm]
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
PARSER_ENGINE = "parser/engine/parser_engine.py"
MARK = "# [dsv4-arg-streaming-hotfix] bounded converter cadence"

CONSTANT_ANCHOR = "logger = init_logger(__name__)\n"
CONSTANT_NEW = (
    CONSTANT_ANCHOR
    + "\n"
    + MARK
    + "\n"
    + "_ARG_STREAM_MAX_PENDING_CHARS = 512\n"
)

SLOTS_OLD = '''        "string_keys",
        "streamed_json",
    )'''
SLOTS_NEW = '''        "string_keys",
        "streamed_json",
        "_arg_stream_pending_chars",
    )'''

INIT_OLD = '''        self.string_keys: set[str] | None = None
        self.streamed_json: str = ""'''
INIT_NEW = '''        self.string_keys: set[str] | None = None
        self.streamed_json: str = ""
        self._arg_stream_pending_chars: int = 0'''

APPEND_OLD = '''    def append_args(self, value: str) -> None:
        self._args_parts.append(value)
        self._args_joined = None'''
APPEND_NEW = '''    def append_args(self, value: str) -> None:
        self._args_parts.append(value)
        self._args_joined = None
        self._arg_stream_pending_chars += len(value)'''

GATE_OLD = '''        structural = self._arg_structural_chars
        if structural is not None and structural.isdisjoint(raw_delta):
            return None

        slot = self._tool_slots[idx]
        try:'''
GATE_NEW = '''        slot = self._tool_slots[idx]
        structural = self._arg_structural_chars
        if structural is not None and structural.isdisjoint(raw_delta):
            if slot._arg_stream_pending_chars < _ARG_STREAM_MAX_PENDING_CHARS:
                return None
        slot._arg_stream_pending_chars = 0

        try:'''

EDITS = (
    ("flush constant", CONSTANT_ANCHOR, CONSTANT_NEW),
    ("slot field", SLOTS_OLD, SLOTS_NEW),
    ("slot initialization", INIT_OLD, INIT_NEW),
    ("argument accounting", APPEND_OLD, APPEND_NEW),
    ("bounded conversion gate", GATE_OLD, GATE_NEW),
)


def _is_applied(source: str) -> bool:
    return MARK in source and all(new in source for _, _, new in EDITS[1:])


def _apply(path: Path) -> tuple[bool, str]:
    source = path.read_text(encoding="utf-8")
    if _is_applied(source):
        return False, "already applied"

    missing = [name for name, old, _ in EDITS if old not in source]
    if missing:
        raise RuntimeError("missing source anchors: " + ", ".join(missing))

    patched = source
    for _, old, new in EDITS:
        patched = patched.replace(old, new, 1)
    compile(patched, str(path), "exec")

    mode = stat.S_IMODE(path.stat().st_mode)
    tmp = path.with_name(path.name + ".dsv4-arg-streaming.tmp")
    try:
        tmp.write_text(patched, encoding="utf-8")
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return True, "applied"


def main() -> int:
    status = len(sys.argv) > 1 and sys.argv[1] == "--status"
    root_arg = sys.argv[2] if status and len(sys.argv) > 2 else None
    if not status and len(sys.argv) > 1:
        root_arg = sys.argv[1]
    root = Path(root_arg) if root_arg else DEFAULT_VLLM
    path = root / PARSER_ENGINE
    if not path.is_file():
        print(f"[FAIL] parser engine not found: {path}", file=sys.stderr)
        return 1

    if status:
        applied = _is_applied(path.read_text(encoding="utf-8"))
        print("DeepSeek argument stream cadence:", "APPLIED" if applied else "NOT APPLIED")
        return 0 if applied else 1

    try:
        changed, result = _apply(path)
    except (OSError, RuntimeError, SyntaxError) as exc:
        print(f"[FAIL] DeepSeek argument streaming hotfix: {exc}", file=sys.stderr)
        return 1
    print(f"[dsv4-arg-streaming-hotfix] {result}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
