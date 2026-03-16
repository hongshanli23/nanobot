"""Typed session transcript helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


TRANSCRIPT_ITEM_TYPE = "transcript_item"
SUMMARY_ITEM_TYPE = "summary"
USER_ITEM_TYPE = "user"
ASSISTANT_ITEM_TYPE = "assistant"
TOOL_CALL_ITEM_TYPE = "tool_call"
TOOL_RESULT_ITEM_TYPE = "tool_result"
DEFAULT_DETAILED_TURNS = 3


@dataclass(slots=True)
class TranscriptItem:
    """Minimal typed transcript entry persisted in session JSONL."""

    type: str
    content: Any = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    turn_id: str | None = None
    call_id: str | None = None
    tool_name: str | None = None
    arguments: str | None = None
    raw_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "_type": TRANSCRIPT_ITEM_TYPE,
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.turn_id:
            record["turn_id"] = self.turn_id
        if self.call_id:
            record["call_id"] = self.call_id
        if self.tool_name:
            record["tool_name"] = self.tool_name
        if self.arguments is not None:
            record["arguments"] = self.arguments
        if self.raw_path:
            record["raw_path"] = self.raw_path
        record.update(self.meta)
        return record


def is_typed_record(record: dict[str, Any]) -> bool:
    """Return True when the record is a structured transcript item."""
    if record.get("_type") == TRANSCRIPT_ITEM_TYPE:
        return True
    return record.get("type") in {
        USER_ITEM_TYPE,
        ASSISTANT_ITEM_TYPE,
        TOOL_CALL_ITEM_TYPE,
        TOOL_RESULT_ITEM_TYPE,
        SUMMARY_ITEM_TYPE,
    }


def build_tool_call_record(
    tool_call_id: str,
    tool_name: str,
    arguments: str,
    *,
    turn_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one structured tool-call record."""
    return TranscriptItem(
        type=TOOL_CALL_ITEM_TYPE,
        content="",
        timestamp=timestamp or datetime.now().isoformat(),
        turn_id=turn_id,
        call_id=tool_call_id,
        tool_name=tool_name,
        arguments=arguments,
    ).to_record()


def build_summary_record(
    content: str,
    *,
    turn_id: str | None = None,
    timestamp: str | None = None,
    tools_used: list[str] | None = None,
) -> dict[str, Any]:
    """Build one structured summary record for a completed turn."""
    return TranscriptItem(
        type=SUMMARY_ITEM_TYPE,
        content=content,
        timestamp=timestamp or datetime.now().isoformat(),
        turn_id=turn_id,
        meta={"tools_used": tools_used or []},
    ).to_record()


def compile_history(
    records: list[dict[str, Any]],
    *,
    max_messages: int | None = None,
    detailed_turns: int = DEFAULT_DETAILED_TURNS,
) -> list[dict[str, Any]]:
    """Compile transcript records into normalized provider-facing chat messages."""
    compiled_turns: list[list[dict[str, Any]]] = []
    turns = _group_records_by_turn(records)
    recent_turn_ids = {
        turn["turn_id"]
        for turn in turns[-detailed_turns:]
        if turn.get("turn_id") is not None
    }

    for turn in turns:
        summary = turn.get("summary")
        if (
            summary is not None
            and turn.get("turn_id") not in recent_turn_ids
        ):
            compiled_turns.append(_compile_summary_turn(summary))
            continue

        compiled_turns.append(_compile_detailed_turn(turn["records"]))

    compiled_turns = [turn for turn in compiled_turns if turn]
    if max_messages is None or max_messages <= 0:
        return [message for turn in compiled_turns for message in turn]

    return _limit_compiled_turns(compiled_turns, max_messages=max_messages)


def _group_records_by_turn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group transcript records so history can be compiled one turn at a time."""
    turns: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None

    for index, record in enumerate(records):
        if not is_typed_record(record):
            turns.append({"turn_id": f"legacy-{index}", "records": [record], "summary": None})
            current_turn = None
            continue

        turn_id = record.get("turn_id") or f"legacy-{index}"
        if current_turn is None or current_turn["turn_id"] != turn_id:
            current_turn = {"turn_id": turn_id, "records": [], "summary": None}
            turns.append(current_turn)

        current_turn["records"].append(record)
        if record.get("type") == SUMMARY_ITEM_TYPE:
            current_turn["summary"] = record

    return turns


def _compile_summary_turn(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile one summary-only turn for older context."""
    content = record.get("content", "")
    if not content:
        return []
    return [{"role": "assistant", "content": content}]


def _compile_detailed_turn(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compile one turn and repair malformed tool fragments when possible."""
    compiled: list[dict[str, Any]] = []
    seen_tool_calls: set[str] = set()

    for record in records:
        if not is_typed_record(record):
            entry: dict[str, Any] = {
                "role": record["role"],
                "content": record.get("content", ""),
            }
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in record:
                    entry[key] = record[key]
            compiled.append(entry)
            continue

        item_type = record.get("type")
        if item_type == USER_ITEM_TYPE:
            compiled.append({"role": "user", "content": record.get("content", "")})
            continue

        if item_type == ASSISTANT_ITEM_TYPE:
            compiled.append({"role": "assistant", "content": record.get("content")})
            continue

        if item_type == TOOL_CALL_ITEM_TYPE:
            # Tool calls are stored separately, then stitched back onto the
            # assistant message shape expected by providers at prompt time.
            if compiled and compiled[-1].get("role") == "assistant":
                assistant_msg = compiled[-1]
            else:
                assistant_msg = {"role": "assistant", "content": None}
                compiled.append(assistant_msg)
            call_id = record.get("call_id")
            if call_id:
                seen_tool_calls.add(call_id)
            assistant_msg.setdefault("tool_calls", []).append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": record.get("tool_name"),
                    "arguments": record.get("arguments") or "{}",
                },
            })
            continue

        if item_type == TOOL_RESULT_ITEM_TYPE:
            call_id = record.get("call_id")
            if call_id and call_id not in seen_tool_calls:
                # Drop orphan tool results instead of replaying invalid history
                # back into the model on the next turn.
                continue
            compiled.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": record.get("tool_name"),
                "content": record.get("content", ""),
            })
            continue

    return compiled


def _limit_compiled_turns(
    compiled_turns: list[list[dict[str, Any]]],
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    """Keep whole recent turns when possible so slicing does not orphan tools."""
    selected: list[list[dict[str, Any]]] = []
    total = 0

    for turn in reversed(compiled_turns):
        turn_size = len(turn)
        if selected and total + turn_size > max_messages:
            break
        if not selected and turn_size > max_messages:
            return _trim_turn_to_boundary(turn[-max_messages:])
        selected.append(turn)
        total += turn_size

    selected.reverse()
    return [message for turn in selected for message in turn]


def _trim_turn_to_boundary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim a partial turn without starting from a tool fragment."""
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            return messages[index:]
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return messages[index:]
    return []


def render_for_memory(record: dict[str, Any]) -> tuple[str, Any, list[str]]:
    """Return a normalized view suitable for memory consolidation prompts."""
    if not is_typed_record(record):
        tools_used = record.get("tools_used") or []
        return record.get("role", "unknown"), record.get("content"), tools_used

    item_type = record.get("type")
    if item_type == USER_ITEM_TYPE:
        return "user", record.get("content"), []
    if item_type == ASSISTANT_ITEM_TYPE:
        return "assistant", record.get("content"), []
    if item_type == TOOL_RESULT_ITEM_TYPE:
        tool_name = record.get("tool_name")
        return "tool", record.get("content"), [tool_name] if tool_name else []
    if item_type == SUMMARY_ITEM_TYPE:
        # Turn summaries are for prompt reconstruction only; memory
        # consolidation should read the original detailed records.
        return "internal", None, []
    return "internal", None, []


def stringify_tool_result(result: Any) -> str:
    """Normalize arbitrary tool output into one storable string."""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)


def read_raw_output(path_value: str | None) -> str | None:
    """Best-effort load of externally stored tool output."""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
