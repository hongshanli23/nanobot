from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session
from nanobot.session.transcript import TranscriptItem
from nanobot.session.transcript import build_summary_record


def _mk_loop() -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)
    loop._TOOL_RESULT_MAX_CHARS = 500
    return loop


def test_save_turn_skips_multimodal_user_when_only_runtime_context() -> None:
    loop = _mk_loop()
    session = Session(key="test:runtime-only")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{"role": "user", "content": [{"type": "text", "text": runtime}]}],
        skip=0,
    )
    assert session.messages == []


def test_save_turn_keeps_image_placeholder_after_runtime_strip() -> None:
    loop = _mk_loop()
    session = Session(key="test:image")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": runtime},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }],
        skip=0,
    )
    assert session.messages[0]["content"] == [{"type": "text", "text": "[image]"}]
    assert session.messages[0]["type"] == "user"


def test_save_turn_externalizes_raw_tool_output(tmp_path) -> None:
    loop = _mk_loop()
    session = Session(key="test:tool", artifacts_dir=tmp_path)
    raw_output = "0123456789" * 80

    loop._save_turn(
        session,
        [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"foo.txt\"}"},
            }],
        }, {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "read_file",
            "content": raw_output,
        }],
        skip=0,
    )

    assert [item["type"] for item in session.messages] == ["assistant", "tool_call", "tool_result", "summary"]
    tool_result = session.messages[-2]
    assert tool_result["raw_path"]
    assert len(tool_result["content"]) < len(raw_output)
    assert tmp_path.joinpath("call_123.txt").read_text() == raw_output
    assert session.messages[-1]["content"].startswith("[Turn summary]")


def test_get_history_collapses_older_turns_to_summary() -> None:
    loop = _mk_loop()
    session = Session(key="test:summary-collapse")

    loop._save_turn(
        session,
        [
            {"role": "user", "content": "inspect foo.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"foo.py\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "line1\nline2"},
            {"role": "assistant", "content": "The file defines one helper."},
        ],
        skip=0,
    )
    for index in range(2, 5):
        loop._save_turn(
            session,
            [
                {"role": "user", "content": f"turn {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
            skip=0,
        )

    history = session.get_history(max_messages=20)

    assert history[0]["role"] == "assistant"
    assert history[0]["content"].startswith("[Turn summary]")
    assert "read_file" in history[0]["content"]
    assert "inspect foo.py" in history[0]["content"]
    assert all(message.get("role") != "tool" for message in history[:1])
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "turn 2"


def test_get_history_drops_orphan_tool_results() -> None:
    session = Session(key="test:orphan-tool")
    session.messages = [
        TranscriptItem(
            type="assistant",
            content="I checked the file.",
            turn_id="turn-1",
        ).to_record(),
        TranscriptItem(
            type="tool_result",
            content="unexpected output",
            turn_id="turn-1",
            call_id="missing-call",
            tool_name="read_file",
        ).to_record(),
        build_summary_record("[Turn summary] Assistant: I checked the file.", turn_id="turn-1"),
    ]

    history = session.get_history(max_messages=10)

    assert history == [{"role": "assistant", "content": "I checked the file."}]


def test_get_history_preserves_legacy_tool_linkage() -> None:
    session = Session(key="test:legacy-tool-linkage")
    session.messages = [
        {"role": "user", "content": "inspect foo.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_legacy",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"foo.py\"}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_legacy", "name": "read_file", "content": "line1\nline2"},
        {"role": "assistant", "content": "The file defines one helper."},
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "hi"},
    ]

    history = session.get_history(max_messages=4)

    assert history == [
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "hi"},
    ]


class _DummyProvider(LLMProvider):
    async def chat(self, *args, **kwargs):
        raise NotImplementedError

    def get_default_model(self) -> str:
        return "dummy-model"


@pytest.mark.asyncio
async def test_run_agent_loop_surfaces_provider_exception() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    loop._TOOL_RESULT_MAX_CHARS = 500
    loop.max_iterations = 40
    loop.provider = _DummyProvider()
    loop.provider.chat = AsyncMock(side_effect=RuntimeError("boom"))
    loop.tools = MagicMock()
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.model = "dummy-model"
    loop.temperature = 0.1
    loop.max_tokens = 4096
    loop.reasoning_effort = None
    loop.context = MagicMock()

    final_content, tools_used, messages = await loop._run_agent_loop([
        {"role": "user", "content": "hi"}
    ])

    assert final_content == "Error calling AI model: RuntimeError('boom')"
    assert tools_used == []
    assert messages == [{"role": "user", "content": "hi"}]
