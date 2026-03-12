from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session


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
