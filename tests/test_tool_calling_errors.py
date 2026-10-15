"""Action exceptions must reach execution history so the model can recover."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.agent.tool_calling_agent as module
import src.agent.types as agent_types
from src.agent.types import ThinkOutput
from src.session import SessionContext


@pytest.mark.parametrize("failure", ["missing_argument", "execution", "skill", "parsing"])
def test_action_error_is_saved_and_next_step_can_recover(tmp_path, monkeypatch, failure):
    async def run():
        action_type = "skill" if failure == "skill" else "tool"
        first = ThinkOutput(
            thinking="test", evaluation_previous_goal="", memory="", next_goal="run",
            actions=[
                {"type": "tool", "name": "prepare", "args": "{}"},
                {"type": action_type, "name": "bash", "args": "{}"},
                {"type": "tool", "name": "done", "args": "{}"},
            ],
        )
        corrected = first.model_copy(update={"actions": [
            first.actions[1].model_copy(update={"args": '{"command": "ls"}'}),
        ]})
        model = AsyncMock(side_effect=[
            SimpleNamespace(extra=SimpleNamespace(parsed_model=first)),
            SimpleNamespace(extra=SimpleNamespace(parsed_model=corrected)),
        ])
        monkeypatch.setattr(module, "model_manager", model)
        events = []
        stored_events = []

        async def save_event(**kwargs):
            events.append(kwargs["data"])
            stored_events.append(SimpleNamespace(**kwargs))

        memory = SimpleNamespace(
            add_event=save_event,
            get_state=AsyncMock(return_value={
                "events": stored_events, "summaries": [], "insights": [],
            }),
        )
        monkeypatch.setattr(module, "memory_manager", memory)
        monkeypatch.setattr(agent_types, "memory_manager", memory)

        async def bash(command):
            return SimpleNamespace(message=f"executed {command}", success=True, extra=None)

        async def call(*, name, input, ctx):
            if name == "prepare":
                return SimpleNamespace(message="prepared", success=True, extra=None)
            if not input and failure in ("execution", "skill"):
                raise RuntimeError("service unavailable")
            return await bash(**input)

        caller = AsyncMock(side_effect=call)
        monkeypatch.setattr(module, "tcp", caller)
        monkeypatch.setattr(module, "scp", caller)
        if failure == "parsing":
            original_parse = module.parse_tool_args
            count = 0

            def parse(args):
                nonlocal count
                count += 1
                if count == 2:
                    raise ValueError("invalid action arguments")
                return original_parse(args)

            monkeypatch.setattr(module, "parse_tool_args", parse)

        agent = module.ToolCallingAgent(
            workdir=str(tmp_path), model_name="test", memory_name="test",
            use_memory=True, use_todo=False,
        )
        ctx = SessionContext()
        record = SimpleNamespace(tool=None)
        result = await agent._think_and_tool([], "task", 0, record=record, ctx=ctx)
        assert result["done"] is False
        actions = events[0]["actions"]
        assert actions[0]["output"] == "prepared"
        error = actions[1]
        assert error["name"] == "bash"
        assert error["args"] == "{}"
        assert error["success"] is False
        expected = {
            "missing_argument": "TypeError", "execution": "service unavailable",
            "skill": "service unavailable", "parsing": "invalid action arguments",
        }[failure]
        assert expected in error["output"]
        assert record.tool["actions"][1] == error
        assert all(c.kwargs["name"] != "done" for c in caller.await_args_list)

        # Use the production history renderer to expose the saved error.
        context = await agent._get_agent_context("run", ctx=ctx)
        feedback = [SimpleNamespace(content=context["agent_context"])]
        await agent._think_and_tool(feedback, "task", 1, ctx=ctx)
        assert expected in model.await_args.kwargs["messages"][0].content
        assert events[1]["actions"][0]["output"] == "executed ls"

    asyncio.run(run())
