"""Regression: evidence beyond the history cutoff must reach the planner model."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.agent.planning_agent as planning
from src.prompt.template.planning import PlanningAgentMessagePrompt, PlanningSystemPrompt
from src.session import SessionContext


@pytest.mark.parametrize("results", [
    {},
    {"tool_calling": {
        "success": True,
        "result": "command output\n" * 100 + '\n页数: 2\n标题: "最终证据"',
        "error": None,
    }},
    {"tool_calling": {"success": False, "result": "", "error": "permission denied"}},
])
def test_latest_results_reach_model_without_history_truncation(tmp_path, monkeypatch, results):
    async def run():
        system_prompt = PlanningSystemPrompt()
        agent_prompt = PlanningAgentMessagePrompt()

        async def render_messages(*, prompt_name, system_modules, agent_modules):
            return [
                await system_prompt.get_message(system_modules),
                await agent_prompt.get_message(agent_modules),
            ]

        decision = planning.PlanDecision(
            thinking="test", analysis="", plan_update="test complete",
            is_done=True, final_result="test result",
        )
        model = AsyncMock(return_value=SimpleNamespace(extra=SimpleNamespace(parsed_model=decision)))
        monkeypatch.setattr(planning, "model_manager", model)
        monkeypatch.setattr(planning, "prompt_manager", SimpleNamespace(get_messages=render_messages))
        agent = planning.PlanningAgent(workdir=str(tmp_path), model_name="test")
        # Simulate the Bus's abbreviated history, which lacks the evidence at the end.
        summary = str(results.get("tool_calling", {}).get("result", ""))[:300]
        await agent(
            task="返回页数和标题", ctx=SessionContext(), task_id="test",
            execution_history=summary, round_results=results,
        )
        model.assert_awaited_once()
        message = model.await_args.kwargs["messages"][1]
        text = message.content[0].text
        payload = text.split("<latest_round_results>", 1)[1].split("</latest_round_results>", 1)[0]
        assert json.loads(payload) == results
        assert "返回页数和标题" in text
        if results.get("tool_calling", {}).get("success"):
            assert "最终证据" not in summary
            assert "最终证据" in text
        assert list(tmp_path.glob("*.plan.md")), "Plan persistence should still work"

    asyncio.run(run())
