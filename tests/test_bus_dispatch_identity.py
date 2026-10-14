"""Exercise real Bus + Planner bookkeeping with mocked model/agent execution."""
import asyncio
import copy
import importlib
import json
from types import SimpleNamespace

from src.agent.planning_agent import PlanningAgent, PlanDecision
import src.agent.planning_agent as planning
from src.agent.types import AgentResponse
from src.interaction import AgentBus
from src.session import SessionContext
from src.task import Task
from src.task.dispatch import make_dispatch_id

bus_module = importlib.import_module('src.interaction.bus')


def test_duplicate_agents_keep_results_and_plan_entries(tmp_path, monkeypatch):
    async def run():
        planner = PlanningAgent(workdir=str(tmp_path), model_name='test')
        bus = AgentBus()
        bus._known_agents = {'planning': 'planner', 'tool_calling': 'executor'}
        calls = []
        snapshots = []
        dispatched_messages = []
        barrier = asyncio.Event()
        arrivals = 0
        round_number = 0

        async def get_messages(**kwargs):
            snapshots.append(json.loads(kwargs['agent_modules']['latest_round_results']))
            return []

        async def model(**kwargs):
            nonlocal round_number
            round_number += 1
            if round_number == 1:
                tasks = [('tool_calling', 'A'), ('invalid-agent', 'skip'),
                         ('tool_calling', 'B'), ('tool_calling', 'C')]
            elif round_number == 2:
                # All invalid: previous results must not contaminate this PlanRound.
                tasks = [('invalid-agent', 'skip again')]
            elif round_number == 3:
                tasks = [('tool_calling', 'retry B')]
            else:
                tasks = []
            decision = PlanDecision(
                thinking='test', analysis='', plan_update='test plan',
                dispatches=[{'agent_name': a, 'task': t} for a, t in tasks],
                is_done=not tasks, final_result='complete' if not tasks else None,
            )
            return SimpleNamespace(extra=SimpleNamespace(parsed_model=decision))

        async def acp(name, input, ctx=None, **kwargs):
            nonlocal arrivals
            if name == 'planning':
                return await planner(**input, ctx=ctx, **kwargs)
            task = input['task']
            calls.append(task)
            if task in ('A', 'B', 'C'):
                arrivals += 1
                if arrivals == 3:
                    barrier.set()
                # Requires the Bus to actually start all three calls concurrently.
                await asyncio.wait_for(barrier.wait(), timeout=2)
                if task == 'B':
                    raise RuntimeError('B failed')
                if task == 'A':
                    await asyncio.sleep(0.01)  # Finish out of dispatch order.
            return AgentResponse(success=True, message='result ' + task)

        original_call = bus._call_agent
        async def capture_call(name, message, **kwargs):
            reply = await original_call(name, message, **kwargs)
            assert reply.payload['dispatch_id'] == message.payload['dispatch_id']
            dispatched_messages.append(copy.deepcopy(message))
            return reply

        monkeypatch.setattr(bus, '_call_agent', capture_call)
        monkeypatch.setattr(bus_module, 'acp', acp)
        monkeypatch.setattr(planning, 'model_manager', model)
        monkeypatch.setattr(planning, 'prompt_manager', SimpleNamespace(get_messages=get_messages))
        ctx = SessionContext()
        task = Task(content='check files', session_id=ctx.id)
        try:
            response = await asyncio.wait_for(bus.submit(task, session_ctx=ctx, max_rounds=5), 5)
        finally:
            await bus.shutdown()
        assert response.payload['success'] is True
        assert sorted(calls) == ['A', 'B', 'C', 'retry B']
        ids = [make_dispatch_id(task.id, 1, i) for i in (1, 3, 4)]
        first = snapshots[1]
        assert list(first) == ids  # Filtering did not renumber surviving dispatches.
        assert [first[k]['task'] for k in ids] == ['A', 'B', 'C']
        assert first[ids[0]]['result'] == 'result A'
        assert first[ids[1]]['success'] is False
        assert first[ids[1]]['error'] == 'B failed'
        assert first[ids[2]]['result'] == 'result C'
        assert snapshots[2] == first  # Invalid-only round preserves prior results.
        retry_id = make_dispatch_id(task.id, 3, 1)
        assert list(snapshots[3]) == [retry_id]
        assert snapshots[3][retry_id]['result'] == 'result retry B'
        plan = planner._plan_files[ctx.id]
        assert plan.rounds[0].results == first
        assert plan.rounds[1].results == {}  # No stale result backfill.
        assert plan.rounds[2].results == snapshots[3]
        todos = plan._build_todos()
        assert len({t['id'] for t in todos}) == 6
        assert [t['content'] for t in todos] == [
            'tool_calling: A', 'invalid-agent: skip', 'tool_calling: B',
            'tool_calling: C', 'invalid-agent: skip again', 'tool_calling: retry B',
        ]
        assert [t['status'] for t in todos] == [
            'completed', 'pending', 'completed', 'completed', 'pending', 'completed',
        ]
        assert len({t['dispatch_id'] for t in todos}) == 6
        rendered = plan._render_round(plan.rounds[0])
        assert '- [x] **`tool_calling`**: A' in rendered
        assert '- [x] ~~**`tool_calling`**: B~~ ❌' in rendered
        assert '- [x] **`tool_calling`**: C' in rendered
        saved = next(tmp_path.glob('*.plan.md')).read_text()
        for text in ('result A', 'B failed', 'result C', retry_id):
            assert text in saved
        assert len(dispatched_messages) == 4
    asyncio.run(run())
