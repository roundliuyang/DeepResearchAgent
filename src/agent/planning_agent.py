"""Planning agent — pure LLM reasoning + plan.md management.

Responsibility boundary
-----------------------
The PlanningAgent has exactly **two** responsibilities:

1. **LLM reasoning**: given a task context (original task, available agents,
   execution history), produce a ``PlanDecision`` — the structured answer to
   "what should we do next?".
2. **plan.md management**: maintain a ``plan.md`` file in ``workdir/<session_id>/``
   that records every round's decisions, dispatches, results, and analysis.

It does **NOT**:
- Import or call the AgentBus.
- Dispatch sub-agents.
- Run a multi-round loop.

All dispatching, result collection, and loop control is the bus's job.
The bus calls this agent once per round via ACP (``acp(name="planning", ...)``)
and reads the returned ``PlanDecision`` to decide what to do next.

Call contract
-------------
The bus passes a dict-serialised context as ``task`` (the string).
The planner returns ``AgentResponse`` with ``extra.data["decision"]``
containing the serialised ``PlanDecision``.

To feed results back, the bus calls the planner again with an updated context
string that includes the previous execution history.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext


# ---------------------------------------------------------------------------
# LLM structured-output schema
# ---------------------------------------------------------------------------

class SubTaskDispatch(BaseModel):
    """One sub-task to dispatch to a named agent."""

    agent_name: str = Field(
        description="Exact name of the agent to call (must match an available agent)."
    )
    task: str = Field(
        description="The sub-task description to send to this agent."
    )
    files: List[str] = Field(
        default_factory=list,
        description="Optional file paths to attach.",
    )


class PlanDecision(BaseModel):
    """Structured output the LLM produces for each planning round.

    The AgentBus reads this to determine what to dispatch next.
    """

    thinking: str = Field(
        description="Chain-of-thought reasoning about the current state."
    )
    analysis: str = Field(
        description=(
            "Evaluation of the previous round's results. "
            "Leave empty on the first round."
        ),
    )
    plan_update: str = Field(
        description="Updated high-level description of the overall plan."
    )
    dispatches: List[SubTaskDispatch] = Field(
        default_factory=list,
        description=(
            "Sub-tasks to dispatch in this round.  All listed agents will "
            "run concurrently on the bus.  Must be empty when is_done=True."
        ),
    )
    is_done: bool = Field(
        default=False,
        description="Set True only when the entire original task is fully complete.",
    )
    final_result: Optional[str] = Field(
        default=None,
        description="Comprehensive final answer.  Required when is_done=True.",
    )


# ---------------------------------------------------------------------------
# plan.md data model
# ---------------------------------------------------------------------------

@dataclass
class PlanRound:
    """Execution record for one planning round."""

    number: int
    goal: str
    agents: List[str]
    delivery_mode: str              # "UNICAST" | "BROADCAST"
    subtasks: Dict[str, str]        # agent_name → task text
    results: Dict[str, Any]         # agent_name → {success, result, error}
    analysis: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class PlanFile:
    """Manages the ``plan.md`` file for a single planning session.

    Structure mirrors Cursor plan files::

        ---
        name: <title>
        overview: "<task description>"
        todos:
          - id: step-1-agent_name
            content: "agent_name: subtask description"
            status: completed | pending
        isProject: false
        ---

        # <title>

        ## Execution Flow
        ```mermaid
        graph LR
          subgraph execution [Execution Flow]
            s1[Step 1: agent] --> s2[Step 2: agent]
          end
        ```

        ## Execution Log
        ### Round N — <timestamp>
        ...

        ## Final Result
        ...
    """

    def __init__(self, path: str, task: str, task_id: str, session_id: str) -> None:
        self.path = path
        self.full_task = task
        self.task_title = (task[:150] + "...") if len(task) > 150 else task
        self.task_id = task_id
        self.session_id = session_id
        self.status = "running"
        self.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.rounds: List[PlanRound] = []
        self.final_result: Optional[str] = None

    # -- mutation ----------------------------------------------------------

    def add_round(self, round_: PlanRound) -> None:
        self.rounds.append(round_)

    def update_last_analysis(self, analysis: str) -> None:
        if self.rounds and analysis:
            self.rounds[-1].analysis = analysis

    def finalize(self, result: str, success: bool) -> None:
        self.status = "done" if success else "failed"
        self.final_result = result

    # -- persistence -------------------------------------------------------

    async def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        content = self._render()
        await asyncio.to_thread(self._write_sync, content)

    def _write_sync(self, content: str) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # -- context for LLM (plain text, no mermaid) --------------------------

    def execution_log_text(self) -> str:
        """Plain-text execution log passed to the LLM as context."""
        if not self.rounds:
            return "(no rounds completed yet)"
        lines: List[str] = []
        for r in self.rounds:
            lines.append(f"=== Round {r.number} — {r.timestamp} ===")
            lines.append(f"Goal: {r.goal}")
            lines.append(f"Dispatched ({r.delivery_mode}): {', '.join(r.agents)}")
            for agent in r.agents:
                lines.append(f"  {agent} subtask: {r.subtasks.get(agent, '')[:200]}")
            lines.append("Results:")
            for agent, res in r.results.items():
                ok = res.get("success", False)
                text = str(res.get("result") or res.get("error") or "")[:300]
                lines.append(f"  {'OK' if ok else 'FAIL'} {agent}: {text}")
            if r.analysis:
                lines.append(f"Analysis: {r.analysis[:300]}")
            lines.append("")
        return "\n".join(lines)

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _node_id(name: str) -> str:
        return name.replace("-", "_").replace(".", "_").replace(" ", "_")

    def _build_todos(self) -> List[Dict[str, str]]:
        """Build todo items from execution rounds for YAML frontmatter."""
        todos: List[Dict[str, str]] = []
        task_index = 0
        for r in self.rounds:
            for a in r.agents:
                task_index += 1
                st = r.subtasks.get(a, "")[:200]
                res = r.results.get(a, {})
                ok = res.get("success")
                if ok is True:
                    status = "completed"
                elif ok is False:
                    status = "completed"
                else:
                    status = "pending"
                todo_id = f"step-{task_index}-{self._node_id(a)}"
                todos.append({
                    "id": todo_id,
                    "content": f"{a}: {st}",
                    "status": status,
                })
        return todos

    @staticmethod
    def _yaml_escape(s: str) -> str:
        """Escape a string for use as a YAML double-quoted value."""
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')

    def _render_mermaid(self) -> List[str]:
        lines = ["```mermaid", "graph LR"]
        agent_rounds = [r for r in self.rounds if r.agents]

        if not agent_rounds:
            title = self.task_title.replace('"', "'")
            lines.append("  subgraph plan [Plan]")
            lines.append(f'    start(["{title}"])')
            if self.status in ("done", "failed"):
                lines.append(f'    start --> finish(["{self.status}"])')
            lines.append("  end")
            lines.append("```")
            return lines

        lines.append("  subgraph execution [Execution Flow]")
        task_index = 0
        prev_id = None
        for r in agent_rounds:
            for a in r.agents:
                task_index += 1
                a_id = f"s{task_index}"
                label = f"Step {task_index}: {a}"
                lines.append(f"    {a_id}[{label}]")
                if prev_id:
                    lines.append(f"    {prev_id} --> {a_id}")
                prev_id = a_id

        if self.status == "done" and prev_id:
            lines.append("    finish([Done])")
            lines.append(f"    {prev_id} --> finish")
        elif self.status == "failed" and prev_id:
            lines.append("    finish([Failed])")
            lines.append(f"    {prev_id} --> finish")

        lines.append("  end")
        lines.append("```")
        return lines

    def _render_round(self, r: PlanRound) -> List[str]:
        lines: List[str] = [f"### Round {r.number} — {r.timestamp}", ""]
        lines.append(f"> {r.goal}")
        lines.append("")
        if r.agents:
            lines.append(f"**Dispatched ({r.delivery_mode}):** {', '.join(f'`{a}`' for a in r.agents)}")
            lines.append("")
            for a in r.agents:
                st = r.subtasks.get(a, "")[:300]
                res = r.results.get(a, {})
                ok = res.get("success")
                if ok is True:
                    lines.append(f"- [x] **`{a}`**: {st}")
                    result_text = str(res.get("result") or "")[:200]
                    if result_text:
                        lines.append(f"  - Result: {result_text}")
                elif ok is False:
                    err = str(res.get("error") or "")[:200]
                    lines.append(f"- [x] ~~**`{a}`**: {st}~~ ❌")
                    if err:
                        lines.append(f"  - Error: {err}")
                else:
                    lines.append(f"- [ ] **`{a}`**: {st}")
            lines.append("")
        if r.analysis:
            lines += ["**Analysis:**", f"> {r.analysis[:300]}", ""]
        lines += ["---", ""]
        return lines

    def _render(self) -> str:
        """将当前 PlanFile 状态全量渲染为 plan.md 文件内容。

        输出结构（Cursor plan 格式）:
            ---
            name:          任务标题，截断至150字符（self.task_title）。
            overview:      原始任务完整描述（self.full_task），经 YAML 转义。
            todos:         由 self.rounds 通过 _build_todos() 自动构建。
              - id:        格式 "step-{序号}-{agent_name}"，如 "step-1-tool_calling"。
              - content:   "{agent_name}: {子任务描述[:200]}"。
              - status:    results 中有 success=True/False 时为 "completed"，否则 "pending"。
            isProject:     固定为 false（单任务计划，非多项目跟踪器）。
            ---
            # 标题
            ## Execution Flow    由 _render_mermaid() 生成的 Mermaid 流程图。
            ## Execution Log     由 _render_round() 生成的每轮执行详情。
            ## Final Result      仅在 finalize() 设置 self.final_result 后才出现。
        """
        todos = self._build_todos()
        esc = self._yaml_escape

        # --- YAML frontmatter (Cursor plan format) ---
        lines: List[str] = [
            "---",
            f"name: {self.task_title}",
            f'overview: "{esc(self.full_task)}"',
        ]
        if todos:
            lines.append("todos:")
            for t in todos:
                lines.append(f'  - id: {t["id"]}')
                lines.append(f'    content: "{esc(t["content"])}"')
                lines.append(f'    status: {t["status"]}')
        else:
            lines.append("todos: []")
        lines.append("isProject: false")
        lines.append("---")
        lines.append("")

        # --- Title ---
        lines.append(f"# {self.task_title}")
        lines.append("")

        # --- Execution Flow (Mermaid) ---
        lines.append("## Execution Flow")
        lines.append("")
        lines += self._render_mermaid()
        lines += ["", ""]

        # --- Execution Log ---
        lines += ["## Execution Log", ""]
        if not self.rounds:
            lines += ["*(planning in progress...)*", ""]
        else:
            for r in self.rounds:
                lines += self._render_round(r)

        # --- Final Result ---
        if self.final_result is not None:
            tag = "Completed" if self.status == "done" else "Failed"
            lines += [f"## Final Result — {tag}", "", self.final_result, ""]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

@AGENT.register_module(force=True)
class PlanningAgent(Agent):
    """Pure LLM planning agent.

    One LLM call per invocation.  Returns a ``PlanDecision`` to the caller
    (the AgentBus), which owns the multi-round loop and all dispatching.

    Also maintains a ``plan.md`` file that records every round.  The bus
    feeds results back by calling this agent again with an updated context,
    and the agent appends the new round to plan.md before returning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="planning")
    description: str = Field(
        default=(
            "Decomposes tasks and decides which sub-agents to call next. "
            "Returns a PlanDecision; the AgentBus drives the loop."
        ),
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    # The PlanFile is stored per-session.  The bus creates it on the first
    # call and passes it back via kwargs on subsequent calls.
    _plan_files: Dict[str, PlanFile] = {}

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        **kwargs,
    ):
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name or "planning",
            memory_name=memory_name,
            require_grad=require_grad,
            **kwargs,
        )
        self._plan_files: Dict[str, PlanFile] = {}

    async def initialize(self) -> None:
        await super().initialize()

    # ------------------------------------------------------------------
    # plan.md lifecycle (called by the bus via kwargs)
    # ------------------------------------------------------------------

    def get_or_create_plan_file(
        self,
        session_id: str,
        task_id: str,
        task: str,
    ) -> PlanFile:
        """Return the existing PlanFile for a session, or create one."""
        if session_id not in self._plan_files:
            plan_path = os.path.join(self.workdir, f"{session_id}.plan.md")
            self._plan_files[session_id] = PlanFile(
                path=plan_path,
                task=task,
                task_id=task_id,
                session_id=session_id,
            )
        return self._plan_files[session_id]

    def remove_plan_file(self, session_id: str) -> None:
        """Clean up in-memory plan state for a completed session."""
        self._plan_files.pop(session_id, None)

    # ------------------------------------------------------------------
    # Main call — one LLM round
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        ctx: Optional[SessionContext] = None,
        **kwargs,
    ) -> AgentResponse:
        """Execute one planning round.

        Expected kwargs (set by the bus):
            task_id (str):          top-level task identifier.
            round_number (int):     current round (1-based).
            max_rounds (int):       max rounds allowed.
            agent_contract (str):   markdown description of available agents.
            execution_history (str): plain-text log of all completed rounds.
            round_results (dict):   agent_name → {success, result, error}
                                    from the *previous* round (empty on round 1).

        Returns:
            AgentResponse with extra.data["decision"] = PlanDecision dict.
        """
        if ctx is None:
            ctx = kwargs.get("ctx") or SessionContext()

        task_id = kwargs.get("task_id", "task_unknown")
        round_number = kwargs.get("round_number", 1)
        max_rounds = kwargs.get("max_rounds", 10)
        agent_contract = kwargs.get("agent_contract", "")
        execution_history = kwargs.get("execution_history", "")
        round_results = kwargs.get("round_results", {})

        logger.info(
            f"| 🧠 PlanningAgent round {round_number}/{max_rounds} "
            f"(session={ctx.id})"
        )

        # ------------------------------------------------------------------
        # Update plan.md with results from the PREVIOUS round
        # ------------------------------------------------------------------
        plan_file = self.get_or_create_plan_file(ctx.id, task_id, task)

        if round_results and plan_file.rounds:
            plan_file.rounds[-1].results = round_results

        # ------------------------------------------------------------------
        # Build LLM messages via prompt_manager
        # ------------------------------------------------------------------
        history_text = execution_history if execution_history else "(no rounds completed yet)"

        messages = await prompt_manager.get_messages(
            prompt_name=self.prompt_name,
            system_modules={"agent_contract": agent_contract},
            agent_modules={
                "task": task,
                "round_number": str(round_number),
                "max_rounds": str(max_rounds),
                "execution_history": history_text,
            },
        )

        # ------------------------------------------------------------------
        # LLM call
        # ------------------------------------------------------------------
        try:
            llm_output = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=PlanDecision,    # 强制LLM输出PlanDecision结构
            )
            # PlanningAgent LLM → 输出 dispatches=[{agent_name:"tool_calling", task:""Execute the hello world skill. Call the appropriate tool that prints or returns 'Hello, World!' or a similar greeting message.""}]
            decision: PlanDecision = llm_output.extra.parsed_model
        except Exception as exc:
            logger.error(f"| PlanningAgent LLM error: {exc}", exc_info=True)
            decision = PlanDecision(
                thinking=f"LLM call failed: {exc}",
                analysis="",
                plan_update="Planning failed due to LLM error.",
                dispatches=[],
                is_done=True,
                final_result=f"Planning failed: {exc}",
            )

        logger.info(f"| 📋 Plan: {decision.plan_update[:200]}")

        # ------------------------------------------------------------------
        # Update plan.md with THIS round's decision
        # ------------------------------------------------------------------

        # Backfill analysis onto previous round
        if decision.analysis:
            plan_file.update_last_analysis(decision.analysis)

        if decision.is_done:
            plan_file.finalize(    # "done", 写入 final_result
                result=decision.final_result or "",
                success=True,
            )
            await plan_file.save()    # 第二次写盘，触发 _render() 全量重渲染
            logger.info("| PlanningAgent: task complete")
        elif decision.dispatches:
            delivery = "BROADCAST" if len(decision.dispatches) > 1 else "UNICAST"
            agent_names = [d.agent_name for d in decision.dispatches]
            subtasks = {d.agent_name: d.task for d in decision.dispatches}

            plan_round = PlanRound(
                number=round_number,
                goal=decision.plan_update,
                agents=agent_names,
                delivery_mode=delivery,
                subtasks=subtasks,
                results={},  # will be filled by the bus on the next round
            )
            plan_file.add_round(plan_round)
            await plan_file.save()
            logger.info(f"| PlanningAgent: dispatching {agent_names}")
        else:
            await plan_file.save()

        # ------------------------------------------------------------------
        # Return decision to the bus
        # ------------------------------------------------------------------
        '''
        decision.model_dump():
            {
                "thinking": "The task is to call the tool_calling agent to use a hello world skill. This is a single atomic sub-task. The only available agent is 'tool_calling', which can call tools. I will dispatch it now with a clear instruction to execute the hello world skill. No concurrency is possible here.",                  # LLM 推理过程
                "analysis": "",                                                                      # 上一轮结果分析（第1轮为空）
                "plan_update": "Dispatch the tool_calling agent to invoke the hello world skill.",   # 计划更新描述
                "dispatches": [                                                                      # 本轮要派发的子任务
                    {
                        "agent_name": "tool_calling",
                        "task": "Execute the skill that prints 'hello world'. Use any available tool or function to output the hello world message.",
                        "files": []
                    }
                ],
                "is_done": False,       # 是否全部完成
                "final_result": None    # 最终结果（is_done=True 时才有值）
            }
        plan_file.path:
            {
                "final_result": null,
                "full_task": "Call tool calling agent to use hello world skill.",
                "path": "workdir/bus/agent/planning_agent\\session_20260811-212303_d18ef434.plan.md",
                "rounds": [{
                    "number": 1,
                    "goal": "Dispatch the tool_calling agent to invoke the hello world skill.",
                    "agents": [
                        "tool_calling"
                    ],
                    "delivery": "any available tool or function to output the hello world message.",
                    "results": {},
                    "analysis": "",
                    "timestamp": "2026-08-11T13:23:30Z"
                }],
                "session_id": "session_20260811-212303_d18ef434",
                "status": "running",
                "task_id": "task_20260811-212303_afab6d48",
                "task_title": "Call tool calling agent to use hello world skill."
            }
        '''
        return AgentResponse(
            success=True,
            message=decision.plan_update,
            extra=AgentExtra(
                data={
                    "decision": decision.model_dump(),
                    "plan_path": plan_file.path,
                },
            ),
        )
