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
    """大模型在每一轮规划中生成的结构化决策。

    AgentBus 读取 is_done 判断是否完成，否则根据 dispatches 分派子任务。
    thinking 说明当前决策依据，analysis 评价上一轮结果，plan_update 概括计划。

    第 1 轮示例：分派任务。
        {
            "thinking": "任务需要执行 hello world 技能，可交给 tool_calling。",
            "analysis": "",
            "plan_update": "调用工具智能体生成问候语。",
            "dispatches": [{
                "agent_name": "tool_calling",
                "task": "执行 hello world 技能并返回问候语。",
                "files": [],
            }],
            "is_done": False,
            "final_result": None,
        }
    执行结果：总线调用 tool_calling，收集结果后进入下一轮规划。

    第 2 轮示例：提交最终结果。
        {
            "thinking": "所需问候语已生成，可以结束任务。",
            "analysis": "上一轮执行成功，返回了有效的问候语。",
            "plan_update": "所有子任务已完成。",
            "dispatches": [],
            "is_done": True,
            "final_result": "技能执行成功：Hey there, World! 👋 Welcome aboard!",
        }
    执行结果：总线交付最终结果并退出循环，不再分派子任务。

    注意：Field 的 description 会作为结构化输出说明提供给模型。
    “完成时分派列表为空、最终结果必填”是描述中的要求，当前类未定义
    跨字段校验器来强制检查这两个条件。
    """

    # 当前决策的依据；总线不会根据这段文字判断任务是否完成。
    thinking: str = Field(
        description="Chain-of-thought reasoning about the current state."
    )
    # 回顾上一轮子任务的执行情况，例如成功、失败或仍缺少哪些结果；首轮为空。
    analysis: str = Field(
        description=(
            "Evaluation of the previous round's results. "
            "Leave empty on the first round."
        ),
    )
    # 本轮更新后的整体计划概述，用于日志和计划记录。
    # 总线在完成分支中还会将其作为 final_result 为空时的备用结果。
    plan_update: str = Field(
        description="Updated high-level description of the overall plan."
    )
    # 本轮要执行的子任务列表；每项包含目标 Agent 名称、任务描述和附件路径。
    # default_factory=list 为每个决策创建独立的空列表，避免共享可变默认值。
    dispatches: List[SubTaskDispatch] = Field(
        default_factory=list,
        description=(
            "Sub-tasks to dispatch in this round.  All listed agents will "
            "run concurrently on the bus.  Must be empty when is_done=True."
        ),
    )
    # 整个原始任务的完成标记，不是某个子任务的成功标记。
    # 默认 False；总线为 True 时交付结果，为 False 时继续检查分派列表。
    is_done: bool = Field(
        default=False,
        description="Set True only when the entire original task is fully complete.",
    )
    # 面向用户的最终答复；规划过程中可为 None，完成时应汇总实际执行结果。
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
        """将当前计划的完整内容渲染为 Markdown，并保存到 self.path。

        首次保存时创建文件，后续保存覆盖原有内容。等待写盘完成后返回，
        目录创建、渲染或写入失败时，异常向调用方传播。
        """
        # 确保计划文件所在目录存在；exist_ok=True 表示目录已存在时不报错。
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # 将内存中的任务信息、各轮计划与结果、最终答复渲染成完整 Markdown。
        content = self._render()
        # 在线程中执行同步文件写入，避免写盘操作阻塞事件循环；await 等待写入完成。
        # _write_sync() 使用 "w" 模式：文件不存在则创建，存在则覆盖全部内容。
        await asyncio.to_thread(self._write_sync, content)

    def _write_sync(self, content: str) -> None:
        """将完整的计划文本同步写入 self.path，由 save() 在线程中调用。

        Args:
            content: _render() 生成的完整 Markdown 内容。

        文件不存在时创建，存在时覆盖原有内容；写入失败时异常向调用方传播。
        """
        # 使用 UTF-8 保存中文及特殊字符；"w" 模式会清空已有文件，再写入新内容。
        # with 在代码块结束或发生异常时自动关闭文件，fh 是打开的文件对象。
        with open(self.path, "w", encoding="utf-8") as fh:
            # 写入全部计划文本，而不是在旧内容末尾追加。
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
            name:          任务标题，超过150字符时取前150字符并追加省略号。
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
        # 根据各轮子任务及其执行结果生成待办项，包含 id、content 和 status。
        todos = self._build_todos()
        # 保存 YAML 转义函数的引用，供下方处理描述文本中的引号、换行等字符。
        esc = self._yaml_escape

        # 1. 组装文件头部的 YAML 元数据，以 --- 开始，记录任务名称和完整描述。
        # lines 中每个元素是一段文本，最后统一用换行符连接。
        lines: List[str] = [
            "---",
            f"name: {self.task_title}",
            f'overview: "{esc(self.full_task)}"',
        ]
        if todos:
            # 每个子任务输出一个 YAML 列表项；缩进表示字段属于同一个待办项。
            lines.append("todos:")
            for t in todos:
                lines.append(f'  - id: {t["id"]}')
                lines.append(f'    content: "{esc(t["content"])}"')
                lines.append(f'    status: {t["status"]}')
        else:
            # 尚无子任务时显式输出空列表，保持文件头结构完整。
            lines.append("todos: []")
        # 标记为非项目计划，并用 --- 结束 YAML 文件头；空行分隔正文。
        lines.append("isProject: false")
        lines.append("---")
        lines.append("")

        # 2. 输出 Markdown 一级标题，使用任务标题。
        lines.append(f"# {self.task_title}")
        lines.append("")

        # 3. 输出执行流程章节，辅助方法生成 Mermaid 代码块及流程节点。
        lines.append("## Execution Flow")
        lines.append("")
        # += 将辅助方法返回的多行文本逐项加入 lines。
        lines += self._render_mermaid()
        lines += ["", ""]

        # 4. 输出执行日志，按 rounds 中的记录顺序展示各轮任务与结果。
        lines += ["## Execution Log", ""]
        if not self.rounds:
            # 尚未添加轮次记录时显示“规划进行中”的占位提示。
            lines += ["*(planning in progress...)*", ""]
        else:
            for r in self.rounds:
                # 将该轮目标、分派对象、子任务、执行结果和分析转换为 Markdown。
                lines += self._render_round(r)

        # 5. 已设置最终结果时输出总结章节；None 表示尚未提供最终结果。
        # 空字符串也满足此条件，此时仍会生成章节标题。
        if self.final_result is not None:
            # done 显示 Completed，其他状态在此分支中显示 Failed。
            tag = "Completed" if self.status == "done" else "Failed"
            lines += [f"## Final Result — {tag}", "", self.final_result, ""]

        # 将各段文本拼接为完整 Markdown 字符串；本方法只渲染，写盘由 save() 完成。
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

@AGENT.register_module(force=True)
class PlanningAgent(Agent):
    """基于大模型的规划智能体，每次调用执行一轮规划。

    每轮调用一次模型，根据任务、可用智能体和执行历史生成 PlanDecision，
    并将其封装在 AgentResponse 中返回。AgentBus 负责多轮循环、执行子任务
    和收集结果，本类负责决定下一步分派哪些任务或是否结束。

    同时维护计划文件：总线在下一次调用时传回上一轮执行结果，本类将其
    回填到对应记录，再根据当前决策追加新轮次或写入最终结果并保存。
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
    # 计划文件生命周期：由本智能体的 __call__() 根据会话和任务信息管理
    # ------------------------------------------------------------------

    def get_or_create_plan_file(
        self,
        session_id: str,
        task_id: str,
        task: str,
    ) -> PlanFile:
        """获取当前会话的计划对象；尚未缓存时创建并保存到内存中。

        同一会话的多轮规划复用同一个 PlanFile，以保留之前的轮次和结果。
        此方法只创建内存对象、确定文件路径；实际文件由后续调用
        await plan_file.save() 时创建或更新，不会在此读取已有磁盘文件。

        Args:
            session_id: 会话 ID，用作缓存键及文件名的前缀。
            task_id: 顶层任务 ID，仅在首次创建计划对象时保存。
            task: 原始任务文本，用于首次创建对象时设置标题和任务描述。

        Returns:
            该会话对应的 PlanFile。已缓存时直接返回原对象，
            不会用本次传入的 task_id 或 task 覆盖原有内容。
        """
        # 以 session_id 判断是否为本会话首次创建计划对象。
        if session_id not in self._plan_files:
            # 文件名为“会话 ID.plan.md”，存放在智能体配置的 workdir 下。
            # 若 workdir 是相对路径，则相对于进程运行时的工作目录解析。
            # 例如从 examples 目录运行，workdir="workdir/bus/agent/planning_agent"：
            # examples/workdir/bus/agent/planning_agent/
            #     session_20260924-112903_e1f66418.plan.md
            plan_path = os.path.join(self.workdir, f"{session_id}.plan.md")
            # 创建内存中的计划状态：保存任务信息，初始轮次为空、状态为 running。
            # 后续规划会向该对象添加轮次、回填结果，再通过 save() 写入磁盘。
            self._plan_files[session_id] = PlanFile(
                path=plan_path,
                task=task,
                task_id=task_id,
                session_id=session_id,
            )
        # 首轮返回新建对象，后续轮次返回同一会话已缓存的对象。
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
        """执行一轮规划，更新计划文件，并向总线返回结构化决策。

        调用链：AgentBus -> acp(name="planning", ...) -> AgentContextManager
        -> 当前实例的 __call__()。本次返回后，由总线读取决策并执行下一步。

        PlanFile 内容填充示例（第 1 轮分派，第 2 轮确认完成）：
            1. 首次创建计划对象：get_or_create_plan_file() 内调用 PlanFile.__init__()，
               保存任务标题和原始任务描述，供渲染文件标题、name 和 overview。
            2. 第 1 轮生成分派决策：构建 PlanRound，再调用 add_round()，
               填入 Round 1 的计划说明、目标 Agent 和子任务。
            3. 第 2 轮接收上轮结果：plan_file.rounds[-1].results = round_results，
               填入 Round 1 的执行结果，例如 Result 中的问候语。
            4. 第 2 轮生成结果分析：update_last_analysis(decision.analysis)，
               填入 Round 1 的 Analysis 执行评价。
            5. 第 2 轮确认完成：finalize(result=..., success=True)，
               填入最终结果，并将计划状态标记为 done。
            上述操作先修改内存对象；save() 调用 _render() 生成完整 Markdown，
            再写入文件。第 2 轮未分派新任务，因此不会新增 Round 2 执行记录。

        Args:
            task: 顶层任务文本。
            files: 任务附件路径列表；当前方法接收该参数，但未将其加入模型消息。
            ctx: 共享会话上下文；未提供时创建，用其 ID 区分会话的计划文件。
            **kwargs: 总线传入的规划参数：
                task_id: 顶层任务 ID，默认 "task_unknown"。
                round_number: 当前轮数，从 1 开始，默认 1。
                max_rounds: 最大规划轮数，默认 10，由总线控制循环上限。
                agent_contract: 可用子智能体的名称与能力描述，默认空字符串。
                execution_history: 已完成轮次的执行摘要，默认空字符串。
                round_results: 上一轮结果，格式为
                    {Agent 名称: {success, result, error}}，第一轮为空字典。

        Returns:
            AgentResponse，其 extra.data["decision"] 为 PlanDecision 字典，
            extra.data["plan_path"] 为计划文件路径。

        模型调用或读取解析结果失败时，构造 is_done=True 的兜底决策，
        将错误说明写入 final_result，供总线结束本次任务。
        """
        # 优先复用总线传入的上下文，确保多轮规划使用同一个会话 ID。
        if ctx is None:
            ctx = kwargs.get("ctx") or SessionContext()

        # 读取本轮状态：轮数、可用 Agent、累计历史及上一轮的执行结果。
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
        # 获取当前会话的计划文件，并回填上一轮子任务的执行结果。
        # ------------------------------------------------------------------
        # 【创建/复用 PlanFile】首轮创建内存对象，填入原始任务、任务 ID、会话 ID，
        # 并确定路径 self.workdir/<session_id>.plan.md；后续轮次复用同一对象。
        # 此时尚未写盘，文件的创建和更新由下方 await plan_file.save() 完成。
        plan_file = self.get_or_create_plan_file(ctx.id, task_id, task)

        # 【填充执行结果】总线已执行上一轮子任务，将 success/result/error 传回。
        # 例如第 2 轮把 tool_calling 返回的问候语写入 Round 1.results，
        # 保存后显示在该轮的 Result 中；首轮无历史结果，跳过此步骤。
        if round_results and plan_file.rounds:
            plan_file.rounds[-1].results = round_results

        # ------------------------------------------------------------------
        # 通过提示词模板组装模型消息：系统消息含 Agent 能力，任务消息含执行历史。
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
        # 调用模型生成本轮决策，由模型管理器按 PlanDecision 结构解析输出。
        # ------------------------------------------------------------------
        try:
            llm_output = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=PlanDecision,    # 指定结构化响应格式
            )
            # 取得解析后的决策对象，包含 dispatches、is_done 和 final_result 等字段。
            decision: PlanDecision = llm_output.extra.parsed_model
        except Exception as exc:
            logger.error(f"| PlanningAgent LLM error: {exc}", exc_info=True)
            # 生成终止决策，避免总线继续分派；最终结果携带模型调用失败的原因。
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
        # 根据本轮决策更新计划文件：回填分析、记录分派或标记完成。
        # ------------------------------------------------------------------

        # 【填充结果分析】把当前决策对上一轮的评价写入最后一条轮次记录的 analysis，
        # 保存后显示为该轮的 Analysis；没有已有轮次时，方法内部不作修改。
        if decision.analysis:
            plan_file.update_last_analysis(decision.analysis)

        if decision.is_done:
            # 【填充最终结果】设置 PlanFile.final_result，并将 status 设为 done。
            # 此分支不会添加新的 PlanRound：例如第 2 轮确认完成，文件仍只有 Round 1。
            plan_file.finalize(    # "done", 写入 final_result
                result=decision.final_result or "",
                success=True,
            )
            # 【保存完成状态】_render() 根据全部内存数据生成 Markdown，再覆盖写盘。
            # 文件包含已回填的结果、分析及 Final Result — Completed，流程图也随之更新。
            await plan_file.save()
            logger.info("| PlanningAgent: task complete")
        elif decision.dispatches:
            # 分派决策：记录本轮目标、Agent 和子任务，实际调用由 AgentBus 执行。
            delivery = "BROADCAST" if len(decision.dispatches) > 1 else "UNICAST"
            agent_names = [d.agent_name for d in decision.dispatches]
            subtasks = {d.agent_name: d.task for d in decision.dispatches}

            # 【填充本轮计划】将模型的分派决策转换为一条执行记录，
            # 保存轮号、计划目标、Agent 名称、分派方式和各 Agent 的子任务描述。
            # 此刻子任务尚未执行，results 为空；analysis 和 timestamp 使用默认值。
            plan_round = PlanRound(
                number=round_number,
                goal=decision.plan_update,
                agents=agent_names,
                delivery_mode=delivery,
                subtasks=subtasks,
                results={},  # 下一轮调用时，用总线传回的 round_results 回填
            )
            # 将本轮记录追加到 PlanFile.rounds，保留先前轮次的内容。
            plan_file.add_round(plan_round)
            # 【保存分派计划】首轮在此创建磁盘文件，后续轮次覆盖保存完整内容。
            # _render() 自动生成任务标题、todos、执行流程图及各轮执行日志。
            await plan_file.save()
            logger.info(f"| PlanningAgent: dispatching {agent_names}")
        else:
            # 未完成且没有分派项时不新增轮次，仍保存已回填的结果和分析，
            # 交由总线判断如何处理。
            await plan_file.save()

        # ------------------------------------------------------------------
        # 将决策序列化并封装为 AgentResponse，交回总线处理。
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
