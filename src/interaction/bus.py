"""AgentBus — async message bus that drives the planner loop.

Architecture
------------
::

    submit(Task)
       │
       ▼
    session queue  (one per session_id)
       │
       ▼
    session worker  (one asyncio.Task per session_id)
       │
       ├─── Round 1: call planner → get PlanDecision
       │       │
       │       ├── dispatches: [agent_a, agent_b]   (concurrent via gather)
       │       └── collect results
       │
       ├─── Round 2: call planner again (with results) → get PlanDecision
       │       │
       │       └── dispatches: [agent_c]             (unicast)
       │
       └─── Round N: planner returns is_done=True → resolve caller Future

Concurrency model
~~~~~~~~~~~~~~~~~
Cross-session
    Each ``session_id`` owns a dedicated ``asyncio.Queue`` + worker.
    Different sessions run in complete isolation.

Intra-round
    All agents listed in one ``PlanDecision.dispatches`` are dispatched
    concurrently via ``asyncio.gather``.

Correlation
    ``submit()`` registers an ``asyncio.Future`` keyed by ``correlation_id``.
    The worker resolves it when the planning loop finishes (or errors).

Usage
~~~~~
::

    from src.interaction import bus
    from src.task import Task
    from src.session import SessionContext

    await bus.initialize()                # sync agents from ACP
    ctx = SessionContext()
    task = Task(content="...", session_id=ctx.id)
    response = await bus.submit(task, session_ctx=ctx)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.agent.server import acp
from src.interaction.types import BusEvent, BusMessage, BusMessageType, DeliveryMode
from src.logger import logger
from src.session import SessionContext
from src.task import Task

_DEFAULT_MAX_ROUNDS = 10


class AgentBus:
    """按 session 隔离的 Agent 消息总线，用来编排完整的 planner loop。

    AgentBus 负责接收顶层 Task，把任务送入对应 session 的队列，
    再由 session worker 驱动 planner 进行任务分解、子 agent 调度、
    结果收集和下一轮规划，直到 planner 返回最终结果。
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, planner_name: str = "planning") -> None:
        """初始化 AgentBus 的运行状态。

        Args:
            planner_name: 负责规划和任务分解的 agent 名称，默认是 ``planning``。
        """
        # planner agent 的名称；submit() 创建的顶层消息会先发给它。
        self.planner_name = planner_name

        # session_id -> asyncio.Queue。每个 session 有独立队列，保证同一
        # session 内的任务顺序处理，不同 session 之间互不阻塞。
        self._session_queues: Dict[str, asyncio.Queue] = {}
        # session_id -> 后台 worker 任务。worker 持续消费对应 session 队列。
        self._session_workers: Dict[str, asyncio.Task] = {}
        # session_id -> SessionContext。planner 和子 agent 调用时复用同一上下文。
        self._session_contexts: Dict[str, SessionContext] = {}

        # correlation_id -> Future。submit() 等待这个 future，planner loop
        # 完成后通过 _resolve_future() 写入最终 RESPONSE 或 ERROR。
        self._pending_responses: Dict[str, asyncio.Future] = {}

        # agent_name -> description。由 initialize() 从 ACP 同步，用于告诉
        # planner 当前有哪些可调度的子 agent。
        self._known_agents: Dict[str, str] = {}  # name → description
        # 轻量事件日志，用于观察消息入队、planner round、agent dispatch 等过程。
        self._event_log: List[BusEvent] = []
        # 延迟创建的异步锁，用于保护 session 初始化等共享状态变更。
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """获取总线内部的异步锁；首次使用时延迟创建。

        这里不在 __init__ 中直接创建锁，是为了避免 AgentBus 在没有运行中
        event loop 的上下文里初始化时绑定到错误的 loop。
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # Agent registry  (synced from ACP)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """从 ACP 获取已注册 Agent 的名称和能力描述，补充总线的 Agent 名录。

        应在 acp.initialize(...) 完成后调用。本方法不创建 Agent 实例，
        而是通过 acp.list() 获取名称，再通过 acp.get_info(name) 获取详情，
        将其保存到 self._known_agents，格式为 {Agent 名称: 能力描述}。
        例如，初始化后可包含 planning 和 tool_calling 两条记录。

        后续 _run_planner_loop() 会根据该字典构造可用子 Agent 列表
        （排除 planner 自身），供 planner 规划任务，并校验分派的 Agent 名称。

        仅添加能获取到信息且名称尚未记录的 Agent；描述为空时保存空字符串。
        已有记录不会被覆盖，也不会移除 ACP 中已不存在的 Agent。
        同步过程中若发生异常，则记录警告并结束本次同步，不向调用方抛出；
        异常发生前已添加的记录会保留。
        """
        try:
            # 获取 ACP 中已注册的 Agent 名称
            agent_names = await acp.list()
            for name in agent_names:
                # 获取该 Agent 的信息，其中包含 description
                info = await acp.get_info(name)
                if info and name not in self._known_agents:
                    # 实际赋值位置：Agent 名称作为键，能力描述作为值
                    self._known_agents[name] = info.description or ""
            logger.info(
                f"| Bus: synced {len(self._known_agents)} agents from ACP: "
                f"{list(self._known_agents.keys())}"
            )
        except Exception as exc:
            logger.warning(f"| Bus: failed to sync agents from ACP: {exc}")

    def register_agent(self, name: str, description: str = "") -> None:
        self._known_agents[name] = description

    def unregister_agent(self, name: str) -> None:
        self._known_agents.pop(name, None)

    def list_agents(self) -> List[str]:
        return list(self._known_agents.keys())

    def is_agent(self, name: str) -> bool:
        return name in self._known_agents

    def get_agent_description(self, name: str) -> Optional[str]:
        return self._known_agents.get(name)

    def _resolve_agent_name(self, raw_name: str) -> Optional[str]:
        """尝试将模型返回的 Agent 名称匹配为总线中已有的名称。

        依次尝试精确匹配、规范化后去除后缀、忽略大小写匹配和子串匹配，
        兼容模型多写 "_agent"、使用连字符或大小写不一致等情况。
        任一步匹配成功即返回；本方法只解析名称，不创建或调用 Agent。

        Args:
            raw_name: 模型返回的原始名称，例如 "Tool-Calling-Agent"。

        Returns:
            匹配到的已知 Agent 名称，例如 "tool_calling"；未匹配到返回 None。

        示例（已知名称包含 "tool_calling"）：
            "tool_calling"       -> "tool_calling"（精确匹配）
            "Tool-Calling-Agent" -> "tool_calling"（规范化并去除后缀）
            "TOOL_CALLING"       -> "tool_calling"（忽略大小写）
            "calling"            -> "tool_calling"（子串匹配）

        子串匹配按字典遍历顺序返回首个命中项，不检查是否存在多个候选。
        """
        # 字典的键是总线已知的 Agent 名称；本方法只使用键进行匹配。
        known = self._known_agents
        # 1. 原始名称完全一致时直接返回，优先保留精确匹配结果。
        if raw_name in known:
            return raw_name
        # 规范化输入：转为小写，将连字符替换为下划线。
        lower = raw_name.lower().replace("-", "_")
        # 2. 尝试去除末尾的 "_agent" 或 " agent"，再精确查找候选名称。
        # 例如 "tool_calling_agent" -> "tool_calling"；此步骤不改变 lower。
        for suffix in ("_agent", " agent"):
            if lower.endswith(suffix):
                candidate = lower[: -len(suffix)]
                if candidate in known:
                    return candidate
        # 3. 将已知名称转为小写，与规范化后的输入比较；返回原始注册名称。
        for k in known:
            if k.lower() == lower:
                return k
        # 4. 最后尝试双向子串匹配：已知名称包含输入，或输入包含已知名称。
        # 例如 "calling" 是 "tool_calling" 的子串；命中第一个候选即返回。
        # 此处使用原始键 k，不会额外将已知名称转为小写。
        for k in known:
            if k in lower or lower in k:
                return k
        # 所有规则均未命中，交由调用方处理无法识别的 Agent 名称。
        return None

    # ------------------------------------------------------------------
    # Public API — submit
    # ------------------------------------------------------------------

    async def submit(
        self,
        task: Task,
        session_ctx: Optional[SessionContext] = None,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
    ) -> BusMessage:
        """提交一个任务；总线会运行完整的 planner 循环，并返回最终结果。

        调用链路:
            examples/run_bus.py
                -> bus.submit(...)
                -> _ensure_session(...)
                -> _session_worker(...)
                -> _run_planner_loop(...)
                -> _call_planner_raw(...)
                -> acp(name="planning", ...)
                -> PlanningAgent.__call__(...)
                -> PlanDecision.dispatches
                -> _call_agent(...) for each dispatched sub-agent
                -> acp(name=<sub_agent>, ...)
                -> 收集本轮结果，并在下一轮回传给 planner.

        这个方法只负责把顶层任务放入对应 session 的队列，并等待session worker 解析 correlation future。
        真正的任务规划、子任务分解、子 agent 调度和结果收集都发生在 _run_planner_loop.

        Args:
            task:        Top-level work unit.
            session_ctx: Session context.  Auto-created if ``None``.
            max_rounds:  Maximum number of planner iterations.

        Returns:
            Terminal ``BusMessage`` (RESPONSE or ERROR).
        """
        if session_ctx is None:
            session_ctx = SessionContext()
        if task.session_id is None:
            task.session_id = session_ctx.id

        # 确保当前 session 已经有独立的队列和 worker，然后才能把任务入队。
        await self._ensure_session(task.session_id, session_ctx)

        # 初始消息永远发给 planner；后续由 planner 决定把哪些具体子任务派给哪些 agent。
        msg = BusMessage.task_message(
            session_id=task.session_id,
            task_id=task.id,
            content=task.content,
            files=task.files,
            recipients=[self.planner_name],
            delivery_mode=DeliveryMode.UNICAST,
        )
        msg.payload["max_rounds"] = max_rounds

        # 创建一个 Future，用于等待这次任务的最终结果。
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        # 记录“消息关联 ID -> Future”的对应关系，
        # 后台 worker 就能根据 correlation_id 找到应该把结果交给谁。
        self._pending_responses[msg.correlation_id] = future

        # 将任务状态设为 RUNNING，并更新状态变更时间 updated_at
        task.mark_running()
        # 将任务消息放入当前 session 的队列，等待 _session_worker() 按顺序取出，
        # 再调用 _run_planner_loop() 完成任务规划和执行。
        await self._session_queues[task.session_id].put(msg)
        self._log_event(msg, "message_enqueued")
        logger.info(f"| Bus: task '{task.id}' enqueued (session='{task.session_id}')")

        try:
            # 暂停当前协程，等待后台 worker 将任务的最终结果写入 Future；
            # 结果就绪后继续执行，将结果赋给 response，并据此更新任务状态。
            response: BusMessage = await future
            if response.type == BusMessageType.ERROR:
                task.mark_failed()
            else:
                task.mark_done()
            return response
        except asyncio.CancelledError:
            task.mark_cancelled()
            raise
        except Exception:
            task.mark_failed()
            raise

    # ------------------------------------------------------------------
    # Public API — direct dispatch (used ONLY for non-planner agents)
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        agent_name: str,
        message: BusMessage,
        session_ctx: Optional[SessionContext] = None,
    ) -> BusMessage:
        """Dispatch a single message to a single agent via ACP.

        This is a **direct call**, not queued.  The bus uses it internally
        to call sub-agents inside the planner loop, and it can also be used
        externally for one-off agent calls outside the planning flow.
        """
        ctx = session_ctx or self._session_contexts.get(message.session_id)
        return await self._call_agent(agent_name, message, ctx=ctx)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel all workers and unblock all pending Futures."""
        logger.info("| Bus: shutting down")
        for sid, worker in list(self._session_workers.items()):
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        for cid, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(RuntimeError("AgentBus shut down"))
        self._session_workers.clear()
        self._session_queues.clear()
        self._session_contexts.clear()
        self._pending_responses.clear()
        logger.info("| Bus: shutdown complete")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_event_log(self, session_id: Optional[str] = None) -> List[BusEvent]:
        if session_id:
            return [e for e in self._event_log if e.session_id == session_id]
        return list(self._event_log)

    def list_sessions(self) -> List[str]:
        return list(self._session_queues.keys())

    # ------------------------------------------------------------------
    # Internal — session management
    # ------------------------------------------------------------------

    async def _ensure_session(
        self, session_id: str, session_ctx: SessionContext
    ) -> None:
        """确保指定 session 的队列、上下文和后台 worker 已初始化。

        如果这是第一次看到该 session_id，则创建一个独立的 asyncio.Queue，
        保存对应的 SessionContext，并启动一个 _session_worker() 后台任务。
        后续同一 session 的 submit() 会复用这些资源，不会重复创建 worker。
        """
        # 使用锁保护 session 初始化，避免并发 submit 同一个 session 时重复创建资源。
        async with self._get_lock():
            if session_id not in self._session_queues:
                # 每个 session 拥有独立队列；同一 session 内的任务按入队顺序处理。
                self._session_queues[session_id] = asyncio.Queue()
                # 保存 session 上下文，后续 planner 和子 agent 调用都会复用它。
                self._session_contexts[session_id] = session_ctx

                # 创建一个后台任务，让 _session_worker(session_id) 持续运行；
                # 它会不断从当前 session 的队列里取任务并执行 planner loop
                worker = asyncio.create_task(
                    self._session_worker(session_id),
                    name=f"bus-worker-{session_id}",
                )
                # 记录 worker 句柄，便于 shutdown 时统一取消和清理。
                self._session_workers[session_id] = worker
                logger.info(f"| Bus: session worker created for '{session_id}'")

    # ------------------------------------------------------------------
    # Internal — session worker (drains the queue)
    # ------------------------------------------------------------------

    async def _session_worker(self, session_id: str) -> None:
        """长期运行的 session 后台协程，按队列顺序处理该 session 内的任务。

        每个 session 会对应一个 worker。worker 从该 session 的队列中不断
        取出 BusMessage，并交给 _run_planner_loop() 执行完整的
        planner -> dispatch -> collect -> repeat 流程。

        同一 session 内任务是串行处理的；不同 session 的 worker 可以并发运行。
        """
        # 取出当前 session 专属队列；该队列在 _ensure_session() 中创建。
        queue = self._session_queues[session_id]
        logger.info(f"| Bus: worker started (session='{session_id}')")

        while True:
            try:
                # 阻塞等待下一个入队任务。submit() 会把顶层任务消息放进这个队列。
                message: BusMessage = await queue.get()
                try:
                    # 对单个任务运行完整 planner loop；成功时内部会 resolve future。
                    await self._run_planner_loop(message)
                except Exception as exc:
                    # planner loop 内部出现未捕获异常时，也要把错误写回 submit()
                    # 正在等待的 future，避免调用方永久挂起。
                    logger.error(
                        f"| Bus: error in planner loop for '{message.id}': {exc}",
                        exc_info=True,
                    )
                    self._resolve_future(
                        message.correlation_id,
                        BusMessage.error_message(
                            session_id=message.session_id,
                            task_id=message.task_id,
                            correlation_id=message.correlation_id,
                            error=str(exc),
                            parent_id=message.id,
                        ),
                    )
                finally:
                    # 无论成功、失败还是异常，都标记本条队列消息已处理完成。
                    queue.task_done()
            except asyncio.CancelledError:
                # shutdown() 会取消 worker；收到取消信号后退出循环。
                logger.info(f"| Bus: worker cancelled (session='{session_id}')")
                break

    # ------------------------------------------------------------------
    # Internal — THE planner loop  (the core of the bus)
    # ------------------------------------------------------------------

    async def _run_planner_loop(self, message: BusMessage) -> None:
        """处理一个顶层任务，循环执行规划、分派子任务和收集结果。

        由 _session_worker() 从队列取出消息后调用。每轮通过
        _call_planner_raw() 调用 planner，获取计划决策字典：
        若 is_done 为真，则交付最终结果；否则按 dispatches 并发调用
        子 Agent，收集结果后再让 planner 决定下一步。
        任务如何分解由 planner 决定，本方法负责调度与结果传递。
        max_rounds 限制规划轮数，不代表子任务数量或固定执行步骤数。

        结果通过 _resolve_future() 写入 submit() 等待的 Future，
        而不是通过本方法的 return 返回。消息过期、planner 无决策、
        未完成却无子任务或耗尽轮数时，向调用方交付 ERROR 消息。
        未在此处捕获的异常交给 _session_worker() 处理。

        调用链路：
            submit() 入队 -> _session_worker() 取出消息 -> 本方法
            -> _call_planner_raw() 获取决策 -> _call_agent() 执行子任务
            -> 收集结果并进入下一轮 -> _resolve_future() 交付最终消息
            -> submit() 中的 await future 得到结果并继续执行。

        Args:
            message: 顶层任务消息，包含会话、任务、关联 ID，以及
                payload 中的任务内容、附件和可选的最大规划轮数。

        Returns:
            None。最终结果通过对应的 Future 交付。
        """
        # 开始处理前检查消息是否超过有效期（TTL，单位为秒）。
        # 已过期则交付错误消息并结束；此处不是整个执行过程的超时控制。
        if message.is_expired():
            self._log_event(message, "ttl_expired")
            self._resolve_future(
                message.correlation_id,
                BusMessage.error_message(
                    session_id=message.session_id,
                    task_id=message.task_id,
                    correlation_id=message.correlation_id,
                    error=f"Message expired (TTL={message.ttl}s)",
                    parent_id=message.id,
                ),
            )
            return

        # 从顶层消息中读取任务输入和轮数限制，并取得当前会话的共享上下文。
        session_id = message.session_id
        task_id = message.task_id
        task_content = message.payload.get("content", "")
        task_files = message.payload.get("files", [])
        max_rounds = message.payload.get("max_rounds", _DEFAULT_MAX_ROUNDS)
        ctx = self._session_contexts.get(session_id)

        # 整理总线已知的子 Agent 名称和能力描述，告诉 planner 可以分派给谁。
        # 列表不包含 planner 自身，并要求决策使用列表中的准确名称。
        sub_agent_lines = []
        # 运行时示例：self._known_agents = {
        #     "planning": "拆解任务，并决定下一步调用哪些子 Agent。"
        #                 "返回一个 PlanDecision（计划决策），由 AgentBus 驱动整个执行循环。",
        #     "tool_calling": "通过调用工具来完成任务的 Agent。",
        # }
        # name 是 Agent 名称，desc 是能力描述；下面会跳过 planning，保留 tool_calling。
        for name, desc in self._known_agents.items():
            if name == self.planner_name:
                continue
            sub_agent_lines.append(f"- **{name}**: {desc or 'No description'}")
        if sub_agent_lines:
            # 'Available agents (use these EXACT names in dispatches):
            # - **tool_calling**: A tool calling agent that can call tools to complete tasks.'
            agent_contract = (
                "Available agents (use these EXACT names in dispatches):\n"
                + "\n".join(sub_agent_lines)
            )
        else:
            agent_contract = "(no sub-agents available)"

        # 保存最近一轮实际执行的结果；第一轮尚未执行子任务，因此为空。
        round_results: Dict[str, Any] = {}
        # 累积各轮的计划和执行摘要，供 planner 了解之前的处理过程。
        execution_history = ""

        # 每轮先规划，再执行该轮子任务；收集完结果后才进入下一轮规划。
        for round_num in range(1, max_rounds + 1):
            logger.info(
                f"| Bus: planner round {round_num}/{max_rounds} "
                f"(session='{session_id}')"
            )
            self._log_event(message, "planner_round_start", detail=str(round_num))

            # 1. 将原始任务、可用智能体及其能力描述、历史执行摘要和最近一轮结果传给 Planner，
            #    由它判断任务是否完成，并返回本轮需要分派的子任务或最终结果。这也明确了 decision_dict 承载的是本轮规划决策
            decision_dict = await self._call_planner_raw(
                task_content=task_content,
                task_files=task_files,
                ctx=ctx,
                task_id=task_id,
                round_number=round_num,
                max_rounds=max_rounds,
                agent_contract=agent_contract,
                execution_history=execution_history,
                round_results=round_results,
            )

            # 没有取得决策时无法继续调度，向等待结果的 submit() 交付错误消息。
            if decision_dict is None:
                logger.error("| Bus: planner returned None — aborting")
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.error_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        error="Planner returned no decision",
                        parent_id=message.id,
                    ),
                )
                return

            # 读取完成标记、最终答案、本轮子任务列表和计划说明。
            # 每条 dispatch 提供目标 agent_name、子任务 task 和可选的 files。
            is_done = decision_dict.get("is_done", False)
            final_result = decision_dict.get("final_result")
            dispatches = decision_dict.get("dispatches", [])
            plan_update = decision_dict.get("plan_update", "")

            # 两轮规划示例（仅保留关键字段，任务描述作简化）：
            # 第 1 轮（round 1/10）：分派任务，调用工具智能体执行 hello world 技能。
            # {
            #     "analysis": "",
            #     "plan_update": "分派 tool_calling 智能体执行 hello world 技能。",
            #     "is_done": False,
            #     "dispatches": [{
            #         "agent_name": "tool_calling",
            #         "files": [],
            #         "task": "执行 hello world 技能，返回问候语。",
            #     }],
            #     "final_result": "",
            # }
            # 执行结果：跳过 if is_done，dispatches 非空，继续校验目标 Agent，
            # 调用 tool_calling 执行子任务，收集结果并写入执行历史，再进入第 2 轮。
            #
            # 第 2 轮（round 2/10）：正确标记完成，并提供最终结果。
            # {
            #     "analysis": "上一轮执行成功，返回了有效的问候语。",
            #     "plan_update": "All sub-tasks completed.",
            #     "is_done": True,
            #     "dispatches": [],
            #     "final_result": "技能执行成功：Hey there, World! 👋 Welcome aboard!",
            # }
            # 执行结果：进入 if is_done，向 submit() 交付 success=True 和
            # final_result，然后 return；不会触发空分派错误，也不会进入第 3 轮。

            # 2. planner 判断任务已完成：将正常结果写入 Future，然后结束循环。
            #    final_result 为空值时使用 plan_update；此分支不再执行 dispatches。
            if is_done:
                logger.info(f"| Bus: planner signalled done (round {round_num})")
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.response_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        sender=self.planner_name,
                        success=True,
                        result=final_result or plan_update,
                    ),
                )
                return

            # 3. 尚未完成却没有任何子任务，说明本轮无法推进，交付错误并结束。
            if not dispatches:
                logger.warning(
                    f"| Bus: planner returned no dispatches in round {round_num} — aborting"
                )
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.error_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        error="Planner stalled (no dispatches and not done)",
                        parent_id=message.id,
                    ),
                )
                return

            # 4. 校验目标 Agent 名称；未精确匹配时尝试名称解析，
            #    使用解析后的名称执行，无法识别的分派项则跳过。
            known = set(self._known_agents.keys())
            valid_dispatches = []
            for d in dispatches:
                raw_name = d["agent_name"]
                if raw_name in known:
                    valid_dispatches.append(d)
                else:
                    resolved = self._resolve_agent_name(raw_name)
                    if resolved:
                        logger.warning(
                            f"| Bus: LLM returned '{raw_name}', resolved to '{resolved}'"
                        )
                        d["agent_name"] = resolved
                        valid_dispatches.append(d)
                    else:
                        logger.error(
                            f"| Bus: unknown agent '{raw_name}' — skipping "
                            f"(known: {list(known)})"
                        )
            dispatches = valid_dispatches

            # 原本有子任务，但名称全部无效：记录原因，进入下一轮重新规划。
            # 本轮没有执行子任务，因此保留原有 round_results。
            if not dispatches:
                logger.warning(
                    "| Bus: all dispatched agent names were invalid — aborting round"
                )
                execution_history += (
                    f"=== Round {round_num} ===\n"
                    f"ERROR: All agent names were invalid.\n\n"
                )
                continue

            agent_names = [d["agent_name"] for d in dispatches]
            # 此处的 BROADCAST 仅用于日志描述多目标分派；下面每条消息仍为单播。
            delivery = "BROADCAST" if len(agent_names) > 1 else "UNICAST"
            logger.info(f"| Bus: {delivery} → {agent_names}")

            # 为每条分派项构造一条 PLAN 消息，携带该子任务的内容和附件。
            # 这些消息随后直接交给 _call_agent()，不再进入 session 队列。
            sub_messages = [
                BusMessage(
                    type=BusMessageType.PLAN,
                    session_id=session_id,
                    task_id=task_id,
                    sender="bus",
                    recipients=[d["agent_name"]],
                    delivery_mode=DeliveryMode.UNICAST,
                    payload={
                        "content": d["task"],
                        "files": d.get("files", []),
                    },
                )
                for d in dispatches
            ]

            # 并发执行本轮 dispatches 中的任务，全部结束后才进入下一轮；依赖任务由 Planner 分轮。
            # 结果按输入顺序返回， return_exceptions=True,将子调用抛出的异常也作为结果列表项收集
            raw_responses = await asyncio.gather(
                *[
                    self._call_agent(m.recipients[0], m, ctx=ctx)
                    for m in sub_messages
                ],
                return_exceptions=True,
            )

            # 完整执行结果示例（保留全部字段及原始值，按字段换行展示）：
            # 本轮只有一个 tool_calling 子任务，因此返回含一个 BusMessage 的列表。
            # raw_responses = [
            #     BusMessage(
            #         id="msg_20260924-105051_7c3974ce",
            #         type=BusMessageType.RESPONSE,
            #         session_id="session_20260924-104855_91e0e90b",
            #         task_id="task_20260924-104855_8555236e",
            #         correlation_id="corr_20260924-104920_cb1b1ebb",
            #         parent_id="msg_20260924-104920_c7a98686",
            #         sender="tool_calling",
            #         recipients=["bus"],
            #         delivery_mode=DeliveryMode.UNICAST,
            #         payload={
            #             "success": True,
            #             "result": (
            #                 "Successfully invoked hello-world skill. Result:\n\n"
            #                 "---\n**Greeting:** Hey there, World! 👋 Welcome aboard!\n"
            #                 "---\n\n*Generated by hello-world skill*"
            #             ),
            #             "error": None,
            #         },
            #         created_at=datetime.datetime(
            #             2026, 9, 24, 2, 50, 51, 246489, tzinfo=datetime.timezone.utc
            #         ),
            #         ttl=None,
            #     ),
            # ]
            # 含义：tool_calling 已成功执行技能，并向 bus 返回问候语。
            # 下方循环读取 payload 的 success、result、error，写入 round_results，
            # 供下一轮 Planner 判断整体任务是否完成；子任务成功本身不会结束规划循环。

            # 5. 用本轮结果替换上一轮结果，并构造本轮的文本摘要。
            round_results = {}
            history_parts: List[str] = [
                f"=== Round {round_num} ===",
                f"Plan: {plan_update}",
                f"Dispatched ({delivery}): {', '.join(agent_names)}",
            ]

            # gather 保留输入顺序，因此可以将分派项与返回结果一一配对。
            # 结果以 Agent 名称为键；同名 Agent 多次执行时，后一个会覆盖前一个。
            for d, raw in zip(dispatches, raw_responses):
                name = d["agent_name"]
                subtask = d["task"]
                history_parts.append(f"  {name} subtask: {subtask[:200]}")

                # 调用抛出异常时记录失败，供下一轮 planner 判断如何继续。
                if isinstance(raw, BaseException):
                    logger.error(f"| Bus: agent '{name}' error: {raw}")
                    round_results[name] = {"success": False, "error": str(raw)}
                    history_parts.append(f"  FAIL {name}: {str(raw)[:300]}")
                else:
                    # 正常返回 BusMessage 也可能表示失败，需读取 payload.success。
                    # 保留结果或错误文本；历史摘要只截取前 300 个字符。
                    ok = raw.payload.get("success", False)
                    result_text = str(
                        raw.payload.get("result") or raw.payload.get("error") or ""
                    )
                    round_results[name] = {
                        "success": ok,
                        "result": result_text,
                        "error": raw.payload.get("error"),
                    }
                    tag = "OK" if ok else "FAIL"
                    history_parts.append(f"  {tag} {name}: {result_text[:300]}")
                    logger.info(f"| Bus: agent '{name}' → {tag}")

            # 将本轮摘要追加到累计历史；下一轮调用 planner 时传入更新后的数据。
            history_parts.append("")
            execution_history += "\n".join(history_parts) + "\n"

            self._log_event(message, "round_complete", detail=str(round_num))

        # 达到最大轮数仍未收到完成决策，先向 submit() 交付超限错误。
        logger.warning(f"| Bus: max rounds ({max_rounds}) reached")
        self._resolve_future(
            message.correlation_id,
            BusMessage.error_message(
                session_id=session_id,
                task_id=task_id,
                correlation_id=message.correlation_id,
                error=f"Planner did not finish within {max_rounds} rounds",
                parent_id=message.id,
            ),
        )

        # 额外调用一次 planner，传入超出上限的轮号，供它进行失败收尾。
        # 此次返回的决策不再处理，也不会执行新子任务或改变已交付的错误结果。
        await self._call_planner_raw(
            task_content=task_content,
            task_files=task_files,
            ctx=ctx,
            task_id=task_id,
            round_number=max_rounds + 1,
            max_rounds=max_rounds,
            agent_contract=agent_contract,
            execution_history=execution_history,
            round_results=round_results,
        )

    # ------------------------------------------------------------------
    # 内部方法：通过 ACP 调用规划智能体，提取 PlanDecision 决策字典
    # ------------------------------------------------------------------

    async def _call_planner_raw(
        self,
        task_content: str,
        task_files: List[str],
        ctx: Optional[SessionContext],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """通过 ACP 调用规划智能体，返回响应中的原始决策字典。

        ACP 根据 self.planner_name 路由到已初始化的规划智能体。
        Planner 将 PlanDecision 序列化到 AgentResponse.extra.data["decision"]，
        本方法仅提取该字段，供上层规划循环解析和处理，不在此校验决策结构。

        Args:
            task_content: 传给规划智能体的任务文本。
            task_files: 任务关联的文件路径列表；为空时向智能体传入 None。
            ctx: 共享会话上下文；原样传递给 ACP，可为 None。
            **kwargs: 透传给 ACP 的规划参数，例如 task_id、round_number、
                max_rounds、agent_contract、execution_history 和 round_results。

        Returns:
            Planner 返回的 decision 字典；响应缺少有效的 extra、data 或
            decision 字段时返回 None。调用或提取过程中发生 Exception 时，
            记录包含堆栈的错误日志并返回 None，交由调用方处理。
        """
        try:
            # 按名称调用规划智能体，传入任务、附件、会话上下文和本轮规划参数。
            agent_response = await acp(
                name=self.planner_name,
                input={
                    "task": task_content,
                    "files": task_files if task_files else None,
                },
                ctx=ctx,
                **kwargs,
            )
            # 决策位于结构化扩展数据中；缺少扩展数据时跳过提取。
            if hasattr(agent_response, "extra") and agent_response.extra:
                # data 为空时使用空字典，缺少 decision 键时 get() 返回 None。
                data = agent_response.extra.data or {}
                return data.get("decision")
        except Exception as exc:
            # 将调用或响应提取异常记录下来，使用 None 表示本次未取得决策。
            logger.error(f"| Bus: planner call failed: {exc}", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # 内部方法：通过 ACP 调用指定 Agent，将执行结果封装为总线消息
    # ------------------------------------------------------------------

    async def _call_agent(
        self,
        agent_name: str,
        message: BusMessage,
        ctx: Optional[SessionContext] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BusMessage:
        """执行一个子任务，并将目标 Agent 的返回值转换为 BusMessage。

        从消息中提取任务和附件，通过 ACP 按名称调用已初始化的 Agent。
        本方法处理一次调用；多个子任务的并发调度由上层 asyncio.gather() 负责。

        Args:
            agent_name: 目标 Agent 名称，例如 "tool_calling"。
            message: 子任务消息，payload.content 为任务文本，payload.files 为附件。
            ctx: 共享会话上下文，传给 ACP；为 None 时由 ACP 的上下文管理器创建。
            extra_kwargs: 传给 Agent 的额外参数；复制到新字典后展开传递。

        Returns:
            正常返回时生成 RESPONSE 消息，payload 包含 success、result 和 error。
            Agent 返回 success=False 时仍为 RESPONSE，但 error 会记录失败说明。
            try 块内发生 Exception 时生成 ERROR 消息，保存异常文本。

        例如 tool_calling 成功返回问候语后，上层会收到 payload 为
        {"success": True, "result": "Hey there, World! 👋 Welcome aboard!",
         "error": None} 的 RESPONSE 消息，供下一轮 Planner 分析。
        取消等不属于 Exception 的异常不在此处捕获。
        """
        # 记录本次分派事件及目标名称，方便追踪子任务的执行过程。
        self._log_event(message, "agent_dispatched", agent_name=agent_name)
        logger.info(f"| Bus: dispatching → '{agent_name}'")

        try:
            # 复制额外参数，避免修改调用方传入的字典；未提供时使用空字典。
            call_kwargs: Dict[str, Any] = {}
            if extra_kwargs:
                call_kwargs.update(extra_kwargs)

            # 按名称找到 Agent 实例，将消息中的 content 映射为其 task 参数。
            # 例如 agent_name="tool_calling" 时，最终进入 ToolCallingAgent.__call__()。
            agent_result = await acp(
                name=agent_name,
                input={
                    "task": message.payload.get("content", ""),
                    "files": message.payload.get("files", []),
                },
                ctx=ctx,
                **call_kwargs,
            )

            # 优先读取返回对象的 success 属性；没有该属性时使用对象的布尔值。
            success = getattr(agent_result, "success", bool(agent_result))
            # 优先读取 message 属性作为结果；没有该属性时使用对象的字符串表示。
            result_data = getattr(agent_result, "message", str(agent_result))
            # 成功时无错误信息；返回失败时，将结果文本同时作为错误说明。
            error_str = None if success else result_data

            # 封装为 RESPONSE：保留会话、任务和关联 ID，以目标 Agent 为发送者。
            # parent_id 指向输入的子任务消息，便于追溯该结果对应哪次分派。
            return BusMessage.response_message(
                session_id=message.session_id,
                task_id=message.task_id,
                correlation_id=message.correlation_id,
                sender=agent_name,
                success=success,
                result=result_data,
                error=error_str,
                parent_id=message.id,
            )

        except Exception as exc:
            # 调用或结果封装过程中抛出普通异常时，记录堆栈并返回 ERROR 消息。
            # 上层可将该错误与其他子任务结果一起收集，再交给 Planner 判断下一步。
            logger.error(f"| Bus: agent '{agent_name}' raised: {exc}", exc_info=True)
            return BusMessage.error_message(
                session_id=message.session_id,
                task_id=message.task_id,
                correlation_id=message.correlation_id,
                error=str(exc),
                parent_id=message.id,
            )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _resolve_future(self, correlation_id: str, response: BusMessage) -> None:
        """将任务的最终结果交给正在等待的 submit() 协程。

        submit() 提交任务时，会按 correlation_id 保存一个 Future，
        并通过 await future 等待结果。本方法找到该 Future 并设置结果后，
        等待它的协程便可恢复执行，取得 response。
        如果 Future 不存在、已完成或已取消，则不再设置结果。

        Args:
            correlation_id: 关联请求与结果的 ID，用于查找对应的 Future。
            response: 任务的最终消息，可以是正常结果或 ERROR 消息。
        """
        # 取出并移除对应的 Future，结束对此请求的跟踪；不存在时返回 None。
        future = self._pending_responses.pop(correlation_id, None)
        # done() 对已完成或已取消的 Future 都返回 True，避免重复设置结果。
        if future is not None and not future.done():
            # await future 将得到此 response；ERROR 消息也作为结果传递，
            # 由 submit() 检查消息类型并将任务标记为失败。
            future.set_result(response)

    def _log_event(
        self,
        message: BusMessage,
        event_type: str,
        agent_name: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """将一条消息相关的事件追加到总线的内存事件列表。

        记录消息所属的会话、任务和消息 ID，以及事件类型和可选详情，
        便于追踪任务处理过程。本方法只保存事件记录，不负责消息分发，
        也不会将记录写入磁盘。

        Args:
            message: 发生该事件的消息，提供会话、任务和消息 ID。
            event_type: 事件类型，例如 message_enqueued（消息已入队）。
            agent_name: 与事件相关的 Agent 名称；没有时可省略。
            detail: 事件的补充说明；没有时可省略。
        """
        # 构造事件并追加到列表，保留已有记录；事件 ID 和时间由 BusEvent 自动生成。
        self._event_log.append(
            BusEvent(
                # 这些 ID 将事件关联到具体的会话、任务和消息，便于后续查询。
                session_id=message.session_id,
                task_id=message.task_id,
                message_id=message.id,
                event_type=event_type,
                agent_name=agent_name,
                detail=detail,
            )
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

bus = AgentBus()
