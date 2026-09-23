"""Message and event types for the AgentBus inter-agent communication layer.

These types are the core vocabulary of the bus:

* `BusMessage`  — the envelope that carries every request and response.
* `DeliveryMode` — unicast (point-to-point), broadcast (fan-out), or anycast
                   (best-matching agent, resolved by ACP FAISS retrieval).
* `BusMessageType` — distinguishes task submissions, agent responses, planner
                     plans, errors, and heartbeats.
* `BusEvent`    — lightweight observability record appended to the session log.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.utils import generate_unique_id


class DeliveryMode(str, Enum):
    """How a BusMessage is delivered to its recipients.

    UNICAST   — exactly one recipient (point-to-point).
    BROADCAST — all listed recipients receive the message concurrently
                (bus uses asyncio.gather internally).
    ANYCAST   — bus selects the best-matching agent via ACP semantic
                retrieval; `recipients` may be left empty.
    """

    UNICAST = "unicast"
    BROADCAST = "broadcast"
    ANYCAST = "anycast"


class BusMessageType(str, Enum):
    """Semantic classification of bus messages."""

    TASK = "task"          # top-level task from external caller → planner
    RESPONSE = "response"  # agent result → planner or bus
    PLAN = "plan"          # planner routing directive → sub-agents
    ERROR = "error"        # delivery or execution failure
    HEARTBEAT = "heartbeat"  # liveness ping (no payload required)


class BusMessage(BaseModel):
    """Envelope that wraps every request and response on the AgentBus.

    Correlation and session fields together give the bus the identifiers
    it needs to route messages without a global lock:

    * ``session_id``    — isolates concurrent sessions; each session owns
                          a dedicated asyncio.Queue and worker Task.
    * ``task_id``       — groups all messages belonging to one top-level Task.
    * ``correlation_id`` — pairs a request with its reply (Future key).
    * ``parent_id``     — reconstructs the message chain for tracing.
    """

    id: str = Field(
        default_factory=lambda: generate_unique_id("msg"),
        description="Unique identifier for this message.",
    )
    type: BusMessageType = Field(description="Semantic type of the message.")
    session_id: str = Field(description="Session that owns this message.")
    task_id: str = Field(description="Top-level task this message belongs to.")
    correlation_id: str = Field(
        default_factory=lambda: generate_unique_id("corr"),
        description=(
            "Matches a reply to its originating request.  The bus registers "
            "an asyncio.Future keyed by this value and resolves it when the "
            "corresponding RESPONSE message arrives."
        ),
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of the message that triggered this one (for chain tracing).",
    )
    sender: str = Field(
        default="bus",
        description="Name of the originating agent, or 'bus' for bus-generated messages.",
    )
    recipients: List[str] = Field(
        default_factory=list,
        description=(
            "Target agent names.  For UNICAST exactly one entry is expected. "
            "For BROADCAST all listed agents receive the message. "
            "For ANYCAST this may be empty; the bus resolves the best agent."
        ),
    )
    delivery_mode: DeliveryMode = Field(
        default=DeliveryMode.UNICAST,
        description="Delivery semantics for this message.",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Message body.  Conventions by type: "
            "TASK → {'content': str, 'files': list}; "
            "PLAN → {'plan_text': str, 'plan_path': str}; "
            "RESPONSE → {'success': bool, 'result': Any, 'error': str|None}."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC creation timestamp.",
    )
    ttl: Optional[int] = Field(
        default=None,
        description=(
            "Time-to-live in seconds from created_at.  The session worker "
            "checks expiry before dispatching; expired messages are resolved "
            "as FAILED so callers are never stuck.  None means no expiry."
        ),
    )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def task_message(
        cls,
        session_id: str,
        task_id: str,
        content: str,
        files: Optional[List[str]] = None,
        recipients: Optional[List[str]] = None,
        delivery_mode: DeliveryMode = DeliveryMode.UNICAST,
        ttl: Optional[int] = None,
    ) -> "BusMessage":
        """根据顶层 Task 的内容构造一个 TASK 类型的总线消息。

        这个便捷构造器主要供 AgentBus.submit() 使用：submit() 先把
        Task.content 和 Task.files 包进 BusMessage.payload，然后把消息放入
        当前 session 的队列。后续 _session_worker() 会取出这个 TASK 消息，
        交给 _run_planner_loop()，再由 planner 进行任务分解和子 agent 调度。
        """
        return cls(
            # TASK 表示外部提交的顶层任务，第一站通常是 planner。
            type=BusMessageType.TASK,
            # session_id 决定消息进入哪个 session 队列。
            session_id=session_id,
            # task_id 用来把同一个顶层任务产生的多轮消息串起来。
            task_id=task_id,
            # 顶层任务消息由 bus 创建，因此 sender 固定为 "bus"。
            sender="bus",
            # submit() 中通常传入 [planner_name]，表示先交给 planner 处理。
            recipients=recipients or [],
            # 默认 UNICAST，即点对点发给一个接收者。
            delivery_mode=delivery_mode,
            # planner 会从 payload 中读取原始任务内容和附件文件列表。
            payload={"content": content, "files": files or []},
            # 可选过期时间；超时消息会被 worker 解析为失败，避免调用方一直等待。
            ttl=ttl,
        )

    @classmethod
    def response_message(
        cls,
        session_id: str,
        task_id: str,
        correlation_id: str,
        sender: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> "BusMessage":
        """Build a RESPONSE-type message from an agent result."""
        return cls(
            type=BusMessageType.RESPONSE,
            session_id=session_id,
            task_id=task_id,
            correlation_id=correlation_id,
            parent_id=parent_id,
            sender=sender,
            recipients=["bus"],
            delivery_mode=DeliveryMode.UNICAST,
            payload={"success": success, "result": result, "error": error},
        )

    @classmethod
    def error_message(
        cls,
        session_id: str,
        task_id: str,
        correlation_id: str,
        error: str,
        parent_id: Optional[str] = None,
    ) -> "BusMessage":
        """Build an ERROR-type message."""
        return cls(
            type=BusMessageType.ERROR,
            session_id=session_id,
            task_id=task_id,
            correlation_id=correlation_id,
            parent_id=parent_id,
            sender="bus",
            recipients=["bus"],
            delivery_mode=DeliveryMode.UNICAST,
            payload={"success": False, "result": None, "error": error},
        )

    def is_expired(self) -> bool:
        """Return True if the message has exceeded its TTL."""
        if self.ttl is None:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl


class BusEvent(BaseModel):
    """Lightweight observability record written to the session event log.

    BusEvents are append-only and never affect routing.  They are useful
    for debugging, replay, and audit trails.
    """

    id: str = Field(default_factory=lambda: generate_unique_id("event"))
    session_id: str
    task_id: str
    message_id: str
    event_type: str = Field(
        description="E.g. 'message_enqueued', 'message_dispatched', 'response_received', 'ttl_expired'."
    )
    agent_name: Optional[str] = None
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
