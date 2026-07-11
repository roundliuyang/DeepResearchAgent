"""Agent Context Protocol (ACP) Types

Core type definitions for the Agent Context Protocol and common Agent
abstractions, aligned with the design of `src.tool.types`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union


import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.dynamic import dynamic_manager
from src.environment.server import ecp
from src.logger import logger
from src.memory import EventType, memory_manager
from src.message.types import HumanMessage, Message, SystemMessage
from src.model import model_manager
from src.prompt import prompt_manager
from src.tool.server import tcp
from src.skill.server import scp
from src.utils import (
    dedent,
    get_file_info,
)
from src.session import SessionContext

class InputArgs(BaseModel):
    task: str = Field(description="The task to complete.")
    files: Optional[List[str]] = Field(default=None, description="The files to attach to the task.")

class ACPErrorCode(Enum):
    """ACP error codes."""
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AGENT_NOT_FOUND = -32001

class ACPError(BaseModel):
    """ACP error structure."""
    code: ACPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None

class ACPRequest(BaseModel):
    """ACP request structure."""
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None

class ACPResponse(BaseModel):
    """ACP response structure."""
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ACPError] = None

class AgentConfig(BaseModel):
    """Agent configuration for registration, similar to `ToolConfig`."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the agent")
    description: str = Field(description="The description of the agent")
    version: str = Field(default="1.0.0", description="Version of the agent")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")

    cls: Optional[Any] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=dict,description="The initialization configuration of the agent",)
    instance: Optional[Any] = None
    
    code: Optional[str] = Field(default=None, description="Source code for dynamically generated agent classes (used when cls cannot be imported from a module)")

    function_calling: Optional[Dict[str, Any]] = Field(
        default=None, description="Default function calling representation"
    )
    text: Optional[str] = Field(
        default=None, description="Default text representation of the agent"
    )
    args_schema: Optional[Type[BaseModel]] = Field(
        default=None, description="Default args schema (BaseModel type)"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        
        result = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "version": self.version,
            "require_grad": self.require_grad,
            
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
            
            "function_calling": self.function_calling,
            "text": self.text,
            "args_schema": dynamic_manager.serialize_args_schema(self.args_schema) if self.args_schema else None,
        }
        
        return result
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Validate the model from a dictionary."""
        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata", {})
        version = data.get("version")
        require_grad = data.get("require_grad", False)
        
        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code, 
                        class_name=class_name,
                        base_class=Agent,
                        context="agent"
                    )
                except Exception as e:
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None
            
        config = data.get("config", {})
        instance = data.get("instance", None)

        function_calling = data.get("function_calling")
        text = data.get("text")
        args_schema = dynamic_manager.deserialize_args_schema(data.get("args_schema"))
        
        return cls(name=name, 
            description=description,
            metadata=metadata,
            version=version,
            require_grad=require_grad,
            cls=cls_, 
            config=config, 
            instance=instance, 
            function_calling=function_calling, 
            text=text, 
            args_schema=args_schema
        )

    def __str__(self) -> str:
        return (
            f"AgentConfig(name={self.name}, "
            f"description={self.description}, "
            f"require_grad={self.require_grad})"
        )

    def __repr__(self) -> str:
        return self.__str__()


def format_actions(actions: List[BaseModel]) -> str:
    """Format actions (tool/skill calls) as a Markdown table using pandas."""
    rows = []
    for action in actions:
        if isinstance(action.args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in action.args.items())
        else:
            args_str = str(action.args)

        rows.append({
            "Type": action.type if hasattr(action, "type") else "tool",
            "Name": action.name,
            "Args": args_str,
            "Output": action.output if hasattr(action, "output") and action.output is not None else None,
        })

    df = pd.DataFrame(rows)

    if df["Output"].isna().all():
        df = df.drop(columns=["Output"])
    else:
        df["Output"] = df["Output"].fillna("None")

    return df.to_markdown(index=True)


class ActionInputArgs(BaseModel):
    type: str = Field(default="tool", description='The type of this action: "tool" or "skill".')
    name: str = Field(description="The name of the tool or skill.")
    args: str = Field(description='The arguments as a JSON string. Must be a valid JSON object string. e.g., "{\"result\": \"D\", \"reasoning\": \"Step 1: ...\"}"')


class ThinkOutput(BaseModel):
    thinking: str = Field(
        description="A structured <think>-style reasoning block."
    )
    evaluation_previous_goal: str = Field(
        description="One-sentence analysis of your last action."
    )
    memory: str = Field(description="1-3 sentences of specific memory.")
    next_goal: str = Field(
        description="State the next immediate goals and actions."
    )
    actions: List[ActionInputArgs] = Field(
        description=(
            'The list of actions (tool or skill calls) to execute in sequence. '
            'Each action has a "type" ("tool" or "skill"), a "name", and "args" (JSON string). '
            'e.g., [{"type": "tool", "name": "done", "args": "{\"result\": \"D\"}"}, '
            '{"type": "skill", "name": "hello-world", "args": "{\"name\": \"Alice\"}"}]'
        )
    )

    def __str__(self) -> str:
        return (
            f"Thinking: {self.thinking}\n"
            f"Evaluation of Previous Goal: {self.evaluation_previous_goal}\n"
            f"Memory: {self.memory}\n"
            f"Next Goal: {self.next_goal}\n"
            f"Actions:\n{format_actions(self.actions)}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()

class Agent(BaseModel):
    """Base class for all agents, mirroring the design of `Tool`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the agent.")
    description: str = Field(description="The description of the agent.")
    metadata: Dict[str, Any] = Field(description="The metadata of the agent.")
    version: str = Field(default="1.0.0", description="Version of the agent")
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        require_grad: bool = False,
        use_memory: bool = True,
        use_todo: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Set default values
        self.name = name or self.name
        self.description = description or self.description
        self.metadata = metadata or self.metadata
        self.require_grad = require_grad

        # Set working directory
        self.workdir = workdir

        # Set prompt name and modules
        self.prompt_name = prompt_name
        self.memory_name = memory_name
        self.use_memory = use_memory
        self.model_name = model_name

        # Setup steps
        self.max_steps = max_steps if max_steps > 0 else int(1e8)
        self.max_tools = max_tools

        self.review_steps = review_steps
        self.use_todo = use_todo

    async def initialize(self) -> None:
        """Initialize the agent."""
        logger.info(f"| 📁 Agent working directory: {self.workdir}")

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.model_name}, prompt_name={self.prompt_name})"

    def __repr__(self) -> str:
        return self.__str__()

    async def _extract_file_content(self, file: str) -> Dict[str, Any]:
        """Extract file information and a short summary."""

        info = get_file_info(file)

        # Extract file content
        input_payload = {
            "name": "mdify",
            "input": {
                "file_path": file,
                "output_format": "markdown",
            },
        }
        tool_response = await tcp(**input_payload)
        file_content = tool_response.message

        # Use LLM to summarize the file content
        system_prompt = "You are a helpful assistant that summarizes file content."

        user_prompt = dedent(
            f"""
            Summarize the following file content as 1-3 sentences:
            {file_content}
        """
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        model_response = await model_manager(model=self.model_name, messages=messages)

        info["content"] = file_content
        info["summary"] = model_response.message

        return info

    async def _generate_enhanced_task(self, task: str, files: List[Dict[str, Any]]) -> str:
        """Generate enhanced task with attached file summaries."""

        attach_files_string = "\n".join(
            [f"File: {file['path']}\nSummary: {file['summary']}" for file in files]
        )

        enhanced_task = dedent(
            f"""
            - Task:
            {task}
            - Attach files:
            {attach_files_string}
        """)
        return enhanced_task

    async def _get_agent_context(self, 
                                 task: str,
                                 step_number: int = 0,
                                 ctx: SessionContext = None,
                                 **kwargs) -> Dict[str, Any]:
        """Get the agent context."""
        task = f"<task>{task}</task>"
        
        id = ctx.id if ctx else None

        step_info_description = (
            f"Step {step_number + 1} of {self.max_steps} max possible steps\n"
        )
        time_str = datetime.now().isoformat()
        step_info_description += f"Current date and time: {time_str}"
        step_info = dedent(f"""
            <step_info>
            {step_info_description}
            </step_info>
        """)

        # ===== 获取记忆系统状态(仅在启用use_memory时执行) =====
        memory = ""     # 初始化记忆上下文字符串
        if self.use_memory and self.memory_name:
            # 从记忆管理器中获取当前会话的记忆状态
            # 参数说明:
            # - name: 记忆系统名称,标识使用哪个记忆实例
            # - n: review_steps,指定获取最近N步的详细事件历史(默认5步)
            # - ctx: 会话上下文,用于隔离不同会话的记忆空间
            state = await memory_manager.get_state(
                name=self.memory_name,
                n=self.review_steps,
                ctx=ctx
            )

            # 从记忆状态中提取三个关键部分:

            # 1. events: 最近 N 步的详细事件列表
            #    每个事件包含: step_number, event_type, data(thinking, actions等)
            #    用于构建<agent_history>标签,让大模型了解近期执行细节
            events = state["events"]

            # 2. summaries: 历史对话的摘要信息
            #    由记忆系统自动生成的阶段性总结,压缩更早的历史
            #    用于构建<summaries>标签,提供长期记忆的概览
            summaries = state["summaries"]

            # 3. insights: 从历史中提取的关键洞察和经验教训
            #    记忆系统识别的重要模式、成功经验或失败教训
            #    用于构建<insights>标签,帮助大模型避免重复错误
            insights = state["insights"]
            
            # 构建代理历史记录的XML结构,包含所有已执行步骤的详细信息
            memory += "<agent_history>"

            # 遍历记忆系统中的历史事件,按步骤编号依次构建历史记录
            for event in events:
                # 为每个步骤创建独立的XML标签,便于大模型理解步骤边界
                memory += f"<step_{event.step_number}>\n"

                # 根据事件类型提取不同的信息
                if event.event_type == EventType.TASK_START:
                    # 任务开始事件:记录初始任务描述
                    # 优先使用'task'字段,如果不存在则回退到'message'字段
                    memory += f"Task Start: {event.data.get('task', event.data.get('message', ''))}\n"
                elif event.event_type == EventType.TASK_END:
                    # 任务结束事件:记录最终结果
                    memory += f"Task End: {event.data.get('result', '')}\n"
                elif event.event_type == EventType.TOOL_STEP:
                    # 工具执行步骤:记录完整的思考-行动循环信息
                    # 1. 对前一步目标的评价(成功/失败/不确定)
                    memory += f"Evaluation of Previous Step: {event.data.get('evaluation_previous_goal', '')}\n"

                    # 2. 当前步骤的记忆内容(用于跟踪进度和关键信息)
                    memory += f"Memory: {event.data.get('memory', '')}\n"

                    # 3. 下一步的目标和计划
                    memory += f"Next Goal: {event.data.get('next_goal', '')}\n"

                    # 4. 行动执行结果列表(包含所有tool/skill的调用和输出)
                    # 优先使用'actions'字段(新格式),如果不存在则回退到'tool'字段(旧格式)
                    memory += f"Action Results: {event.data.get('actions', event.data.get('tool', ''))}\n"
                # 每个步骤之间添加空行,提高可读性
                memory += "\n"
                # 关闭当前步骤的XML标签
                memory += f"</step_{event.step_number}>\n"

            # 关闭代理历史记录的根标签
            memory += "</agent_history>"
            
            # Generate memory
            memory += "<memory>"
            if len(summaries) > 0:
                memory += dedent(
                    f"""
                    <summaries>
                    {chr(10).join([str(summary) for summary in summaries])}
                    </summaries>
                """
                )
            else:
                memory += "<summaries>[Current summaries are empty.]</summaries>\n"
            if len(insights) > 0:
                memory += dedent(
                    f"""
                    <insights>
                    {chr(10).join([str(insight) for insight in insights])}
                    </insights>
                """
                )
            else:
                memory += "<insights>[Current insights are empty.]</insights>\n"
            memory += "</memory>"

        else:
            memory += "<agent_history>[Agent history is disabled.]</agent_history>\n"
            memory += "<memory>[Memory is disabled.]</memory>\n"

        if self.use_todo:
            todo = "<todo>"
            todo_tool = await tcp.get("todo")
            todo_contents = todo_tool.get_todo_content(ctx=ctx)
            todo += todo_contents
            todo += "</todo>"
        else:
            todo = "<todo>[Todo is disabled.]</todo>\n"

        agent_context = dedent(f"""
            <agent_context>
            {task}
            {step_info}
            {memory}
            {todo}
            </agent_context>
        """)

        return {
            "agent_context": agent_context,
        }

    async def _get_environment_context(self,
                                       ctx: SessionContext,
                                       **kwargs) -> Dict[str, Any]:
        """Get the environment state."""
        
        id = ctx.id
        
        environment_context = "<environment_context>"
        # Only iterate over environments specified in config, not all registered environments
        for env_name in config.env_names:
            env_info = await ecp.get_info(env_name)
            rule_string = env_info.rules
            rule_string = dedent(f"""
                <rules>
                {rule_string}
                </rules>
            """)

            env_state = await ecp.get_state(env_name, ctx=ctx)
            state_string = "<state>"
            state_string += env_state["state"]
            extra = env_state["extra"]

            if "screenshots" in extra:
                for screenshot in extra["screenshots"]:
                    state_string += (
                        f"\n<img src={screenshot.screenshot_path} "
                        f"alt={screenshot.screenshot_description}/>"
                    )
            state_string += "</state>"

            environment_context += dedent(f"""
                <{env_name}>
                {rule_string}
                {state_string}
                </{env_name}>
            """)

        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }

    async def _get_tool_context(self, ctx: SessionContext, **kwargs) -> Dict[str, Any]:
        """Get the tool context."""
        tool_context = "<tool_context>"

        tool_context += dedent(f"""
            <available_tools>
            {await tcp.get_contract()}
            </available_tools>
        """)

        tool_context += "</tool_context>"
        return {
            "tool_context": tool_context,
        }

    async def _get_skill_context(self, ctx: SessionContext, **kwargs) -> Dict[str, Any]:
        """Get the skill context from loaded skills via SCP.
        
        从技能上下文管理器(SCP)获取已加载技能的元数据摘要,
        并将其格式化为XML标签包裹的上下文字符串,用于注入到Agent的prompt中。
        
        Args:
            ctx: 会话上下文对象(当前未使用,保留接口一致性)
            **kwargs: 额外关键字参数(预留扩展)
            
        Returns:
            包含skill_context键的字典,值为格式化后的技能上下文字符串
            - 无技能时: "<skill_context>[No skills loaded.]</skill_context>\n"
            - 有技能时: "<skill_context>\n{每个技能的元数据摘要}\n</skill_context>"
        """
        # 通过SCP获取所有已加载技能的简要元数据(名称、描述、版本、路径等,非完整SKILL.md内容)
        skill_content = await scp.get_context()
        
        # 根据是否有技能加载,构建不同的skill_context字符串
        if not skill_content:
            # 无技能时返回占位提示
            skill_context = "<skill_context>[No skills loaded.]</skill_context>\n"
        else:
            # 有技能时将元数据摘要包裹在XML标签中
            skill_context = f"<skill_context>\n{skill_content}\n</skill_context>"
        
        return {
            "skill_context": skill_context,
        }

    async def _get_messages(self, 
                            task: str, 
                            ctx: SessionContext,
                            **kwargs) -> List[Message]:
        """Build system+agent messages using prompt templates and context.

        构建发送给大模型的完整消息列表,包括系统提示词和动态上下文。
        通过组合静态系统模块和动态代理消息模块,生成最终的prompt。

        Args:
            task: 用户任务描述字符串
            ctx: 会话上下文对象,包含历史对话、记忆等信息
            **kwargs: 额外关键字参数(预留扩展)

        Returns:
            包含system message和user message的消息列表,用于调用大模型API
        """

        # 构建系统提示词的静态模块(不随任务变化), max_tools: 最大工具调用次数限制, workdir: 工作目录路径
        system_modules = dict(max_tools=self.max_tools,workdir=self.workdir)

        # 构建代理消息的动态模块(随任务变化)
        agent_message_modules = dict(task=task)

        # 依次注入各类动态上下文到代理消息模块中

        # 1. 注入代理上下文:当前任务状态、历史记录、记忆、计划等
        agent_message_modules.update(await self._get_agent_context(task, ctx=ctx))

        # 2. 注入环境上下文:当前环境的配置、状态、可用资源等
        agent_message_modules.update(await self._get_environment_context(ctx=ctx))

        # 3. 注入工具上下文:已加载工具的列表、描述、使用规则等
        agent_message_modules.update(await self._get_tool_context(ctx=ctx))

        # 4. 注入技能上下文:已加载技能的元数据摘要(名称、描述、版本、路径等)
        # 注意:这里注入的是简要信息,不是完整的SKILL.md内容
        agent_message_modules.update(await self._get_skill_context(ctx=ctx))

        # 通过prompt管理器组装最终的消息列表
        # 将系统模块和代理模块填入Jinja2模板,生成system和user message
        messages = await prompt_manager.get_messages(
            prompt_name=self.prompt_name,         # 使用的prompt模板名称
            system_modules=system_modules,        # 静态系统模块变量
            agent_modules=agent_message_modules,  # 动态代理模块变量
        )

        return messages

    async def __call__(self, 
                       task: str, 
                       files: Optional[List[str]] = None,
                       ctx: Optional[SessionContext] = None,
                       **kwargs: Any,
                       ) -> AgentResponse:
        """Run the agent. This method should be implemented by the child classes."""
        raise NotImplementedError("__all__ method is not implemented by the child class")


class AgentExtra(BaseModel):
    """Agent extra data."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    file_path: Optional[Union[str, List[str]]] = Field(default=None, description="The file path of the extra data")
    data: Optional[Dict[str, Any]] = Field(default=None, description="The data of the extra data")
    parsed_model: Optional[BaseModel] = Field(default=None, description="The parsed model of the extra data")

class AgentResponse(BaseModel):
    """Agent response."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    success: bool = Field(description="Whether the agent has completed the task.")
    message: str = Field(description="The message of the agent.")
    extra: Optional[AgentExtra] = Field(default=None, description="The extra data of the agent.")

__all__ = [
    "InputArgs",
    "ACPErrorCode",
    "ACPError",
    "ACPRequest",
    "ACPResponse",
    "AgentConfig",
    "ActionInputArgs",
    "Agent",
    "AgentResponse",
    "ThinkOutput",
]
