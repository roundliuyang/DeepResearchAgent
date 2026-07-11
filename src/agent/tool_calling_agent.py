"""Tool calling agent implementation with manual agent logic."""

import asyncio
import os
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agent.types import Agent, AgentResponse, AgentExtra, ThinkOutput
from src.config import config
from src.logger import logger
from src.utils import dedent, parse_tool_args
from src.tool.server import tcp
from src.skill.server import scp
from src.environment.server import ecp
from src.memory import memory_manager, EventType
from src.tracer import Tracer, Record
from src.model import model_manager
from src.registry import AGENT
from src.session import SessionContext

@AGENT.register_module(force=True)
class ToolCallingAgent(Agent):
    """Tool calling agent implementation with manual agent logic."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="tool_calling", description="The name of the tool calling agent.")
    description: str = Field(default="A tool calling agent that can call tools to complete tasks.", description="The description of the tool calling agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool calling agent.")
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
        **kwargs
    ):
        # Set default prompt name for tool calling
        if not prompt_name:
            prompt_name = "tool_calling"
        
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            max_tools=max_tools,
            max_steps=max_steps,
            review_steps=review_steps,
            require_grad=require_grad,
            **kwargs)
    
    async def initialize(self):
        """Initialize the agent."""
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
        await super().initialize()
    
    async def _get_tracer_and_record(self) -> tuple[Tracer, Record]:
        """Get tracer and record for current call (coroutine-safe)."""
        tracer = Tracer()
        record = Record()
        
        if os.path.exists(self.tracer_save_path):
            await tracer.load_from_json(self.tracer_save_path)
            last_record = await tracer.get_last_record()
            if last_record:
                record = last_record
        
        return tracer, record
    
    async def _get_environment_context(self, ctx: SessionContext, record: Record = None, **kwargs) -> Dict[str, Any]:
        """Get the environment state."""
        
        environment_context = "<environment_context>"
        record_observation = {}
        
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
            record_observation[env_name] = extra
            
            if "screenshots" in extra:
                for screenshot in extra["screenshots"]:
                    state_string += f"\n<img src={screenshot.screenshot_path} alt={screenshot.screenshot_description}/>"
            state_string += "</state>"
            
            environment_context += dedent(f"""
                <{env_name}>
                {rule_string}
                {state_string}
                </{env_name}>
            """)
        
        if record is not None:
            record.observation = record_observation
        
        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }
        
    async def _get_tool_context(self, ctx: SessionContext, record: Record = None, **kwargs) -> Dict[str, Any]:
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
        
    async def _think_and_tool(self, 
                              messages: List[BaseMessage], 
                              task_id: str,
                              step_number: int,
                              record: Record = None, 
                              ctx: SessionContext = None, 
                              **kwargs)->Dict[str, Any]:
        """Think and tool calls for one step.

        执行一步完整的思考-行动循环:调用大模型获取思考和行动计划,然后依次执行这些行动。

        Args:
            messages: 包含系统提示词和对话历史的消息列表
            task_id: 任务唯一标识符
            step_number: 当前步骤编号(从0开始)
            record: 记录对象,用于存储当前步骤的执行数据
            ctx: 会话上下文对象
            **kwargs: 额外关键字参数

        Returns:
            包含执行结果的字典:
            - done (bool): 任务是否完成
            - result (str): 任务结果消息
            - reasoning (str): 完成任务的推理过程
        """

        # 初始化返回值的默认值
        done = False      # 任务完成标志
        result = None     # 任务结果
        reasoning = None  # 推理过程

        # 初始化记录数据结构,用于保存当前步骤的详细信息
        record_data = {
            "thinking": None,                     # 模型的思考过程
            "evaluation_previous_goal": None,     # 对前一步目标的评价
            "memory": None,                       # 记忆内容
            "next_goal": None,                    # 下一步目标
            "actions": [],                        # 执行的行动列表
        }
        
        try:
            # ===== 阶段1: 调用大模型进行思考 =====
            logger.info(f"| 🤖 Calling model: {self.model_name}")

            # 调用模型管理器,发送messages并期望返回ThinkOutput格式的响应
            think_output = await model_manager(
                model=self.model_name,        # 使用的模型名称
                messages=messages,            # 包含skill元数据的prompt
                response_format=ThinkOutput   # 期望的结构化输出格式
            )
            # 解析模型响应,提取Pydantic模型对象
            think_output = think_output.extra.parsed_model

            # 从结构化输出中提取各个字段
            thinking = think_output.thinking      # 模型的思考过程
            evaluation_previous_goal = think_output.evaluation_previous_goal   # 对前一步的评价
            memory = think_output.memory          # 记忆内容
            next_goal = think_output.next_goal    # 下一步目标
            actions = think_output.actions        # 要执行的行动列表

            # 将思考过程保存到记录中
            record_data["thinking"] = thinking
            record_data["evaluation_previous_goal"] = evaluation_previous_goal
            record_data["memory"] = memory
            record_data["next_goal"] = next_goal

            # 记录日志,便于调试和追踪
            logger.info(f"| 💭 Thinking: {thinking}")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Actions to execute: {actions}")
            
            # ===== 阶段2: 依次执行大模型返回的行动 =====
            action_results = []     # 存储每个行动的执行结果
            
            for i, action in enumerate(actions):
                # 解析行动的各个字段
                action_type = action.type            # 行动类型: "skill" 或 "tool"
                action_name = action.name            # 行动名称(技能名或工具名)
                action_args_str = action.args        # 行动参数字符串(JSON格式)
                action_args = parse_tool_args(action_args_str) if action_args_str else {}   # 解析为字典

                logger.info(f"| 📝 Action {i+1}/{len(actions)}: [{action_type}] {action_name}")
                logger.info(f"| 📝 Args: {action_args}")

                # 根据行动类型路由到不同的处理器
                if action_type == "skill":
                    # ===== 路由到技能处理器(SCP) =====
                    response = await scp(
                        name=action_name,    # 技能名称
                        input=action_args,   # 技能输入参数
                        ctx=ctx,             # 会话上下文
                    )
                    # 技能执行结果
                    action_result = response.message
                    action_extra = response.extra if hasattr(response, 'extra') else None

                    # 记录技能执行完成日志
                    logger.info(f"| ✅ Skill '{action_name}' completed (success={response.success})")
                    logger.info(f"| 📄 Result: {str(action_result)[:500]}")

                    # 构建行动字典,包含原始行动信息和执行结果
                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    # 构建记录额外信息
                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra['extra'] = action_extra.model_dump()
                    record_data["actions"].append(record_extra)

                else:
                    # ===== 路由到工具处理器(TCP,默认处理type=="tool") =====
                    tool_response = await tcp(
                        name=action_name,     # 工具名称
                        input=action_args,    # 工具输入参数
                        ctx=ctx,              # 会话上下文
                    )
                    # 工具执行结果
                    action_result = tool_response.message
                    action_extra = tool_response.extra if hasattr(tool_response, 'extra') else None

                    logger.info(f"| ✅ Tool '{action_name}' completed")
                    logger.info(f"| 📄 Result: {str(action_result)}")

                    # 构建行动字典,包含原始行动信息和执行结果
                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra['extra'] = action_extra.model_dump()
                    record_data["actions"].append(record_extra)

                    # 检查是否是"done"工具,如果是则标记任务完成并退出循环
                    if action_name == "done":
                        done = True
                        result = action_result
                        # 从extra中提取reasoning字段(如果存在)
                        reasoning = action_extra.data.get('reasoning', None) if action_extra and action_extra.data else None
                        break    # 提前退出,不再执行后续行动

            # ===== 阶段3: 构建事件数据并记录 =====
            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "actions": action_results    # 包含所有行动的执行结果
            }

            # 将当前步骤的工具调用数据保存到record对象
            if record is not None:
                record.tool = record_data
            
            # 获取记忆系统名称
            memory_name = self.memory_name
            
            # 如果启用了记忆功能,将当前步骤的事件添加到记忆系统
            if self.use_memory and memory_name:
                # 第 N 步执行完行动后: 通过 memory_manager.add_event 存入记忆系统
                # 第 N + 1 步开始前,从记忆系统读取最近 N 步的历史事件, 将 event.data.get('actions') 注入到 <agent_history> 标签中
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=step_number,
                    event_type=EventType.TOOL_STEP,    # 事件类型:工具步骤
                    data=event_data,                   # 事件数据:包含思考和行动结果
                    agent_name=self.name,
                    task_id=task_id,
                    ctx=ctx
                )
            
        except Exception as e:
            # 捕获并记录异常,避免整个步骤失败
            logger.error(f"| Error in thinking and tool step: {e}")

        # 构造并返回响应字典
        response_dict = {
            "done": done,
            "result": result,
            "reasoning": reasoning
        }
        return response_dict
        
    async def __call__(self, 
                  task: str, 
                  files: Optional[List[str]] = None,
                  **kwargs
                  ) -> AgentResponse:
        """
        Main entry point for tool calling agent through acp.
        
        Args:
            task (str): The task to complete.
            files (Optional[List[str]]): The files to attach to the task.
            
        Returns:
            AgentResponse: The response of the agent.
        """
        # 记录Agent启动日志,包含任务描述
        logger.info(f"| 🚀 Starting ToolCallingAgent: {task}")

        # 从kwargs中获取会话上下文,如果未提供则创建新的SessionContext
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()

        # 创建追踪器(tracer)和记录(record),用于记录执行过程
        # 如果存在历史tracer文件,会自动加载并恢复最后一条记录
        tracer, record = await self._get_tracer_and_record()

        # 处理附件文件:提取内容并生成增强版任务描述
        if files:
            logger.info(f"| 📂 Attached files: {files}")
            # 并行提取所有文件的内容
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            # 将文件内容整合到任务描述中,使任务更明确
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            # 无附件时直接使用原始任务
            enhanced_task = task
        
        # 获取记忆系统名称(用于后续记忆管理)
        memory_name = self.memory_name

        # 生成唯一任务ID,格式: task_YYYYMMDD-HHMMSS
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        logger.info(f"| 📝 Context ID: {ctx.id}, Task ID: {task_id}")
        
        # ===== 记忆会话管理(仅在启用use_memory时执行) =====
        if self.use_memory and memory_name:
            # 启动记忆会话,为该ctx创建独立的记忆空间
            await memory_manager.start_session(memory_name=memory_name, ctx=ctx)
            
            # 记录任务开始事件到记忆系统
            await memory_manager.add_event(
                memory_name=memory_name,           # 记忆系统名称
                step_number=0,                     # 步骤编号(0表示开始)
                event_type=EventType.TASK_START,   # 事件类型:任务开始
                data=dict(task=enhanced_task),     # 事件数据:增强后的任务描述
                agent_name=self.name,              # Agent名称
                task_id=task_id,                   # 任务ID
                ctx=ctx                            # 会话上下文
            )
        else:
            logger.info(f"| ⏭️ Memory disabled (use_memory={self.use_memory}), skipping session management")
        
        # ===== 初始化消息列表 =====
        # 构建发送给大模型的初始消息,包含系统提示词和动态上下文(技能、工具、环境等)
        messages = await self._get_messages(enhanced_task, ctx=ctx)
        
        # ===== 主循环:执行思考-行动步骤 =====
        step_number = 0      # 步骤计数器
        
        while step_number < self.max_steps:
            # 记录当前步骤进度
            logger.info(f"| 🔄 Step {step_number+1}/{self.max_steps}")
            
            # 执行一步思考与工具调用
            response = await self._think_and_tool(messages, task_id, step_number, ctx=ctx, record=record)
            step_number += 1

            # ===== 更新追踪器并持久化 =====
            # 将当前步骤的观察结果和工具调用记录添加到tracer
            await tracer.add_record(observation=record.observation, 
                                        tool=record.tool,
                                        task_id=task_id,
                                        ctx=ctx)
            # 保存tracer到JSON文件,支持断点续跑
            await tracer.save_to_json(self.tracer_save_path)

            # 注意:记忆已在add_event()中自动保存,无需额外操作

            # 重新构建消息列表,包含最新的执行历史和上下文
            messages = await self._get_messages(enhanced_task, ctx=ctx)

            # 如果任务已完成(done=True),退出循环
            if response["done"]:
                break
        
        # ===== 处理达到最大步骤数的情况 =====
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "result": "The task has not been completed.",
                "reasoning": "Reached the maximum number of steps."
            }
        
        # Get memory system name
        memory_name = self.memory_name
        
        # Add task end event and end session (only if use_memory is enabled)
        if self.use_memory and memory_name:
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=step_number,
                event_type=EventType.TASK_END,
                data=response,
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx
            )
            
            # End session (automatically saves memory to JSON)
            await memory_manager.end_session(memory_name=memory_name, ctx=ctx)
        
        # 最终保存tracer,确保所有记录都已持久化
        await tracer.save_to_json(self.tracer_save_path)

        # 记录Agent完成日志,显示实际执行步数
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")

        # 构造并返回Agent响应
        return AgentResponse(
            success=response["done"],           # 任务是否成功完成
            message=response["result"],         # 任务执行结果消息
            extra=AgentExtra(
                data=response                   # 额外数据:包含done、result、reasoning
            )
        )