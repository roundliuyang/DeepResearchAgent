"""ACP Server (Agent Context Protocol Server)

全局单例的 Agent 管理服务端，采用懒加载模式。
对外暴露统一的注册/查询/调用接口，内部委托给 AgentContextManager 处理
实际的生命周期管理、版本控制和向量检索。

架构分层：
  ACPServer (对外门面)
    └── AgentContextManager (实际逻辑)
          ├── _agent_configs       — 当前活跃版本注册表 {name: AgentConfig}
          └── _agent_history_versions — 版本历史 {name: {version: AgentConfig}}
"""

import os
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.logger import logger
from src.agent.types import AgentConfig, Agent
from src.agent.context import AgentContextManager
from src.utils import assemble_project_path

class ACPServer(BaseModel):
    """ACP Server for managing agent registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents")
    contract_path: str = Field(default=None, description="The path to save the agent contract")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the ACP Server."""
        super().__init__(**kwargs)
        # 注册表缓存：将 ContextManager 中已激活的 AgentConfig 镜像到 Server 层，
        # 供 Server 层方法快速查询，避免每次都穿透到 ContextManager
        self._registered_configs: Dict[str, AgentConfig] = {}  # agent_name -> AgentConfig

        
    # ======================================================================
    # 注册与初始化
    # ======================================================================
    async def initialize(self, agent_names: Optional[List[str]] = None):
        """按 agent_names 并发加载所有 Agent 实例，并同步注册表。
        
        完整流程（委托给 AgentContextManager.initialize）：
        1. 从 AGENT 注册表发现所有已 @AGENT.register_module 的 Agent 类
        2. 从持久化 JSON（agent.json）加载历史版本
        3. 合并两者，取版本号更高者作为当前活跃版本
        4. 按 agent_names 过滤后并发构建实例（gather_with_concurrency）
        5. 持久化 agent.json 和 contract.md
        
        Args:
            agent_names: 需要初始化的 Agent 名称列表；为 None 时初始化全部
        """
        # 确定持久化目录：workdir/agent/
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "agent.json")       # Agent 配置快照
        self.contract_path = os.path.join(self.base_dir, "contract.md")   # Agent 契约文档（供 Planner 阅读）
        logger.info(f"| 📁 ACP Server base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")
        
        # 创建 AgentContextManager 实例，注入 embedding 模型用于 FAISS 向量检索
        self.agent_context_manager = AgentContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
            model_name="openrouter/gemini-3-flash-preview",
            embedding_model_name="openrouter/text-embedding-3-large",
        )
        # 执行实际的并发初始化（注册表发现 + JSON 加载 + 实例构建）
        await self.agent_context_manager.initialize(agent_names=agent_names)
        
        # 将 ContextManager 的活跃注册表镜像到 Server 层的 _registered_configs
        agent_list = await self.agent_context_manager.list()
        for agent_name in agent_list:
            agent_config = await self.agent_context_manager.get_info(agent_name)
            if agent_config and agent_name not in self._registered_configs:
                self._registered_configs[agent_name] = agent_config
        
        logger.info("| ✅ Agents initialization completed")
        
    # ======================================================================
    # CRUD 操作
    # ======================================================================
    async def get_contract(self) -> str:
        """获取所有 Agent 的契约文档（contract.md 内容，供 Planner 了解可用 Agent）"""
        return await self.agent_context_manager.load_contract()
        
    async def register(self, 
                       agent_cls: Type[Agent],
                       agent_config_dict: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None) -> AgentConfig:
        """【Create】注册新 Agent：实例化 → 构建 AgentConfig → 写入版本历史 → 持久化
        
        Args:
            agent_cls: 待注册的 Agent 类
            agent_config_dict: 初始化配置；为 None 时从全局 config 按类名取
            override: 是否覆盖已有注册
            version: 指定版本号；为 None 时由 version_manager 自动生成
            
        Returns:
            AgentConfig: 注册后的 Agent 配置
        """
        # 委托给 ContextManager 执行实际注册（含实例化、FAISS 入库、JSON 持久化）
        agent_config = await self.agent_context_manager.register(
            agent_cls, 
            agent_config_dict=agent_config_dict, 
            override=override,
            version=version
        )
        # 同步更新 Server 层注册表缓存
        self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    async def get_info(self, agent_name: str) -> Optional[AgentConfig]:
        """【Read】按名称查询 Agent 完整配置（AgentConfig），找不到返回 None"""
        return await self.agent_context_manager.get_info(agent_name)
    
    async def list(self) -> List[str]:
        """【Read】列出所有已注册 Agent 的名称"""
        return await self.agent_context_manager.list()
    
    async def get(self, agent_name: str) -> Optional[Agent]:
        """【Read】按名称获取 Agent 运行时实例，找不到返回 None"""
        agent = await self.agent_context_manager.get(agent_name)
        return agent
    
    async def cleanup(self):
        """清理所有 Agent：清空 ContextManager 内部状态 + Server 层注册表缓存"""
        await self.agent_context_manager.cleanup()
        self._registered_configs.clear()
    
    async def update(self, 
                     agent_cls: Type[Agent],
                     agent_config_dict: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> AgentConfig:
        """【Update】更新已有 Agent：用新 class/config 创建新版本并写入历史
        
        Args:
            agent_cls: 新实现的 Agent 类
            agent_config_dict: 新配置；为 None 时从全局 config 按类名取
            new_version: 新版本号；为 None 时自动 patch 递增（如 1.0.0 → 1.0.1）
            description: 版本更新说明
            
        Returns:
            AgentConfig: 更新后的 Agent 配置
        """
        # 委托给 ContextManager 执行版本更新（含实例化、FAISS 更新、JSON 持久化）
        agent_config = await self.agent_context_manager.update(
            agent_cls, agent_config_dict=agent_config_dict, new_version=new_version, description=description
        )
        # 同步更新 Server 层注册表缓存
        self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    async def copy(self, 
                  agent_name: str,
                  new_name: Optional[str] = None, 
                  new_version: Optional[str] = None, 
                  new_config: Optional[Dict[str, Any]] = None) -> AgentConfig:
        """【Create（副本）】复制已有 Agent：可改名/改配置，自动生成新版本
        
        Args:
            agent_name: 源 Agent 名称
            new_name: 新名称；为 None 时复用原名（此时递增版本号）
            new_version: 新版本号；为 None 时自动生成
            new_config: 合并到原配置上的新配置 dict
            
        Returns:
            AgentConfig: 复制后的新 Agent 配置
        """
        # 委托给 ContextManager 执行复制（含实例化、FAISS 入库、JSON 持久化）
        agent_config = await self.agent_context_manager.copy(
            agent_name, new_name, new_version, new_config
        )
        # 同步更新 Server 层注册表缓存
        self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    async def unregister(self, agent_name: str) -> bool:
        """【Delete】注销 Agent：从活跃注册表中移除（版本历史保留）
        
        Args:
            agent_name: 待注销的 Agent 名称
            
        Returns:
            True 表示注销成功，False 表示 Agent 不存在
        """
        # 委托给 ContextManager 执行注销（含 JSON 持久化）
        success = await self.agent_context_manager.unregister(agent_name)
        # 同步清理 Server 层注册表缓存
        if success and agent_name in self._registered_configs:
            del self._registered_configs[agent_name]
        return success
    
    async def restore(self, agent_name: str, version: str, auto_initialize: bool = True) -> Optional[AgentConfig]:
        """【版本回滚】将 Agent 回滚到指定历史版本，并设为当前活跃版本
        
        Args:
            agent_name: Agent 名称
            version: 目标版本字符串（如 "1.0.0"）
            auto_initialize: 是否自动构建实例
            
        Returns:
            恢复后的 AgentConfig；找不到该版本时返回 None
        """
        agent_config = await self.agent_context_manager.restore(agent_name, version, auto_initialize)
        if agent_config:
            self._registered_configs[agent_config.name] = agent_config
        return agent_config
    
    # ======================================================================
    # 向量检索（基于 FAISS 的语义相似度匹配）
    # ======================================================================
    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """按语义查询相似 Agent（用于 Planner 动态选择最合适的 Agent）
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            包含 Agent 信息及相似度分数的字典列表
        """
        return await self.agent_context_manager.retrieve(query=query, k=k)
    
    # ======================================================================
    # 优化器变量接口（将 Agent 源码暴露为可训练变量，供 optimizer 修改）
    # ======================================================================
    async def get_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取 Agent 源码作为 Variable.value，供 optimizer 读取
        
        Args:
            agent_name: 指定 Agent；为 None 时返回全部
            
        Returns:
            {agent_name: Variable}，Variable.variables 字段存储源码字符串
        """
        return await self.agent_context_manager.get_variables(agent_name=agent_name)
    
    async def get_trainable_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """获取可训练（require_grad=True）的 Agent 变量，过滤掉冻结的 Agent
        
        Args:
            agent_name: 指定 Agent；为 None 时返回全部可训练 Agent
            
        Returns:
            {agent_name: Variable}，仅包含 require_grad=True 的 Agent
        """
        return await self.agent_context_manager.get_trainable_variables(agent_name=agent_name)
    
    async def set_variables(self, agent_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> AgentConfig:
        """通过新源码更新 Agent，自动创建新版本（optimizer 的写回入口）
        
        Args:
            agent_name: 待更新的 Agent 名称
            variable_updates: 格式为 {"variables": "新源码字符串"}
            new_version: 新版本号；为 None 时自动 patch 递增
            description: 版本更新说明
            
        Returns:
            AgentConfig: 更新后的配置
        """
        updated_config = await self.agent_context_manager.set_variables(
            agent_name=agent_name, 
            variable_updates=variable_updates, 
            new_version=new_version, 
            description=description
        )
        self._registered_configs[updated_config.name] = updated_config
        return updated_config

    # ======================================================================
    # 调用转发
    # ======================================================================
    async def __call__(self, name: str, input: Dict[str, Any], **kwargs) -> Any:
        """按名称调用 Agent，委托 ContextManager 查找实例并执行
        
        Args:
            name: Agent 名称
            input: 传给 Agent 的输入参数
            **kwargs: 透传给 Agent 的额外参数（如 ctx）
            
        Returns:
            Agent 执行结果
        """
        return await self.agent_context_manager(name, input, **kwargs)


# 全局单例：整个进程共享同一个 ACPServer，各模块通过 from src.agent import acp 引用
acp = ACPServer()
