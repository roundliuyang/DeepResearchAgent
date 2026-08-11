"""Agent Context Manager for managing agent lifecycle and resources with lazy loading."""
import asyncio
import os
from asyncio_atexit import register as async_atexit_register
from typing import Any, Dict, List, Type, Optional, Union, Tuple, TYPE_CHECKING
from datetime import datetime
import inflection
import json
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from src.logger import logger
from src.config import config
from src.environment.faiss.service import FaissService
from src.environment.faiss.types import FaissAddRequest
from src.utils import (
    assemble_project_path,
    gather_with_concurrency,
    file_lock,
    generate_unique_id
)
from src.agent.types import Agent, AgentConfig
from src.session import SessionContext
from src.version import version_manager
from src.dynamic import dynamic_manager
from src.registry import AGENT


class AgentContextManager(BaseModel):
    """Global context manager for all agents with lazy loading and version history."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    base_dir: str = Field(default=None, description="The base directory to use for the agents")
    save_path: str = Field(default=None, description="The path to save the agents configuration JSON")
    contract_path: str = Field(default=None, description="The path to save the agent contract")

    def __init__(
        self,
        base_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        contract_path: Optional[str] = None,
        model_name: str = "openrouter/gemini-3-flash-preview",
        embedding_model_name: str = "openrouter/text-embedding-3-large",
        **kwargs: Any,
    ):
        """Initialize the agent context manager.

        Args:
            base_dir: Base directory for storing agent data
            save_path: Path to save agent configurations
            contract_path: Path to save agent contract
            model_name: The model name used for embedding text (via FaissService)
            embedding_model_name: The embedding model name (kept for parity with tools)
        """
        super().__init__(**kwargs)

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(config.workdir, "agent"))
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| 📁 Agent context manager base directory: {self.base_dir}.")
        if save_path is not None:
            self.save_path = assemble_project_path(save_path)
        else:
            self.save_path = os.path.join(self.base_dir, "agent.json")
        logger.info(f"| 📁 Agent context manager save path: {self.save_path}.")
        if contract_path is not None:
            self.contract_path = assemble_project_path(contract_path)
        else:
            self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Agent context manager contract path: {self.contract_path}.")

        # Current active configs (latest version)
        self._agent_configs: Dict[str, AgentConfig] = {}
        # Agent version history, e.g., {"agent_name": {"1.0.0": AgentConfig, ...}}
        self._agent_history_versions: Dict[str, Dict[str, AgentConfig]] = {}

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        self._cleanup_registered = False
        self._faiss_service: Optional[FaissService] = None
        self._variables_lock = asyncio.Lock()  # Lock for get/set trainable variables

    async def initialize(self, agent_names: Optional[List[str]] = None) -> None:
        """初始化所有已注册 Agent，完整流程：
        1. 向 dynamic_manager 注册 Agent 相关符号（供动态代码注入使用）
        2. 初始化 FAISS 向量检索服务
        3. 从 AGENT 注册表加载所有已注册 Agent 类（_load_from_registry）
        4. 从持久化 JSON 加载历史版本（_load_from_code）
        5. 合并两者：取版本号更高者作为当前活跃版本
        6. 按 agent_names 过滤后并发构建实例（gather_with_concurrency）
        7. 持久化 agent.json + contract.md
        """

        # ---- 步骤 1: 注册 Agent 相关符号到 dynamic_manager ----
        # 动态加载的代码（如 optimizer 修改后的 Agent）需要这些符号才能正确 import
        dynamic_manager.register_symbol("AGENT", AGENT)
        dynamic_manager.register_symbol("Agent", Agent)
        dynamic_manager.register_symbol("AgentConfig", AgentConfig)

        # 注册 context provider，使动态代码能通过 inject 自动获得 Agent 相关导入
        def agent_context_provider():
            return {
                "AGENT": AGENT,
                "Agent": Agent,
                "AgentConfig": AgentConfig,
            }

        dynamic_manager.register_context_provider("agent", agent_context_provider)

        # ---- 步骤 2: 初始化 FAISS 服务，用于 Agent 语义检索 ----
        self._faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name,
        )

        # ---- 步骤 3: 从 AGENT 注册表加载所有已 @AGENT.register_module 装饰的类 ----
        agent_configs: Dict[str, AgentConfig] = {}
        registry_agent_configs: Dict[str, AgentConfig] = await self._load_from_registry()
        agent_configs.update(registry_agent_configs)

        # ---- 步骤 4: 从 agent.json 加载历史版本（含动态生成的 Agent）----
        code_agent_configs: Dict[str, AgentConfig] = await self._load_from_code()

        # ---- 步骤 5: 合并注册表与 JSON，取版本号更高者 ----
        for agent_name, code_config in code_agent_configs.items():
            if agent_name in agent_configs:
                registry_config = agent_configs[agent_name]
                # 只有当 JSON 版本严格大于注册表版本时才覆盖
                if (
                    version_manager.compare_versions(
                        code_config.version, registry_config.version
                    )
                    > 0
                ):
                    logger.info(
                        f"| 🔄 Overriding agent {agent_name} from registry "
                        f"(v{registry_config.version}) with code version (v{code_config.version})"
                    )
                    agent_configs[agent_name] = code_config
                else:
                    logger.info(
                        f"| 📌 Keeping agent {agent_name} from registry (v{registry_config.version}), "
                        f"code version (v{code_config.version}) is not greater"
                    )
                    # 版本相同时将注册表配置（持有真实类引用，非动态类）写入历史
                    if version_manager.compare_versions(code_config.version, registry_config.version) == 0:
                        if agent_name in self._agent_history_versions:
                            self._agent_history_versions[agent_name][registry_config.version] = registry_config
            else:
                agent_configs[agent_name] = code_config

        # ---- 步骤 6: 按 agent_names 过滤并并发构建实例 ----
        if agent_names is not None:
            agent_configs = {name: agent_configs[name] for name in agent_names if name in agent_configs}

        # 并发构建，限制并发数为 10，避免同时初始化过多 Agent 压垂 LLM API
        names = list(agent_configs.keys())
        tasks = [self.build(agent_configs[name]) for name in names]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )

        # 将成功构建的 Agent 存入活跃注册表，失败的记录错误后跳过
        for agent_name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Failed to initialize agent {agent_name}: {result}")
                continue
            self._agent_configs[agent_name] = result
            logger.info(f"| 🎮 Agent {agent_name} initialized")

        # ---- 步骤 7: 持久化 agent.json + contract.md ----
        await self.save_to_json()
        await self.save_contract(agent_names=agent_names)

        # 注册进程退出时的清理回调，确保资源释放
        async_atexit_register(self.cleanup)
        self._cleanup_registered = True

        logger.info("| ✅ Agents initialization completed")

    async def _load_from_registry(self) -> Dict[str, AgentConfig]:
        """从 AGENT 注册表加载所有已注册 Agent。
        
        遍历 AGENT._module_dict（由 @AGENT.register_module() 装饰器填充），
        对每个 Agent 类：
          1. 从全局 config 取对应配置项（按 snake_case 类名查找）
          2. 提取名称/描述/源码/参数等元数据
          3. 构建 AgentConfig 并存入 _agent_history_versions
        """

        agent_configs: Dict[str, AgentConfig] = {}

        async def register_agent_class(agent_cls: Type[Agent]):
            """Register an agent class synchronously.
            
            Args:
                agent_cls: Agent class to register
            """
            try:
                # Get agent config from global config
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})
                agent_require_grad = agent_config_dict.get("require_grad", False) if agent_config_dict and "require_grad" in agent_config_dict else False
                
                # Get agent properties from agent class
                agent_name = agent_cls.model_fields['name'].default
                agent_description = agent_cls.model_fields['description'].default
                agent_metadata = agent_cls.model_fields['metadata'].default
                
                # Get or generate version from version_manager
                agent_version = await version_manager.get_version("agent", agent_name)
                
                # Get full module source code
                agent_code = dynamic_manager.get_full_module_source(agent_cls)
                
                agent_parameters = dynamic_manager.get_parameters(agent_cls)
                agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
                agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
                agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)
                
                # Create agent config (AgentConfig.id is auto-incremented internally if needed)
                agent_config = AgentConfig(
                    name=agent_name,
                    description=agent_description,
                    version=agent_version,
                    require_grad=agent_require_grad,
                    cls=agent_cls,
                    config=agent_config_dict,
                    instance=None,
                    function_calling=agent_function_calling,
                    text=agent_text,
                    args_schema=agent_args_schema,
                    metadata=agent_metadata,
                    code=agent_code,
                )
                
                # Store agent config
                agent_configs[agent_name] = agent_config
                
                # Store in version history (by version string)
                if agent_name not in self._agent_history_versions:
                    self._agent_history_versions[agent_name] = {}
                self._agent_history_versions[agent_name][agent_version] = agent_config
                
                # Register version to version manager
                await version_manager.register_version("agent", agent_name, agent_version)
                
                logger.info(f"| 📝 Registered agent: {agent_name} ({agent_cls.__name__})")
                
            except Exception as e:
                logger.error(f"| ❌ Failed to register agent class {agent_cls.__name__}: {e}")
                raise

        import src.agent  # noqa: F401

        agent_classes = list(AGENT._module_dict.values())
        logger.info(f"| 🔍 Discovering {len(agent_classes)} agents from AGENT registry")

        tasks = [register_agent_class(agent_cls) for agent_cls in agent_classes]
        results = await gather_with_concurrency(
            tasks, max_concurrency=10, return_exceptions=True
        )
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(
            f"| ✅ Discovered and registered {success_count}/{len(agent_classes)} agents from AGENT registry"
        )

        return agent_configs

    async def _load_from_code(self):
        """从持久化 agent.json 加载所有 Agent 历史版本。
        
        JSON 结构：
        {
            "metadata": { "saved_at", "num_agents", "num_versions" },
            "agents": {
                "agent_name": {
                    "current_version": "1.0.0",
                    "versions": {
                        "1.0.0": { ...AgentConfig fields... },
                        ...
                    }
                }
            }
        }
        返回：{agent_name: AgentConfig}，仅包含各 Agent 的当前版本
        """
        
        agent_configs: Dict[str, AgentConfig] = {}
        
        # If save file does not exist yet, nothing to load
        if not os.path.exists(self.save_path):
            logger.info(f"| 📂 Agent config file not found at {self.save_path}, skipping code-based loading")
            return agent_configs
        
        # Load all agent configs from json file
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"| ⚠️ Failed to parse agent config JSON from {self.save_path}: {e}")
            return agent_configs
        
        metadata = load_data.get("metadata", {})
        agents_data = load_data.get("agents", {})

        async def register_agent_class(agent_name: str, agent_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, AgentConfig], Optional[AgentConfig]]]:
            """Load all versions for a single agent from JSON."""
            try:
                current_version = agent_data.get("current_version", "1.0.0")
                versions = agent_data.get("versions", {})
                
                if not versions:
                    logger.warning(f"| ⚠️ Agent {agent_name} has no versions")
                    return None
                
                version_map: Dict[str, AgentConfig] = {}
                current_agent_config: Optional[AgentConfig] = None
                
                for _, version_data in versions.items():
                    agent_config = AgentConfig.model_validate(version_data)
                    version = agent_config.version
                    version_map[version] = agent_config
                    
                    if version == current_version:
                        current_agent_config = agent_config
                
                return agent_name, version_map, current_agent_config
            except Exception as e:
                logger.error(f"| ❌ Failed to load agent {agent_name} from code JSON: {e}")
                return None

        # Launch loading of each agent concurrently with a concurrency limit
        tasks = [
            register_agent_class(agent_name, agent_data) for agent_name, agent_data in agents_data.items()
        ]
        results = await gather_with_concurrency(tasks, max_concurrency=10, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            agent_name, version_map, current_agent_config = result
            if not version_map:
                continue
            # Store all versions in history (mapped by version string)
            self._agent_history_versions[agent_name] = version_map
            # Active config: the one corresponding to current_version
            if current_agent_config is not None:
                agent_configs[agent_name] = current_agent_config
            else:
                # Fallback: if current_version is not found, use the last available version
                logger.warning(f"| ⚠️ Agent {agent_name} current_version not found, using last available version")
                agent_configs[agent_name] = list(version_map.values())[-1]
            
            # Register all versions to version manager
            for agent_config in version_map.values():
                await version_manager.register_version("agent", agent_name, agent_config.version)
            
        logger.info(f"| 📂 Loaded {len(agent_configs)} agents from {self.save_path}")
        return agent_configs

    async def _store(self, agent_config: AgentConfig):
        """Add agent information to the embedding index.
        
        Args:
            agent_config: Agent configuration
        """
        if self._faiss_service is None:
            return
            
        try:
            # Create comprehensive text representation
            agent_text = f"Agent: {agent_config.name}\nDescription: {agent_config.description}"
            
            # Add to FAISS index
            request = FaissAddRequest(
                texts=[agent_text],
                metadatas=[{
                    "name": agent_config.name,
                    "description": agent_config.description
                }]
            )
            
            await self._faiss_service.add_documents(request)
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add agent {agent_config.name} to FAISS index: {e}")

    async def build(self, agent_config: AgentConfig) -> AgentConfig:
        """根据 AgentConfig 创建 Agent 运行时实例。
        
        1. 若已有活跃实例则直接复用（幂等性保护）
        2. 否则用 cls(**config) 实例化，并调用 initialize()（如存在）
        3. 将实例挂载到 AgentConfig.instance 并存入 _agent_configs
        
        Args:
            agent_config: Agent 配置（必须包含 cls 字段）
            
        Returns:
            AgentConfig: 挂载了 instance 的配置
        """
        if agent_config.name in self._agent_configs:
            existing_config = self._agent_configs[agent_config.name]
            if existing_config.instance is not None:
                return existing_config
        
        # Create new agent instance
        try:
            # cls should already be loaded (either from registry or from code in _load_from_code)
            if agent_config.cls is None:
                raise ValueError(f"Cannot create agent {agent_config.name}: no class provided. Class should be loaded during initialization.")
            
            # Instantiate agent instance
            agent_instance = agent_config.cls(**agent_config.config) if agent_config.config else agent_config.cls()
            
            # Initialize agent if it has an initialize method
            if hasattr(agent_instance, "initialize"):
                await agent_instance.initialize()
            
            agent_config.instance = agent_instance
            
            # Store agent metadata
            self._agent_configs[agent_config.name] = agent_config
            
            logger.info(f"| 🔧 Agent {agent_config.name} created and stored")
            
            return agent_config
        except Exception as e:
            logger.error(f"| ❌ Failed to create agent {agent_config.name}: {e}")
            raise

    async def register(
        self,
        agent_cls: Type[Agent],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        override: bool = False,
        version: Optional[str] = None,
    ) -> AgentConfig:
        """注册新 Agent（运行时动态注册，区别于启动时的注册表加载）。
        
        流程：
        1. 实例化 Agent 类
        2. 提取源码/参数，构建 function_calling / text / args_schema
        3. 构建 AgentConfig 并写入 _agent_configs + _agent_history_versions
        4. 注册版本号 + 入 FAISS 索引 + 持久化 JSON + 更新 contract.md
        """
        
        try:
            if agent_config_dict is None:
                # Fallback to global config by class name
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})
            
            # Instantiate agent immediately (register is a runtime operation)
            try:
                agent_instance = agent_cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {agent_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__} with provided config: {e}")
            
            agent_name = agent_instance.name
            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict and "require_grad" in agent_config_dict else agent_instance.require_grad
            
            # Get or generate version from version_manager
            if version is None:
                agent_version = await version_manager.get_version("agent", agent_name)
            else:
                agent_version = version
                
            # Get agent code
            agent_code = dynamic_manager.get_source_code(agent_cls)
            if not agent_code:
                logger.warning(f"| ⚠️ Agent {agent_name} is dynamic but source code cannot be extracted")
            
            # Get agent parameters
            agent_parameters = dynamic_manager.get_parameters(agent_cls)
            agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)
            
            # --- Build AgentConfig ---
            agent_config = AgentConfig(
                name=agent_name,
                description=agent_description,
                metadata=agent_metadata,
                version=agent_version,
                require_grad=agent_require_grad,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )
            
            # --- Persist current config and history ---
            self._agent_configs[agent_name] = agent_config
            
            # Store in dict-based history (for quick lookup by version)
            if agent_name not in self._agent_history_versions:
                self._agent_history_versions[agent_name] = {}
            self._agent_history_versions[agent_name][agent_config.version] = agent_config
            
            # Register version in version manager
            await version_manager.register_version("agent", agent_name, agent_config.version)
            
            # Add to FAISS index
            await self._store(agent_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📝 Registered agent config: {agent_name}: {agent_config.version}")
            return agent_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to register agent: {e}")
            raise

    async def get(self, agent_name: str) -> Optional[Agent]:
        """【Read】获取 Agent 运行时实例，找不到返回 None"""
        agent_config = self._agent_configs.get(agent_name)
        if agent_config is None:
            return None
        return agent_config.instance if agent_config.instance is not None else None
    
    async def get_info(self, agent_name: str) -> Optional[AgentConfig]:
        """【Read】获取 Agent 完整配置（AgentConfig），找不到返回 None"""
        return self._agent_configs.get(agent_name)
    
    async def list(self) -> List[str]:
        """【Read】列出所有已注册 Agent 名称"""
        return [name for name in self._agent_configs.keys()]

    async def update(
        self,
        agent_cls: Type[Agent],
        agent_config_dict: Optional[Dict[str, Any]] = None,
        new_version: Optional[str] = None,
        description: Optional[str] = None,
        code: Optional[str] = None,
    ) -> AgentConfig:
        """【Update】用新 class/config 更新已有 Agent，自动创建新版本。
        
        流程：
        1. 实例化新 Agent 类
        2. 检查 Agent 是否已存在（不存在应使用 register）
        3. 自动生成 patch 版本号（如 1.0.0 → 1.0.1）
        4. 构建新 AgentConfig 并替换 _agent_configs 中的当前版本
        5. 写入历史 + 注册版本 + 更新 FAISS + 持久化
        """
        try:
            if agent_config_dict is None:
                # Fallback to global config by class name
                agent_config_key = inflection.underscore(agent_cls.__name__)
                agent_config_dict = getattr(config, agent_config_key, {})
            
            # Instantiate agent immediately (update is a runtime operation)
            try:
                agent_instance = agent_cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {agent_cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {agent_cls.__name__} with provided config: {e}")
            
            agent_name = agent_instance.name
            
            # Check if agent exists
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found. Use register() to register a new agent.")
            
            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict else agent_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                # Get current version from version_manager and generate next patch version
                new_version = await version_manager.generate_next_version("agent", agent_name, "patch")
            
            # Get agent code - use provided code if available (for dynamically created classes)
            if code is not None:
                agent_code = code
            else:
                agent_code = dynamic_manager.get_source_code(agent_cls)
                if not agent_code:
                    logger.warning(f"| ⚠️ Agent {agent_name} is dynamic but source code cannot be extracted")
            
            # Get agent parameters and build properties using dynamic_manager methods
            agent_parameters = dynamic_manager.get_parameters(agent_cls)
            agent_function_calling = dynamic_manager.build_function_calling(agent_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(agent_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(agent_name, agent_parameters)
            
            # --- Build AgentConfig ---
            updated_config = AgentConfig(
                name=agent_name,  # Keep same name
                description=agent_description,
                metadata=agent_metadata,
                version=new_version,
                require_grad=agent_require_grad,
                cls=agent_cls,
                config=agent_config_dict or {},
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )
            
            # Update the agent config (replaces current version)
            self._agent_configs[agent_name] = updated_config
            
            # Store in version history
            if agent_name not in self._agent_history_versions:
                self._agent_history_versions[agent_name] = {}
            self._agent_history_versions[agent_name][updated_config.version] = updated_config
            
            # Register new version record to version manager
            await version_manager.register_version(
                "agent", 
                agent_name, 
                new_version,
                description=description or f"Updated from {original_config.version}"
            )
            
            # Update embedding index
            await self._store(updated_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 🔄 Updated agent {agent_name} from v{original_config.version} to v{new_version}")
            return updated_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to update agent: {e}")
            raise

    async def copy(
        self,
        agent_name: str,
        new_name: Optional[str] = None,
        new_version: Optional[str] = None,
        new_config: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """【Create（副本）】复制已有 Agent，可改名/改配置。
        
        同名复制：自动递增版本号（patch）
        异名复制：为新名称生成初始版本号
        
        Args:
            agent_name: 源 Agent 名称
            new_name: 新名称；为 None 时复用原名
            new_version: 新版本号；为 None 时自动生成
            new_config: 合并到原配置上的新 dict
            
        Returns:
            AgentConfig: 复制后的新配置
        """
        try:
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found")
            
            if original_config.cls is None:
                raise ValueError(f"Cannot copy agent {agent_name}: no class provided")
            
            # Determine new name
            if new_name is None:
                new_name = agent_name
            
            # Prepare config dict (merge original config with new config)
            agent_config_dict = original_config.config.copy() if original_config.config else {}
            if new_config:
                # Merge new config into original config
                agent_config_dict.update(new_config)
            
            # Instantiate agent instance (copy is a runtime operation)
            try:
                agent_instance = original_config.cls(**agent_config_dict)
            except Exception as e:
                logger.error(f"| ❌ Failed to create agent instance for {original_config.cls.__name__}: {e}")
                raise ValueError(f"Failed to instantiate agent {original_config.cls.__name__} with provided config: {e}")
            
            # Apply name override if provided (after instantiation)
            if new_name != agent_name:
                agent_instance.name = new_name
            
            agent_description = agent_instance.description
            agent_metadata = agent_instance.metadata
            agent_require_grad = agent_config_dict.get("require_grad", agent_instance.require_grad) if agent_config_dict and "require_grad" in agent_config_dict else agent_instance.require_grad
            
            # Determine new version from version_manager
            if new_version is None:
                if new_name == agent_name:
                    # If copying with same name, get next version from version_manager
                    new_version = await version_manager.generate_next_version("agent", new_name, "patch")
                else:
                    # If copying with different name, get or generate version for new name
                    new_version = await version_manager.get_version("agent", new_name)
            
            # Get agent code
            agent_code = dynamic_manager.get_source_code(original_config.cls)
            if not agent_code:
                logger.warning(f"| ⚠️ Agent {new_name} is dynamic but source code cannot be extracted")
            
            # Get agent parameters and build properties using dynamic_manager methods
            agent_parameters = dynamic_manager.get_parameters(original_config.cls)
            agent_function_calling = dynamic_manager.build_function_calling(new_name, agent_description, agent_parameters)
            agent_text = dynamic_manager.build_text_representation(new_name, agent_description, agent_parameters)
            agent_args_schema = dynamic_manager.build_args_schema(new_name, agent_parameters)
            
            # --- Build AgentConfig ---
            new_agent_config = AgentConfig(
                name=new_name,
                description=agent_description,
                metadata=agent_metadata,
                version=new_version,
                require_grad=agent_require_grad,
                cls=original_config.cls,
                config=agent_config_dict,
                instance=agent_instance,
                function_calling=agent_function_calling,
                text=agent_text,
                args_schema=agent_args_schema,
                code=agent_code,
            )
            
            # Register new agent
            self._agent_configs[new_name] = new_agent_config
            
            # Store in version history
            if new_name not in self._agent_history_versions:
                self._agent_history_versions[new_name] = {}
            self._agent_history_versions[new_name][new_version] = new_agent_config
            
            # Register version record to version manager
            await version_manager.register_version(
                "agent", 
                new_name, 
                new_version,
                description=f"Copied from {agent_name}@{original_config.version}"
            )
            
            # Register to embedding index
            await self._store(new_agent_config)
            
            # Persist to JSON
            await self.save_to_json()
            # Save contract to file
            await self.save_contract()
            
            logger.info(f"| 📋 Copied agent {agent_name}@{original_config.version} to {new_name}@{new_version}")
            return new_agent_config
        
        except Exception as e:
            logger.error(f"| ❌ Failed to copy agent: {e}")
            raise

    async def unregister(self, agent_name: str) -> bool:
        """【Delete】从活跃注册表中移除 Agent（版本历史保留，可后续 restore）
        
        Args:
            agent_name: 待注销的 Agent 名称
            
        Returns:
            True 表示成功，False 表示 Agent 不存在
        """
        if agent_name not in self._agent_configs:
            logger.warning(f"| ⚠️ Agent {agent_name} not found")
            return False
        
        agent_config = self._agent_configs[agent_name]
        
        # 从活跃注册表中移除（_agent_history_versions 中的历史保留不删）
        del self._agent_configs[agent_name]

        # Persist to JSON after unregister
        await self.save_to_json()
        # Save contract to file
        await self.save_contract()
        
        logger.info(f"| 🗑️ Unregistered agent {agent_name}@{agent_config.version}")
        return True

    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """将所有 Agent 配置（含全部历史版本）持久化到 agent.json。
        
        注意：instance 不保存（运行时状态），加载时通过 build() 重新构建。
        使用 file_lock 保证并发写入安全。
        
        Args:
            file_path: 保存路径；为 None 时使用 self.save_path
            
        Returns:
            实际保存的文件路径
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            # Ensure parent directory exists
            parent_dir = os.path.dirname(file_path)
            if parent_dir:  # Only create if there's a directory component
                os.makedirs(parent_dir, exist_ok=True)
            
            # Prepare save data - save all versions for each agent
            save_data = {
                "metadata": {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_agents": len(self._agent_configs),
                    "num_versions": sum(len(versions) for versions in self._agent_history_versions.values()),
                },
                "agents": {}
            }
            
            for agent_name, version_map in self._agent_history_versions.items():
                try:
                    versions_data: Dict[str, Dict[str, Any]] = {}
                    for _, agent_config in version_map.items():
                        config_dict = agent_config.model_dump()
                        versions_data[agent_config.version] = config_dict
                    
                    # Get current_version from active config if it exists
                    # If not in active configs, use the latest version from history
                    current_version = None
                    if agent_name in self._agent_configs:
                        current_config = self._agent_configs[agent_name]
                        if current_config is not None:
                            current_version = current_config.version
                    
                    # If not found in active configs, use latest version from history
                    if current_version is None and version_map:
                        # Find latest version by comparing version strings
                        latest_version_str = None
                        for version_str in version_map.keys():
                            if latest_version_str is None:
                                latest_version_str = version_str
                            elif version_manager.compare_versions(version_str, latest_version_str) > 0:
                                latest_version_str = version_str
                        current_version = latest_version_str

                    save_data["agents"][agent_name] = {
                        "versions": versions_data,
                        "current_version": current_version,
                    }
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to serialize agent {agent_name}: {e}")
                    continue

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

            logger.info(
                f"| 💾 Saved {len(self._agent_configs)} agents with version history to {file_path}"
            )
            return str(file_path)

    async def load_from_json(
        self, file_path: Optional[str] = None, auto_initialize: bool = True
    ) -> bool:
        """从 agent.json 加载 Agent 配置（含版本历史）。
        
        instance 不在 JSON 中，需通过 build() 重建；
        仅将 current_version 对应的版本设为活跃项，其余保留在历史中。
        
        Args:
            file_path: 加载路径；为 None 时使用 self.save_path
            auto_initialize: 是否自动调用 build() 构建实例
            
        Returns:
            True 表示加载成功，False 表示文件不存在或解析失败
        """
        file_path = file_path if file_path is not None else self.save_path
        
        async with file_lock(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"| ⚠️ Agent file not found: {file_path}")
                return False
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    load_data = json.load(f)
                
                agents_data = load_data.get("agents", {})
                loaded_count = 0
                
                for agent_name, agent_data in agents_data.items():
                    try:
                        # 每个 Agent 在 JSON 中格式：{"versions": {version_str: config_dict}, "current_version": str}
                        versions_data = agent_data.get("versions")
                        if not isinstance(versions_data, dict):
                            logger.warning(f"| ⚠️ Agent {agent_name} has invalid format for 'versions' (expected dict), skipping")
                            continue
                        
                        current_version_str = agent_data.get("current_version")
                        
                        # 遍历所有版本，找出 current_version 对应的配置作为活跃项；
                        # 若无 current_version 字段，则取版本号最大者兜底
                        version_configs = []
                        latest_config = None
                        latest_version = None
                        
                        for version_str, config_dict in versions_data.items():
                            # 补齐缺失的 version 字段（以 JSON key 为准）
                            if "version" not in config_dict:
                                config_dict["version"] = version_str
                            
                            try:
                                agent_config = AgentConfig.model_validate(config_dict)
                                version_configs.append(agent_config)
                            except Exception as e:
                                logger.warning(f"| ⚠️ Failed to load agent config for {agent_name}@{version_str}: {e}")
                                continue
                            
                            # 确定活跃版本：优先匹配 current_version_str，否则取版本号最大者
                            if latest_config is None or (
                                current_version_str and agent_config.version == current_version_str
                            ) or (
                                not current_version_str and (
                                    latest_version is None or 
                                    version_manager.compare_versions(agent_config.version, latest_version) > 0
                                )
                            ):
                                latest_config = agent_config
                                latest_version = agent_config.version
                        
                        # 将所有版本写入 _agent_history_versions（供 restore 查询）
                        self._agent_history_versions[agent_name] = {
                            cfg.version: cfg for cfg in version_configs
                        }
                        
                        # 仅将活跃版本设为 _agent_configs 中的当前项
                        if latest_config:
                            self._agent_configs[agent_name] = latest_config
                            
                            # 将所有历史版本注册到 version_manager（仅记录版本号，不构建实例）
                            for agent_config in version_configs:
                                await version_manager.register_version("agent", agent_name, agent_config.version)
                            
                            # auto_initialize=True 时构建实例（instance 不保存在 JSON 中，必须重建）
                            if auto_initialize and latest_config.cls is not None:
                                await self.build(latest_config)
                            
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"| ❌ Failed to load agent {agent_name}: {e}")
                        continue
                
                logger.info(f"| 📂 Loaded {loaded_count} agents with version history from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"| ❌ Failed to load agents from {file_path}: {e}")
                return False

    async def restore(
        self, agent_name: str, version: str, auto_initialize: bool = True
    ) -> Optional[AgentConfig]:
        """将 Agent 回滚到指定历史版本，并设为当前活跃版本。
        
        从 _agent_history_versions 中 O(1) 查找目标版本，
        拷贝为新 AgentConfig 并替换 _agent_configs 中的当前项。
        
        Args:
            agent_name: Agent 名称
            version: 目标版本字符串（如 "1.0.0"）
            auto_initialize: 是否自动构建实例
            
        Returns:
            恢复后的 AgentConfig；找不到时返回 None
        """
        # Look up version from dict-based history (O(1) lookup)
        version_config = None
        if agent_name in self._agent_history_versions:
            version_config = self._agent_history_versions[agent_name].get(version)
        
        if version_config is None:
            logger.warning(f"| ⚠️ Version {version} not found for agent {agent_name}")
            return None
        
        # Create a copy to avoid modifying the history
        restored_config = AgentConfig(**version_config.model_dump())
        
        # Set as current active config
        self._agent_configs[agent_name] = restored_config
        
        # Update version manager current version
        version_history = await version_manager.get_version_history("agent", agent_name)
        if version_history:
            # Check if version exists in version history, if not register it
            if version not in version_history.versions:
                await version_manager.register_version("agent", agent_name, version)
            version_history.current_version = version
        else:
            # If version history doesn't exist, register the version first
            await version_manager.register_version("agent", agent_name, version)
        
        # Initialize if requested
        if auto_initialize and restored_config.cls is not None:
            await self.build(restored_config)
        
        # Persist to JSON (current_version changes)
        await self.save_to_json()
        
        logger.info(f"| 🔄 Restored agent {agent_name} to version {version}")
        return restored_config
    
    async def save_contract(self, agent_names: Optional[List[str]] = None):
        """将所有活跃 Agent 的文本描述汇编为 contract.md，供 Planner 读取。
        
        Planner 调度任务时依赖 contract.md 了解各 Agent 的名称、描述和参数，
        格式：每个 Agent 一段，以 "---\n" 分隔，前缀四位序号（0001/0002/...）。
        
        Args:
            agent_names: 指定写入的 Agent；为 None 时写入全部活跃 Agent
        """
        contract = []
        names = agent_names if agent_names is not None else list(self._agent_configs.keys())
        for index, agent_name in enumerate(names):
            agent_info = await self.get_info(agent_name)
            if agent_info is None:
                logger.warning(f"| ⚠️  Skipping agent '{agent_name}' in contract (not found or failed to create)")
                continue
            # agent_info.text: 由 dynamic_manager.build_text_representation 生成
            # 包含 name / description / parameters 等 Planner 需要的可读信息
            text = agent_info.text
            contract.append(f"{index + 1:04d}\n{text}\n")
        # 用 "---\n" 拼接各 Agent 段落，写入 contract.md
        contract_text = "---\n".join(contract)
        with open(self.contract_path, "w", encoding="utf-8") as f:
            f.write(contract_text)
        logger.info(f"| 📝 Saved {len(contract)} agents contract to {self.contract_path}")
        
    async def load_contract(self) -> str:
        """Load the contract for an agent"""
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()
        return contract_text
    
    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve similar agents using FAISS similarity search.
        
        Args:
            query: Query string to search for
            k: Number of results to return (default: 4)
            
        Returns:
            List of dictionaries containing agent information with similarity scores
        """
        if self._faiss_service is None:
            logger.warning("| ⚠️ FAISS service not initialized, cannot retrieve agents")
            return []
        
        try:
            from src.environment.faiss.types import FaissSearchRequest
            
            request = FaissSearchRequest(
                query=query,
                k=k,
                fetch_k=k * 5  # Fetch more candidates before filtering
            )
            
            result = await self._faiss_service.search_similar(request)
            
            if not result.success:
                logger.warning(f"| ⚠️ FAISS search failed: {result.message}")
                return []
            
            # Extract documents and scores from result
            documents = []
            if result.extra and "documents" in result.extra:
                docs = result.extra["documents"]
                scores = result.extra.get("scores", [])
                
                for doc, score in zip(docs, scores):
                    # Extract agent name from metadata
                    metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                    agent_name = metadata.get("name", "")
                    
                    # Get agent config if available
                    agent_config = None
                    if agent_name and agent_name in self._agent_configs:
                        agent_config = self._agent_configs[agent_name]
                    
                    documents.append({
                        "name": agent_name,
                        "description": metadata.get("description", ""),
                        "score": float(score),
                        "content": doc.get("page_content", "") if isinstance(doc, dict) else str(doc),
                        "config": agent_config.model_dump() if agent_config else None
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"| ❌ Error retrieving agents: {e}")
            return []
    
    async def get_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get variables from agents, where each agent's class source code is used as the variable value.
        
        Args:
            agent_name (Optional[str]): Name of a specific agent. If None, returns variables for all agents.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping agent names to Variable objects. Each Variable has:
                - name: agent name
                - type: "agent_code"
                - description: agent description
                - require_grad: agent's require_grad value
                - variables: agent's class source code (as string value)
        """
        # Lazy import to avoid circular dependency
        from src.optimizer.types import Variable
        
        variables: Dict[str, Variable] = {}
        
        if agent_name is not None:
            # Get specific agent
            agent_config = await self.get_info(agent_name)
            if agent_config is None:
                logger.warning(f"| ⚠️ Agent {agent_name} not found")
                return variables
            
            agent_configs = {agent_name: agent_config}
        else:
            # Get all agents
            agent_configs = self._agent_configs
        
        for name, agent_config in agent_configs.items():
            # Get agent code
            agent_code = ""
            if agent_config.cls is not None:
                agent_code = dynamic_manager.get_full_module_source(agent_config.cls) or ""
            elif agent_config.code:
                agent_code = agent_config.code
            
            # Create Variable for this agent
            variable = Variable(
                name=name,
                type="agent_code",
                description=agent_config.description or f"Code for agent {name}",
                require_grad=agent_config.require_grad,
                template=None,
                variables=agent_code  # Store code as the variable value
            )
            variables[name] = variable
        
        return variables
    
    async def get_trainable_variables(self, agent_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get trainable variables from agents, filtering out agents with require_grad=False.
        
        Only returns variables for agents where require_grad=True.
        
        Args:
            agent_name (Optional[str]): Name of a specific agent. If None, returns variables for all trainable agents.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping agent names to Variable objects for trainable agents.
        """
        async with self._variables_lock:
            all_variables = await self.get_variables(agent_name=agent_name)
            trainable_variables = {name: var for name, var in all_variables.items() if var.require_grad}
            return trainable_variables
    
    async def set_variables(self, agent_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> AgentConfig:
        """Set variable values in an agent and create a new version.
        
        Args:
            agent_name: Name of the agent to update
            variable_updates: Dictionary mapping variable names to new values.
                For agents, this is typically {"code": new_code_string}
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            AgentConfig: Updated agent configuration
        """
        async with self._variables_lock:
            original_config = self._agent_configs.get(agent_name)
            if original_config is None:
                raise ValueError(f"Agent {agent_name} not found. Use register() to register a new agent.")
            
            # For agents, variable_updates format is {"name": "agent_name", "variables": "agent code"}
            # Extract the new code from "variables" field
            if "variables" not in variable_updates:
                raise ValueError(f"variable_updates must contain 'variables' field with agent code, got: {list(variable_updates.keys())}")
            
            new_code = variable_updates["variables"]
            if not isinstance(new_code, str):
                raise ValueError(f"Agent code must be a string, got {type(new_code)}")
            
            # Load agent class from code
            class_name = dynamic_manager.extract_class_name_from_code(new_code)
            if not class_name:
                raise ValueError(f"Cannot extract class name from code")
            
            try:
                agent_cls = dynamic_manager.load_class(
                    new_code,
                    class_name=class_name,
                    base_class=Agent,
                    context="agent"
                )
            except Exception as e:
                logger.error(f"| ❌ Failed to load agent class from code: {e}")
                raise ValueError(f"Failed to load agent class from code: {e}")
            
            # Use update() function to handle version management and persistence
            # Pass the code directly to avoid re-extracting from dynamically created class
            update_description = description or f"Updated code for {agent_name}"
            return await self.update(
                agent_cls=agent_cls,
                agent_config_dict=original_config.config,
                new_version=new_version,
                description=update_description,
                code=new_code  # Pass code directly since agent_cls is dynamically created
            )

    async def cleanup(self):
        """清理所有 Agent 状态：清空活跃注册表 + 版本历史 + FAISS 服务"""
        try:
            # 清空内存状态
            self._agent_configs.clear()
            self._agent_history_versions.clear()
                
            # Clean up Faiss service (async)
            if self._faiss_service is not None:
                await self._faiss_service.cleanup()
            logger.info("| 🧹 Agent context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during agent context manager cleanup: {e}")
            
    async def __call__(self, name: str, input: Dict[str, Any], ctx: SessionContext = None, **kwargs) -> Any:
        """按名称调用 Agent：查找实例并执行，透传 ctx 和额外参数
        
        Args:
            name: Agent 名称
            input: 传给 Agent 的输入
            ctx: 会话上下文；为 None 时自动创建
            **kwargs: 透传给 Agent 的额外参数
        Returns:
            Agent 执行结果
        """
        if ctx is None:
            ctx = SessionContext()
        
        agent_info = await self.get_info(name)
        
        # Agent args: ctx + any extra kwargs from the caller
        agent_args = {
            "ctx": ctx,
            **kwargs,
        }
        
        version = agent_info.version
        agent_instance = agent_info.instance
        logger.info(f"| ✅ Using agent {name}@{version}")
        
        return await agent_instance(**input, **agent_args)

