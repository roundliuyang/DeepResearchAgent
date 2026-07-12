"""TCP Server

Server implementation for the Tool Context Protocol with lazy loading support.
"""
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
import asyncio
import os
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.optimizer.types import Variable

from src.logger import logger
from src.config import config
from src.tool.context import ToolContextManager
from src.tool.types import Tool, ToolConfig, ToolResponse
from src.session import SessionContext
from src.utils import assemble_project_path

class TCPServer(BaseModel):
    """TCP Server for managing tool registration and execution with lazy loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    base_dir: str = Field(default=None, description="The base directory to use for the tools")
    save_path: str = Field(default=None, description="The path to save the tools")
    contract_path: str = Field(default=None, description="The path to save the tool contract")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the TCP Server."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, ToolConfig] = {}  # tool_name -> ToolConfig

        
    async def initialize(self, tool_names: Optional[List[str]] = None):
        """
        初始化工具服务器，设置工作目录并委托给 ToolContextManager 完成具体初始化。
        
        Args:
            tool_names: 可选的工具名称列表，如果提供则只初始化指定的工具；
                       如果为 None 则初始化所有已注册的工具
        """
        
        # ===== 阶段1: 设置工作目录和文件路径 =====
        # 构建工具存储的基础目录：workdir/{tag}/tool/
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "tool"))
        os.makedirs(self.base_dir, exist_ok=True)
        # 工具配置保存路径：workdir/{tag}/tool/tool.json
        self.save_path = os.path.join(self.base_dir, "tool.json")
        # 工具契约文件路径：workdir/{tag}/tool/contract.md（包含所有工具的文本描述，用于发送给大模型）
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 TCP Server base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")
        
        # ===== 阶段2: 创建并初始化工具上下文管理器 =====
        # ToolContextManager 负责工具的加载、注册、版本管理、持久化等核心功能
        self.tool_context_manager = ToolContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
            model_name="openrouter/gemini-3-flash-preview",      # 用于生成工具描述的模型
            embedding_model_name="openrouter/text-embedding-3-large",  # 用于工具向量嵌入的模型
        )
        # 委托给 ToolContextManager 执行完整的初始化流程（加载工具、构建实例、保存配置等）
        await self.tool_context_manager.initialize(tool_names=tool_names)
        
        logger.info("| ✅ Tools initialization completed")
        
    async def get_contract(self) -> str:
        """Get the contract for all tools"""
        return await self.tool_context_manager.load_contract()
    
    async def register(self, 
                       tool: Union[Tool, Type[Tool]],
                       config: Optional[Dict[str, Any]] = None,
                       override: bool = False,
                       version: Optional[str] = None,
                       code: Optional[str] = None) -> ToolConfig:
        """Register a tool class or instance asynchronously.
        
        Args:
            tool: Tool class or instance to register
            config: Configuration dict for tool initialization (required when tool is a class)
            override: Whether to override existing registration
            version: Optional version string
            code: Optional explicit source code for the tool class
            
        Returns:
            ToolConfig: Tool configuration
        """
        tool_config = await self.tool_context_manager.register(
            tool, 
            tool_config_dict=config, 
            override=override,
            version=version,
            code=code
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def list(self) -> List[str]:
        """List all registered tools
        
        Args:
            include_disabled: Whether to include disabled tools
            
        Returns:
            List[str]: List of tool names
        """
        return await self.tool_context_manager.list()
    
    
    async def get(self, tool_name: str) -> Tool:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool: Tool instance or None if not found
        """
        tool = await self.tool_context_manager.get(tool_name)
        return tool
    
    async def get_info(self, tool_name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            ToolConfig: Tool configuration or None if not found
        """
        return await self.tool_context_manager.get_info(tool_name)
    
    async def cleanup(self):
        """Cleanup all tools"""
        await self.tool_context_manager.cleanup()
        self._registered_configs.clear()
    
    async def update(self, 
                     tool_name: str, tool: Union[Tool, Type[Tool]], 
                     config: Optional[Dict[str, Any]] = None,
                     new_version: Optional[str] = None, 
                     description: Optional[str] = None) -> ToolConfig:
        """Update an existing tool with new configuration and create a new version
        
        Args:
            tool_name: Name of the tool to update
            tool: New tool class or instance with updated implementation
            config: Configuration dict for tool initialization (required when tool is a class)
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        tool_config = await self.tool_context_manager.update(
            tool_name, tool, tool_config_dict=config, new_version=new_version, description=description
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def copy(self, tool_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> ToolConfig:
        """Copy an existing tool
        
        Args:
            tool_name: Name of the tool to copy
            new_name: New name for the copied tool. If None, uses original name.
            new_version: New version for the copied tool. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            ToolConfig: New tool configuration
        """
        tool_config = await self.tool_context_manager.copy(
            tool_name, new_name, new_version, **override_config
        )
        self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def unregister(self, tool_name: str) -> bool:
        """Unregister a tool
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.tool_context_manager.unregister(tool_name)
        if success and tool_name in self._registered_configs:
            del self._registered_configs[tool_name]
        return success
    
    async def restore(self, tool_name: str, version: str, auto_initialize: bool = True) -> Optional[ToolConfig]:
        """Restore a specific version of a tool from history
        
        Args:
            tool_name: Name of the tool
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored tool
            
        Returns:
            ToolConfig of the restored version, or None if not found
        """
        tool_config = await self.tool_context_manager.restore(tool_name, version, auto_initialize)
        if tool_config:
            self._registered_configs[tool_config.name] = tool_config
        return tool_config
    
    async def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve similar tools using FAISS similarity search.
        
        Args:
            query: Query string to search for
            k: Number of results to return (default: 4)
            
        Returns:
            List of dictionaries containing tool information with similarity scores
        """
        return await self.tool_context_manager.retrieve(query=query, k=k)
    
    async def get_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get variables from tools, where each tool's code is used as the variable value.
        
        Args:
            tool_name (Optional[str]): Name of a specific tool. If None, returns variables for all tools.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping tool names to Variable objects.
        """
        return await self.tool_context_manager.get_variables(tool_name=tool_name)
    
    async def get_trainable_variables(self, tool_name: Optional[str] = None) -> Dict[str, 'Variable']:
        """Get trainable variables from tools, filtering out tools with require_grad=False.
        
        Args:
            tool_name (Optional[str]): Name of a specific tool. If None, returns trainable variables for all tools.
            
        Returns:
            Dict[str, Variable]: Dictionary mapping tool names to Variable objects for tools with require_grad=True.
        """
        return await self.tool_context_manager.get_trainable_variables(tool_name=tool_name)
    
    async def set_variables(self, tool_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> ToolConfig:
        """Set variable values in a tool and create a new version.
        
        Args:
            tool_name: Name of the tool to update
            variable_updates: Dictionary mapping variable names to new values.
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            
        Returns:
            ToolConfig: Updated tool configuration
        """
        updated_config = await self.tool_context_manager.set_variables(
            tool_name=tool_name, 
            variable_updates=variable_updates,
            new_version=new_version, 
            description=description
        )
        self._registered_configs[updated_config.name] = updated_config
        return updated_config
    
    async def __call__(self, 
                       name: str, 
                       input: Dict[str, Any], 
                       timeout: Optional[float] = None,
                       ctx: SessionContext = None,
                       **kwargs
                       ) -> ToolResponse:
        """Call a tool by name with optional timeout and context
        
        Args:
            name: Tool name
            input: Input for the tool
            timeout: Timeout in seconds for this specific call (overrides default_timeout if provided)
            ctx: Tool context
            
        Returns:
            ToolResponse: Tool result
        """
        return await self.tool_context_manager(name, 
                                               input, 
                                               timeout=timeout, 
                                               ctx=ctx, 
                                               **kwargs)


# Global TCP server instance
tcp = TCPServer()