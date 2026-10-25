from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.tool_calling import tool_calling_agent
    # from .tools.browser import browser_tool
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.deep_analyzer import deep_analyzer_tool
    from .tools.mdify import mdify_tool
    from .tools.plotter import plotter_tool
    from .tools.bash import bash_tool
    from .tools.todo import todo_tool
    from .tools.skill_generator import skill_generator_tool
    from .tools.ai_capability_debate import ai_capability_debate_tool
    from .environments.file_system import environment as file_system_environment
    from .memory.general_memory_system import memory_system as general_memory_system
    from .memory.optimizer_memory_system import memory_system as optimizer_memory_system

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
# model_name = "openrouter/gemini-3-flash-preview"
# model_name = "openrouter/claude-sonnet-4.5"
# model_name = "openrouter/deepseek-v4-flash"
model_name = "openrouter/qwen3.7-max"

env_names = [
    "file_system"
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system"
]
agent_names = [
    "tool_calling"
]
tool_names = [
    'bash',                    # 执行 Shell 命令，调用本机已安装的 CLI 工具或脚本
    # 'python_interpreter',    # 执行 Python 代码并返回输出，用于计算和数据处理
    'done',                    # 向 Agent 表明任务已完成
    'todo',                    # 拆解任务、维护 todo.md，并跟踪各步骤的状态和结果
    # 'mdify',                # 将多种格式的文件转换为 Markdown 文本并保存
    # "deep_analyzer",        # 对附带文件的复杂任务进行多步骤分析与信息综合
    # "deep_researcher",      # 执行多轮网页搜索和内容分析，汇总研究结果
    # "browser",              # 操作浏览器，与网页交互以完成任务
    # "plotter",              # 将文本、Markdown 表格或 CSV 数据绘制为 PNG 图表
    "skill_generator",         # 根据需求生成技能包，校验并注册到技能管理器 SCP
    "ai_capability_debate",    # 组织多个大模型讨论，评估 AI 能力边界并汇总共识
    "mcp_importer",            # 连接 MCP 服务并发现工具，将服务注册为可调用的代理工具
]
skill_names = [
    "hello-world",
]

#-----------------MCP CONNECTIONS CONFIG-----------------
# MCP 连接配置示例
mcp_connections = {
    "math_server": {
        "command": "python",
        "args": ["/Users/xx/DeepResearchAgent/servers/math_server.py"],
        "transport": "stdio"
    },
    "docx_server": {
        "command": "python",
        "args": ["/Users/xx/DeepResearchAgent/servers/docx_server.py"],
        "transport": "stdio"
    },
}

#-----------------BASH TOOL CONFIG-----------------
bash_tool.update(
    require_grad=False,
)
#-----------------MDIFY TOOL CONFIG-----------------
mdify_tool.update(
    base_dir="tool/mdify",
)
todo_tool.update(
    base_dir="tool/todo",
    require_grad=False,
)
#-----------------BROWSER TOOL CONFIG-----------------
# browser_tool.update(
#     model_name="openrouter/gpt-4.1",
#     base_dir="tool/browser",
# )
#-----------------DEEP RESEARCHER TOOL CONFIG-----------------
deep_researcher_tool.update(
    model_name="openrouter/o3",
    base_dir="tool/deep_researcher",
)

#-----------------DEEP ANALYZER TOOL CONFIG-----------------
deep_analyzer_tool.update(
    model_name="openrouter/o3",
    base_dir="tool/deep_analyzer",
    require_grad=False,
)

#-----------------PLOTTER TOOL CONFIG-----------------
plotter_tool.update(
    model_name="openrouter/o3",
    base_dir="tool/plotter",
)
#-----------------SKILL GENERATOR TOOL CONFIG-----------------
skill_generator_tool.update(
    model_name=model_name,
    base_dir="skill",
)
#-----------------AI CAPABILITY DEBATE TOOL CONFIG-----------------
ai_capability_debate_tool.update(
    model_name=model_name,
    agent_models=[
        "openrouter/gemini-3-flash-preview",
        "openrouter/gpt-5.2",
        "openrouter/claude-sonnet-4.5",
        "openrouter/grok-4.1-fast",
    ],
    base_dir="tool/ai_capability_debate",
)
#-----------------MEMORY SYSTEM CONFIG-----------------
general_memory_system.update(
    base_dir="memory/general_memory_system",
    model_name=model_name,
    max_summaries=10,
    max_insights=10,
    require_grad=False,
)
optimizer_memory_system.update(
    base_dir="memory/optimizer_memory_system",
    model_name=model_name,
    max_records_per_session=10,
    require_grad=False,
)

#-----------------FILE SYSTEM ENVIRONMENT CONFIG-----------------
file_system_environment.update(
    base_dir="environment/file_system",
    require_grad=False,
)

#-----------------TOOL CALLING AGENT CONFIG-----------------
tool_calling_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
)
