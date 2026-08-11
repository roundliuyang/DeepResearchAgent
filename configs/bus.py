from mmengine.config import read_base
with read_base():
    from .base import memory_config, window_size, max_tokens
    from .agents.planning import planning_agent
    from .agents.tool_calling import tool_calling_agent
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.deep_analyzer import deep_analyzer_tool
    from .tools.bash import bash_tool
    from .tools.todo import todo_tool
    from .tools.skill_generator import skill_generator_tool
    from .environments.file_system import environment as file_system_environment
    from .memory.general_memory_system import memory_system as general_memory_system
    from .memory.optimizer_memory_system import memory_system as optimizer_memory_system

tag = "bus"
workdir = f"workdir/{tag}"
log_path = "bus.log"

use_local_proxy = True
version = "0.1.0"
model_name = "openrouter/deepseek-v4-flash"

env_names = [
    "file_system",
]
memory_names = [
    "general_memory_system",
    "optimizer_memory_system",
]
# Agents on the bus: planner + sub-agents
agent_names = [
    "planning",
    "tool_calling",
]
# Tools available to sub-agents (not the planner — planner dispatches agents only)
tool_names = [
    "bash",
    "python_interpreter",
    "done",
    "todo",
    "skill_generator",
]
skill_names = [
    "hello-world",
]

# -----------------TOOL CONFIG-----------------
bash_tool.update(require_grad=False)
todo_tool.update(base_dir="tool/todo", require_grad=False)
deep_researcher_tool.update(model_name="openrouter/o3", base_dir="tool/deep_researcher")
deep_analyzer_tool.update(model_name="openrouter/o3", base_dir="tool/deep_analyzer", require_grad=False)
skill_generator_tool.update(model_name="openrouter/gemini-3-flash-preview", base_dir="skill")

# -----------------MEMORY CONFIG-----------------
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

# -----------------ENVIRONMENT CONFIG-----------------
file_system_environment.update(base_dir="environment/file_system", require_grad=False)

# -----------------AGENT CONFIG-----------------
planning_agent.update(
    workdir=f"{workdir}/agent/planning_agent",
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
)
tool_calling_agent.update(
    workdir=f"{workdir}/agent/tool_calling_agent",
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
)
