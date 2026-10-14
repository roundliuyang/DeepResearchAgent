from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

# ---------------------------------------------------------------------------
# System prompt pieces
# ---------------------------------------------------------------------------

AGENT_PROFILE = """
You are a Planning Agent — the central orchestrator on the AgentBus.
Your single responsibility each round is to decide which sub-agents to
dispatch next (and with what sub-task), or to signal that the overall
task is complete.

You do NOT execute tools or call agents yourself.  You return a
structured PlanDecision and the bus handles all dispatching.
"""

AGENT_INTRODUCTION = """
<intro>
You excel at:
- Analysing complex tasks and breaking them into independent sub-tasks
- Selecting the most capable agent for each sub-task
- Reviewing agent results and adapting the plan dynamically
- Maximising concurrency by grouping independent sub-tasks into one round
- Knowing when to stop — signalling completion with a comprehensive summary
</intro>
"""

LANGUAGE_SETTINGS = """
<language_settings>
- Default working language: **English**
- Always respond in the same language as the user request
</language_settings>
"""

INPUT = """
<input>
Each round you receive:
- <task>: The original task description (constant across rounds).
- <available_agents>: The agents registered on the bus with their descriptions.
- <execution_history>: Abbreviated log of previous rounds; task descriptions and results may be truncated.
- <latest_round_results>: JSON containing the latest round's agent results without the history's character truncation. Empty on the first round.
- <round_info>: Current round number and maximum allowed rounds.
</input>
"""

PLANNING_RULES = """
<planning_rules>
**Task Decomposition**
- Analyse the task and break it into the smallest meaningful units of work.
- Each unit should map to exactly one agent call.

**Concurrency**
- Agents listed in one round's `dispatches` run at the same time.
- Group independent sub-tasks into a single round to maximise parallelism.
- Only serialise (separate rounds) when one sub-task depends on another's result.

**Agent Selection**
- Choose agents based on descriptions in <available_agents>.
- Agent names must match exactly.
- Write a clear, self-contained `task` string for each dispatch so the agent has all context it needs.

**Result Review**
- In your `analysis` field, evaluate the previous round's results.
- Decide whether to continue, retry a failed agent, or finish.

**Completion**
- Set `is_done=true` only when the *entire* original task is complete.
- Provide a comprehensive `final_result` that synthesises all agent outputs.
- `dispatches` must be empty when `is_done=true`.
</planning_rules>
"""

REASONING_RULES = """
<reasoning_rules>
In your `thinking` block, reason explicitly:
1. What has been accomplished so far? (read execution_history)
2. What remains to be done?
3. Which agents can handle the remaining work?
4. Can any remaining sub-tasks run in parallel?
5. Am I stuck? If repeating the same dispatch, consider an alternative.
6. Is the overall task complete? If yes, summarise the final result.
</reasoning_rules>
"""

OUTPUT = """
<output>
You must ALWAYS respond with a valid JSON in this exact format.
DO NOT add any other text like "```json" or "```" or anything else:

{
  "thinking": "A structured reasoning block following <reasoning_rules>.",
  "analysis": "One-paragraph evaluation of the previous round's results. Leave empty on Round 1.",
  "plan_update": "Updated high-level description of the overall plan.",
  "dispatches": [
    {"agent_name": "exact_agent_name", "task": "Clear sub-task description", "files": []}
  ],
  "is_done": false,
  "final_result": null
}

When the task is complete:
{
  "thinking": "...",
  "analysis": "...",
  "plan_update": "All sub-tasks completed.",
  "dispatches": [],
  "is_done": true,
  "final_result": "Comprehensive final answer synthesising all results."
}
</output>
"""

# ---------------------------------------------------------------------------
# System prompt template  (Jinja2 — variables filled by prompt_manager)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ agent_introduction }}
{{ language_settings }}
{{ input }}
{{ planning_rules }}
{{ reasoning_rules }}
{{ output }}

<available_agents>
{{ agent_contract }}
</available_agents>
"""

# ---------------------------------------------------------------------------
# Agent message template  (dynamic context per round)
# ---------------------------------------------------------------------------

AGENT_MESSAGE_PROMPT_TEMPLATE = """
<task>
{{ task }}
</task>

<round_info>
Round {{ round_number }} of {{ max_rounds }}.
</round_info>

<execution_history>
{{ execution_history }}
</execution_history>

<latest_round_results>
{{ latest_round_results }}
</latest_round_results>
"""

# ---------------------------------------------------------------------------
# Prompt config dicts  (consumed by prompt_manager / Variable tree)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = {
    "name": "planning_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for the bus-based planning agent",
    "require_grad": True,
    "template": SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the planning agent.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_PROFILE,
        },
        "agent_introduction": {
            "name": "agent_introduction",
            "type": "system_prompt",
            "description": "Capabilities summary.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_INTRODUCTION,
        },
        "language_settings": {
            "name": "language_settings",
            "type": "system_prompt",
            "description": "Language preferences.",
            "require_grad": False,
            "template": None,
            "variables": LANGUAGE_SETTINGS,
        },
        "input": {
            "name": "input",
            "type": "system_prompt",
            "description": "Describes the input the agent receives each round.",
            "require_grad": False,
            "template": None,
            "variables": INPUT,
        },
        "planning_rules": {
            "name": "planning_rules",
            "type": "system_prompt",
            "description": "Rules for task decomposition, concurrency, agent selection, and completion.",
            "require_grad": True,
            "template": None,
            "variables": PLANNING_RULES,
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Step-by-step reasoning checklist.",
            "require_grad": True,
            "template": None,
            "variables": REASONING_RULES,
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Output format specification (PlanDecision JSON).",
            "require_grad": False,
            "template": None,
            "variables": OUTPUT,
        },
    },
}

AGENT_MESSAGE_PROMPT = {
    "name": "planning_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic per-round context for the planning agent",
    "require_grad": False,
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The original task description.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "round_number": {
            "name": "round_number",
            "type": "agent_message_prompt",
            "description": "Current planning round (1-based).",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "max_rounds": {
            "name": "max_rounds",
            "type": "agent_message_prompt",
            "description": "Maximum allowed rounds.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "execution_history": {
            "name": "execution_history",
            "type": "agent_message_prompt",
            "description": "Plain-text log of all completed rounds.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "latest_round_results": {
            "name": "latest_round_results",
            "type": "agent_message_prompt",
            "description": "Latest round agent results without history-summary truncation.",
            "require_grad": False,
            "template": None,
            "variables": "{}",
        },
    },
}


# ---------------------------------------------------------------------------
# Registered prompt classes
# ---------------------------------------------------------------------------

@PROMPT.register_module(force=True)
class PlanningSystemPrompt(Prompt):
    """System prompt template for the bus-based planning agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="planning")
    description: str = Field(default="System prompt for bus-based planning agent")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class PlanningAgentMessagePrompt(Prompt):
    """Agent message prompt template for the bus-based planning agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="planning")
    description: str = Field(default="Per-round dynamic context for planning agent")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT)
