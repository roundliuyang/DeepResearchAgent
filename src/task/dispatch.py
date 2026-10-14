"""Stable identity for a planner dispatch, independent of the selected agent."""


def make_dispatch_id(task_id: str, round_number: int, index: int) -> str:
    """Use the original one-based decision index, before invalid agents are filtered."""
    return f"{task_id}-r{round_number}-d{index}"
