"""Canonical execution-checklist model and validation."""
from typing import Any

MAX_TODO_ITEMS = 64
MAX_TODO_STEP_CHARS = 500
MAX_TODO_EXPLANATION_CHARS = 2000
MAX_TODO_PLAN_CHARS = 12000
TODO_STATUSES = {"pending", "in_progress", "completed"}


def validate_plan(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("plan must be a list")
    if len(value) > MAX_TODO_ITEMS:
        raise ValueError(f"plan exceeds {MAX_TODO_ITEMS} items")

    plan: list[dict[str, str]] = []
    in_progress = 0
    total_chars = 0
    for raw_item in value:
        if not isinstance(raw_item, dict):
            raise ValueError("each plan item must be an object")
        if set(raw_item) != {"step", "status"}:
            raise ValueError("each plan item accepts only step and status")
        step_value = raw_item.get("step")
        status = raw_item.get("status")
        if not isinstance(step_value, str):
            raise ValueError("each plan item requires a string step")
        step = step_value.strip()
        if not step:
            raise ValueError("each plan item requires a step")
        if len(step) > MAX_TODO_STEP_CHARS:
            raise ValueError(
                f"plan step exceeds {MAX_TODO_STEP_CHARS} characters"
            )
        total_chars += len(step)
        if total_chars > MAX_TODO_PLAN_CHARS:
            raise ValueError(
                f"plan text exceeds {MAX_TODO_PLAN_CHARS} characters"
            )
        if not isinstance(status, str) or status not in TODO_STATUSES:
            raise ValueError(f"invalid plan status: {status}")
        if status == "in_progress":
            in_progress += 1
        plan.append({"step": step, "status": status})

    if in_progress > 1:
        raise ValueError("at most one step can be in_progress")
    return plan
