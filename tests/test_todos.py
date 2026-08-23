"""Persistent execution-checklist tests."""
import copy
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-todos-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import session
from commands import format_todos
from todos import MAX_TODO_ITEMS, normalize_persisted_plan
from tools import execute_tool, update_plan_tool


class UpdatePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))

    def test_replaces_entire_ordered_plan_without_ids(self) -> None:
        plan = [
            {"step": "Inspect", "status": "completed"},
            {"step": "Fix", "status": "in_progress"},
            {"step": "Test", "status": "pending"},
        ]
        with patch("tools.save_state") as save:
            result = update_plan_tool({"plan": plan}, session.state)

        self.assertTrue(result["ok"])
        self.assertEqual(session.state["todos"], plan)
        self.assertNotIn("id", session.state["todos"][0])
        save.assert_called_once_with()

        replacement = [
            {"step": "Test", "status": "completed"},
        ]
        with patch("tools.save_state"):
            update_plan_tool({"plan": replacement}, session.state)
        self.assertEqual(session.state["todos"], replacement)

    def test_empty_plan_clears_todos(self) -> None:
        session.state["todos"] = [
            {"step": "Old", "status": "pending"},
        ]
        with patch("tools.save_state"):
            result = update_plan_tool({"plan": []}, session.state)
        self.assertTrue(result["ok"])
        self.assertEqual(session.state["todos"], [])

    def test_rejects_multiple_in_progress_atomically(self) -> None:
        original = [{"step": "Old", "status": "pending"}]
        session.state["todos"] = copy.deepcopy(original)
        with patch("tools.save_state") as save:
            result = update_plan_tool(
                {
                    "plan": [
                        {"step": "One", "status": "in_progress"},
                        {"step": "Two", "status": "in_progress"},
                    ]
                },
                session.state,
            )
        self.assertFalse(result["ok"])
        self.assertEqual(session.state["todos"], original)
        save.assert_not_called()

    def test_root_and_subagent_plans_are_isolated(self) -> None:
        child = {"todos": []}
        with patch("tools.save_state"):
            update_plan_tool(
                {"plan": [{"step": "Root", "status": "pending"}]},
                session.state,
            )
            update_plan_tool(
                {"plan": [{"step": "Child", "status": "pending"}]},
                child,
            )
        self.assertEqual(session.state["todos"][0]["step"], "Root")
        self.assertEqual(child["todos"][0]["step"], "Child")

    def test_dispatch_updates_the_explicit_owner_not_its_copy(self) -> None:
        owner = {"todos": []}
        context = dict(owner)
        context["todo_state"] = owner
        call = {
            "name": "update_plan",
            "arguments": json.dumps(
                {"plan": [{"step": "Owned", "status": "pending"}]}
            ),
        }
        with patch("tools.save_state"):
            result = asyncio.run(
                execute_tool(call, execution_context=context)
            )
        self.assertTrue(result["ok"])
        self.assertEqual(
            owner["todos"],
            [{"step": "Owned", "status": "pending"}],
        )
        self.assertEqual(context["todos"], [])

    def test_load_normalization_rejects_noncanonical_items(self) -> None:
        normalized = normalize_persisted_plan(
            [
                {"id": 1, "text": "One", "status": "pending"},
            ]
        )
        self.assertEqual(normalized, [])

    def test_rejects_unknown_fields_and_oversized_plans(self) -> None:
        with patch("tools.save_state") as save:
            unknown = update_plan_tool(
                {
                    "plan": [
                        {
                            "step": "One",
                            "status": "pending",
                            "id": 1,
                        }
                    ]
                },
                session.state,
            )
            oversized = update_plan_tool(
                {
                    "plan": [
                        {"step": str(index), "status": "pending"}
                        for index in range(MAX_TODO_ITEMS + 1)
                    ]
                },
                session.state,
            )
        self.assertFalse(unknown["ok"])
        self.assertFalse(oversized["ok"])
        save.assert_not_called()

    def test_persistence_failure_rolls_back_memory(self) -> None:
        original = [{"step": "Old", "status": "pending"}]
        session.state["todos"] = copy.deepcopy(original)
        with (
            patch("tools.save_state", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            update_plan_tool(
                {"plan": [{"step": "New", "status": "pending"}]},
                session.state,
            )
        self.assertEqual(session.state["todos"], original)

    def test_todos_command_format_has_no_ids(self) -> None:
        session.state["todos"] = [
            {"step": "Inspect", "status": "completed"},
            {"step": "Test", "status": "in_progress"},
        ]
        self.assertEqual(format_todos(), "[x] Inspect\n[>] Test")


if __name__ == "__main__":
    unittest.main()
