import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-history-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ["MAX_TOOL_OUTPUT"] = "400"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chat_store
import session
from tools import chat_history_tool


class HistoryToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._chat_dir = Path(tempfile.mkdtemp(prefix="chats-"))
        self._chat_patch = patch.object(
            chat_store,
            "CHAT_DIR",
            self._chat_dir,
        )
        self._chat_patch.start()
        self.addCleanup(self._chat_patch.stop)
        chat_store._SEQ_CACHE.clear()
        self.main = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
        )
        self.old = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
        )
        session.state["current_session_id"] = self.main
        chat_store.append_item(
            self.main,
            {"role": "user", "content": "alpha secret-one"},
            source="user",
        )
        chat_store.append_item(
            self.old,
            {"role": "user", "content": "beta secret-two"},
            source="user",
        )

    def test_list_returns_sessions(self) -> None:
        result = chat_history_tool({"action": "list"})
        self.assertTrue(result["ok"])
        ids = {row["session_id"] for row in result["sessions"]}
        self.assertIn(self.main, ids)
        self.assertIn(self.old, ids)

    def test_default_search_scopes_current_session(self) -> None:
        result = chat_history_tool(
            {"action": "search", "query": "secret"},
            execution_context={"session_id": self.main},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["hits"][0]["session_id"], self.main)
        self.assertIn("seq", result["hits"][0])

    def test_all_sessions_searches_old_sessions(self) -> None:
        result = chat_history_tool(
            {
                "action": "search",
                "query": "secret",
                "all_sessions": True,
            },
            execution_context={"session_id": self.main},
        )
        ids = {hit["session_id"] for hit in result["hits"]}
        self.assertEqual(ids, {self.main, self.old})

    def test_read_supports_seq_range(self) -> None:
        e2 = chat_store.append_item(
            self.main,
            {"role": "user", "content": "second"},
            source="user",
        )
        result = chat_history_tool(
            {
                "action": "read",
                "session_id": self.main,
                "start_seq": e2["seq"],
                "end_seq": e2["seq"],
            }
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["records"]), 1)
        self.assertIn("second", result["records"][0]["text"])

    def test_tool_output_respects_max_tool_output(self) -> None:
        import tools as tools_mod
        for i in range(30):
            chat_store.append_item(
                self.main,
                {
                    "role": "user",
                    "content": f"pad-{i}-" + ("z" * 80),
                },
                source="user",
            )
        with patch.object(tools_mod, "MAX_TOOL_OUTPUT", 400):
            result = chat_history_tool(
                {
                    "action": "read",
                    "session_id": self.main,
                    "limit": 100,
                }
            )
        self.assertTrue(result["ok"])
        raw = json.dumps(result, ensure_ascii=False)
        self.assertLessEqual(len(raw), 400)
        self.assertNotIn("preview", result)
        if result.get("truncated"):
            self.assertTrue(
                "records" in result or "content" in result
            )
        if "records" in result:
            self.assertLessEqual(
                len(json.dumps(result["records"])),
                400,
            )

    def test_subagent_defaults_to_own_transcript(self) -> None:
        child = chat_store.create_session(
            kind="subagent",
            chat_key=session.current_runtime().key,
            agent_id="abcd",
            parent_session_id=self.main,
        )
        chat_store.append_item(
            child,
            {"role": "user", "content": "child-only-token"},
            source="delegation",
        )
        result = chat_history_tool(
            {"action": "search", "query": "child-only-token"},
            execution_context={"session_id": child},
        )
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["hits"][0]["session_id"], child)

    async def test_subagent_execute_tool_passes_session_context(
        self,
    ) -> None:
        from tools import execute_tool

        child = chat_store.create_session(
            kind="subagent",
            chat_key=session.current_runtime().key,
            agent_id="abcd",
            parent_session_id=self.main,
        )
        chat_store.append_item(
            child,
            {"role": "user", "content": "via-execute-tool"},
            source="delegation",
        )
        session.state["current_session_id"] = self.main
        result = await execute_tool(
            {
                "name": "chat_history",
                "arguments": json.dumps(
                    {
                        "action": "search",
                        "query": "via-execute-tool",
                    }
                ),
            },
            execution_context={
                "session_id": child,
                "todos": [],
            },
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            result["hits"][0]["session_id"],
            child,
        )

    def test_invalid_session_id_rejected(self) -> None:
        result = chat_history_tool(
            {
                "action": "read",
                "session_id": "../etc/passwd",
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("invalid session_id", result["error"])

    def test_list_sessions_payload_fits_max_output(self) -> None:
        import tools as tools_mod
        for _ in range(40):
            chat_store.create_session(
                kind="main",
                chat_key=session.current_runtime().key,
            )
        with patch.object(tools_mod, "MAX_TOOL_OUTPUT", 500):
            result = chat_history_tool({"action": "list", "limit": 100})
        raw = json.dumps(result, ensure_ascii=False)
        self.assertLessEqual(len(raw), 500)
        self.assertTrue(result.get("truncated") or "sessions" in result)
        self.assertNotIn("preview", result)

    def test_final_content_truncation_stays_within_budget(
        self,
    ) -> None:
        import tools as tools_mod
        # One oversized record exercises final whole-payload truncation.
        chat_store.append_item(
            self.main,
            {
                "role": "user",
                "content": "X" * 2000,
            },
            source="user",
        )
        with patch.object(tools_mod, "MAX_TOOL_OUTPUT", 120):
            result = chat_history_tool(
                {
                    "action": "read",
                    "session_id": self.main,
                    "limit": 1,
                }
            )
        raw = json.dumps(result, ensure_ascii=False)
        self.assertLessEqual(len(raw), 120)
        self.assertTrue(result["truncated"])


if __name__ == "__main__":
    unittest.main()
