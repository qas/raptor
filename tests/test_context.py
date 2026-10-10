"""Active context and checkpoint compaction tests."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-context-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ["MODEL_CONTEXT_TOKENS"] = "8192"
os.environ["COMPACT_KEEP_RECENT_TOKENS"] = "50"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from raptor.state import chat_store
from raptor.agent import context
from raptor.model.responses import ContextLengthError, TransientResponsesError

TEST_MODEL_TARGET = {"provider_id": "local", "model": "test-model"}


class ContextTests(unittest.IsolatedAsyncioTestCase):
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
        self.session_id = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")

    def _add_user(self, text: str) -> dict:
        return chat_store.append_item(
            self.session_id,
            {"role": "user", "content": text},
            source="user",
        )

    def _add_assistant(self, text: str) -> dict:
        return chat_store.append_item(
            self.session_id,
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            },
            source="assistant",
        )

    def test_no_checkpoint_all_native_items_active(self) -> None:
        self._add_user("a")
        self._add_assistant("b")
        active = context.build_active_context(self.session_id)
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0]["content"], "a")

    def test_instruction_roles_are_normalized_in_chronological_input(
        self,
    ) -> None:
        chat_store.append_item(
            self.session_id,
            {"role": "developer", "content": "runtime event"},
            source="runtime",
        )

        active = context.build_active_context(self.session_id)

        self.assertEqual(active, [{"role": "user", "content": "runtime event"}])

    def test_checkpoint_tail_normalizes_instruction_roles(self) -> None:
        old = self._add_user("before checkpoint")
        chat_store.append_checkpoint(
            self.session_id,
            summary="checkpoint body",
            through_seq=old["seq"],
        )
        chat_store.append_item(
            self.session_id,
            {"role": "developer", "content": "runtime event"},
            source="runtime",
        )

        active = context.build_active_context(self.session_id)

        self.assertEqual(active[-1], {"role": "user", "content": "runtime event"})

    def test_checkpoint_summary_plus_tail(self) -> None:
        self._add_user("old")
        old = self._add_assistant("old-reply")
        chat_store.append_checkpoint(
            self.session_id,
            summary="checkpoint body",
            through_seq=old["seq"],
        )
        self._add_user("new")
        active = context.build_active_context(self.session_id)
        self.assertEqual(len(active), 2)
        self.assertIn("checkpoint body", active[0]["content"])
        self.assertEqual(active[1]["content"], "new")

    def test_compaction_input_is_one_plain_user_message(self) -> None:
        records = [
            {
                "type": "item",
                "seq": 2,
                "item": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "c1",
                    "arguments": "{}",
                },
            },
            {
                "type": "item",
                "seq": 3,
                "item": {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": '{"ok":true}',
                },
            },
        ]
        rendered = chat_store.render_compaction_records(records)
        self.assertIn("FUNCTION_CALL", rendered)
        self.assertIn("FUNCTION_RESULT", rendered)
        payload = [{"role": "user", "content": rendered}]
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["role"], "user")

    async def test_normal_compaction_preserves_recent_user_tail(
        self,
    ) -> None:
        for i in range(8):
            self._add_user(f"u{i} " + ("x" * 40))
            self._add_assistant(f"a{i} " + ("y" * 40))
        before = len(chat_store.read_events(self.session_id))
        captured: list[list] = []

        async def create(items, instructions):
            captured.append(copy.deepcopy(items))
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "summary-A",
                            }
                        ],
                    }
                ]
            }

        def estimate(items, instructions):
            return 10

        with patch.object(
            context,
            "COMPACT_KEEP_RECENT_TOKENS",
            80,
        ):
            ok = await context.compact_session(
                self.session_id,
                estimate_compaction_request=estimate,
                create_compaction_response=create,
                reason="threshold",
            )
        self.assertTrue(ok)
        self.assertEqual(len(chat_store.read_events(self.session_id)), before + 2)
        cp = chat_store.latest_checkpoint(self.session_id)
        self.assertIsNotNone(cp)
        active = context.build_active_context(self.session_id)
        summary_idx = next(
            i
            for i, item in enumerate(active)
            if "summary-A" in str(item.get("content", ""))
        )
        self.assertGreater(summary_idx, 0)
        self.assertEqual(active[0]["content"], "u0 " + ("x" * 40))
        self.assertTrue(
            any(
                item.get("role") == "user"
                for item in active[summary_idx + 1:]
            )
        )
        for item in active[summary_idx + 1:]:
            if item.get("role") == "user":
                self.assertEqual(item, active[summary_idx + 1])
                break
        self.assertEqual(len(captured[0]), 1)
        self.assertEqual(captured[0][0]["role"], "user")

    async def test_forced_compaction_retires_recent_native_tail(self) -> None:
        latest = self._add_user("only recent turn")
        calls = 0

        async def create(items, instructions):
            nonlocal calls
            calls += 1
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "summary"},
                        ],
                    }
                ]
            }

        with patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 20_000):
            normal = await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="threshold",
            )
            forced = await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                force=True,
                reason="manual",
            )
            repeated = await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                force=True,
                reason="manual",
            )

        self.assertFalse(normal)
        self.assertTrue(forced)
        self.assertFalse(repeated)
        self.assertEqual(calls, 1)
        checkpoint = chat_store.latest_checkpoint(self.session_id)
        self.assertEqual(checkpoint["through_seq"], latest["seq"])

    async def test_new_checkpoint_merges_previous(
        self,
    ) -> None:
        self._add_user("first")
        first = self._add_assistant("r1")
        chat_store.append_checkpoint(
            self.session_id,
            summary="summary A",
            through_seq=first["seq"],
        )
        self._add_user("middle")
        mid = self._add_assistant("r2")
        self._add_user("tail")
        seen: list[str] = []

        async def create(items, instructions):
            seen.append(items[0]["content"])
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "summary B",
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 5),
            patch.object(
                context,
                "context_input_budget",
                lambda: 4000,
            ),
        ):
            ok = await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="threshold",
            )
        self.assertTrue(ok)
        self.assertIn("summary A", seen[0])
        self.assertIn("middle", seen[0])
        cp = chat_store.latest_checkpoint(self.session_id)
        self.assertEqual(cp["summary"], "summary B")
        self.assertGreaterEqual(cp["through_seq"], mid["seq"])

    async def test_oversized_compaction_keeps_contiguous_oldest_prefix(
        self,
    ) -> None:
        oldest = self._add_user("must-not-be-skipped")
        for i in range(5):
            self._add_user(f"msg-{i}")
        sizes = {"n": 0}

        def estimate(items, instructions):
            sizes["n"] += 1
            content = items[0]["content"]
            if "must-not-be-skipped" in content:
                return 10_000
            return 10

        async def create(items, instructions):
            self.assertIn("must-not-be-skipped", items[0]["content"])
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(
                context,
                "context_input_budget",
                lambda: 2000,
            ),
        ):
            ok = await context.compact_session(
                self.session_id,
                estimate_compaction_request=estimate,
                create_compaction_response=create,
                reason="overflow",
            )
        self.assertTrue(ok)
        self.assertGreaterEqual(sizes["n"], 1)
        checkpoint = chat_store.latest_checkpoint(self.session_id)
        self.assertEqual(checkpoint["through_seq"], oldest["seq"])
        self.assertEqual(checkpoint["input_to_seq"], oldest["seq"])

    async def test_compaction_overflow_retries_once(
        self,
    ) -> None:
        first = self._add_user("one")
        self._add_user("two")
        self._add_user("three")
        calls = {"n": 0}
        rendered: list[str] = []

        async def create(items, instructions):
            calls["n"] += 1
            rendered.append(items[0]["content"])
            if calls["n"] == 1:
                raise ContextLengthError("too long")
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "recovered",
                            }
                        ],
                    }
                ]
            }

        with patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0):
            ok = await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="overflow",
            )
        self.assertTrue(ok)
        self.assertEqual(calls["n"], 2)
        self.assertIn("one", rendered[1])
        self.assertNotIn("two", rendered[1])
        self.assertEqual(
            chat_store.latest_checkpoint(self.session_id)["through_seq"],
            first["seq"],
        )

    async def test_internal_continuations_are_not_user_anchors(self) -> None:
        real = self._add_user("authoritative user constraint")
        chat_store.append_item(
            self.session_id,
            {
                "role": "user",
                "content": context.CHECKPOINT_CONTINUATION_INPUT["content"],
            },
            source="internal",
        )

        async def create(items, instructions):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "checkpoint"}
                        ],
                    }
                ]
            }

        with patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0):
            await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="overflow",
                force=True,
            )

        anchors = chat_store.latest_checkpoint(self.session_id)["anchors"]
        self.assertEqual([anchor["seq"] for anchor in anchors], [real["seq"]])

    async def test_second_compaction_overflow_propagates(
        self,
    ) -> None:
        self._add_user("one")
        self._add_user("two")

        async def create(items, instructions):
            raise ContextLengthError("still too long")

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            self.assertRaises(ContextLengthError),
        ):
            await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="overflow",
            )

    async def test_repeated_compaction_covers_every_record_in_order(
        self,
    ) -> None:
        events = [self._add_user(f"request-{i}") for i in range(9)]
        rendered_calls: list[str] = []

        def estimate(items, instructions):
            # A prior checkpoint plus at most two new native records fits.
            native_records = items[0]["content"].count("] USER\n")
            return 10 if native_records <= 2 else 10_000

        async def create(items, instructions):
            rendered_calls.append(items[0]["content"])
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"merged checkpoint {len(rendered_calls)}",
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(context, "context_input_budget", lambda: 2_000),
            patch.object(context, "compaction_generation_budget", lambda: 512),
        ):
            while (
                int(
                    (chat_store.latest_checkpoint(self.session_id) or {}).get(
                        "through_seq"
                    )
                    or 0
                )
                < events[-1]["seq"]
            ):
                self.assertTrue(
                    await context.compact_session(
                        self.session_id,
                        estimate_compaction_request=estimate,
                        create_compaction_response=create,
                        reason="overflow",
                        force=True,
                    )
                )

        for i in range(9):
            appearances = sum(
                f"] USER\nrequest-{i}" in rendered
                for rendered in rendered_calls
            )
            self.assertEqual(appearances, 1, f"request-{i} coverage")
        checkpoint = chat_store.latest_checkpoint(self.session_id)
        self.assertEqual(checkpoint["through_seq"], events[-1]["seq"])
        active = context.build_active_context(self.session_id)
        self.assertEqual(
            [item["content"] for item in active[:-1]],
            [f"request-{i}" for i in range(9)],
        )
        self.assertIn("merged checkpoint", active[-1]["content"])

    async def test_reasoning_only_compaction_retries_for_visible_summary(
        self,
    ) -> None:
        self._add_user("preserve this")
        instructions_seen: list[str] = []
        reasoning_only = {
            "id": "response-reasoning-only",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "status": "completed",
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "sensitive internal reasoning",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 50_918,
                "output_tokens": 1_200,
                "total_tokens": 52_118,
            },
        }

        async def create(items, instructions):
            instructions_seen.append(instructions)
            if len(instructions_seen) == 1:
                return reasoning_only
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "visible checkpoint",
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(context, "log_event") as logged,
        ):
            ok = await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="overflow",
            )

        self.assertTrue(ok)
        self.assertEqual(len(instructions_seen), 2)
        self.assertIn(
            "Emit the checkpoint as the final answer now",
            instructions_seen[1],
        )
        self.assertEqual(
            chat_store.latest_checkpoint(self.session_id)["summary"],
            "visible checkpoint",
        )
        diagnostic = logged.call_args_list[0].args[2]
        self.assertNotIn("response", diagnostic)
        self.assertNotIn("sensitive internal reasoning", str(diagnostic))
        self.assertEqual(diagnostic["attempt"], 1)
        self.assertEqual(diagnostic["output_items"][0]["type"], "reasoning")
        self.assertEqual(diagnostic["usage"]["output_tokens"], 1_200)

    async def test_two_reasoning_only_compactions_fail_atomically(
        self,
    ) -> None:
        self._add_user("do not retire without a checkpoint")
        calls = 0

        async def create(items, instructions):
            nonlocal calls
            calls += 1
            return {
                "status": "completed",
                "output": [{"type": "reasoning", "content": []}],
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(context, "log_event"),
            self.assertRaisesRegex(
                TransientResponsesError,
                "compaction returned no summary",
            ),
        ):
            await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="overflow",
            )

        self.assertEqual(calls, 2)
        self.assertIsNone(chat_store.latest_checkpoint(self.session_id))

    async def test_request_retry_rebuilds_once(self) -> None:
        work = [{"role": "user", "content": "hi"}]
        calls = {"n": 0}

        async def request(items):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ContextLengthError("overflow")
            return {"ok": True, "items": list(items)}

        async def compact(items):
            return [{"role": "user", "content": "Continue"}]

        result = await context.request_with_checkpoint_retry(
            work,
            request_fn=request,
            compact_fn=compact,
            overflow_error=ContextLengthError,
        )
        self.assertEqual(calls["n"], 2)
        self.assertEqual(result["items"][0]["content"], "Continue")

    async def test_second_request_overflow_propagates(self) -> None:
        work = [{"role": "user", "content": "hi"}]

        async def request(items):
            raise ContextLengthError("overflow")

        async def compact(items):
            return [{"role": "user", "content": "Continue"}]

        with self.assertRaises(ContextLengthError):
            await context.request_with_checkpoint_retry(
                work,
                request_fn=request,
                compact_fn=compact,
                overflow_error=ContextLengthError,
            )


class ContextLengthParseTests(unittest.TestCase):
    def _response(
        self,
        status: int,
        body: dict | str,
    ):
        import httpx
        if isinstance(body, dict):
            content = __import__("json").dumps(body).encode()
        else:
            content = str(body).encode()
        return httpx.Response(
            status,
            content=content,
            request=httpx.Request("POST", "http://test/responses"),
        )

    def test_parses_underscored_token_aliases(self) -> None:
        from raptor.model.responses import parse_context_length_error
        response = self._response(
            400,
            {
                "error": {
                    "type": "exceed_context_size_error",
                    "message": "Context size has been exceeded.",
                    "n_prompt_tokens": 314468,
                    "n_ctx": 131072,
                }
            },
        )
        err = parse_context_length_error(response)
        self.assertIsNotNone(err)
        assert err is not None
        self.assertEqual(err.prompt_tokens, 314468)
        self.assertEqual(err.context_tokens, 131072)

    def test_classifies_500_context_exceeded_message(self) -> None:
        from raptor.model.responses import parse_context_length_error
        response = self._response(
            500,
            {
                "error": {
                    "message": "Context size has been exceeded.",
                }
            },
        )
        err = parse_context_length_error(response)
        self.assertIsNotNone(err)

    def test_ignores_unrelated_500(self) -> None:
        from raptor.model.responses import parse_context_length_error
        response = self._response(
            500,
            {"error": {"message": "internal server error"}},
        )
        self.assertIsNone(parse_context_length_error(response))

    def test_accepts_413(self) -> None:
        from raptor.model.responses import parse_context_length_error
        response = self._response(
            413,
            {
                "error": {
                    "type": "exceed_context_size_error",
                    "message": "payload too large",
                    "prompt_tokens": 99,
                    "context_tokens": 50,
                }
            },
        )
        err = parse_context_length_error(response)
        self.assertIsNotNone(err)
        assert err is not None
        self.assertEqual(err.prompt_tokens, 99)
        self.assertEqual(err.context_tokens, 50)


class EnsureUnderBudgetTests(unittest.IsolatedAsyncioTestCase):
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
        self.session_id = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")

    async def test_compaction_request_uses_its_own_input_budget(self) -> None:
        chat_store.append_item(
            self.session_id,
            {"role": "user", "content": "compact me"},
            source="user",
        )
        captured: dict[str, object] = {}

        async def compact(*_args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("captured")

        with patch.object(context, "compact_session", compact):
            with self.assertRaisesRegex(RuntimeError, "captured"):
                await context.ensure_context_under_budget(
                    self.session_id,
                    estimate_active_fn=lambda items: 100 if items else 0,
                    estimate_compaction_request=lambda *_args: 1,
                    create_compaction_response=AsyncMock(),
                    force=True,
                    input_budget=50,
                    compaction_input_budget=20,
                )

        self.assertEqual(captured["input_budget"], 20)

    async def test_real_estimator_compacts_huge_archive_under_budget(
        self,
    ) -> None:
        """Regression: rebuilt request must actually shrink under budget.

        Guards the 314k→314k failure where compaction ran but the next
        request still used an oversized active context.
        """
        from raptor.model.responses import estimate_response_request_tokens
        budget = 10_000
        # Medium blobs: each fits in the compaction request window, but
        # dozens of them make the full active request far over budget.
        chunk = ("ARCHIVE-" + ("y" * 80) + "\n") * 40
        for i in range(30):
            chat_store.append_item(
                self.session_id,
                {
                    "role": "user",
                    "content": f"user-{i}\n{chunk}",
                },
                source="user",
            )
            chat_store.append_item(
                self.session_id,
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"assistant-{i}\n{chunk}",
                        }
                    ],
                },
                source="assistant",
            )

        def estimate_active(items):
            return estimate_response_request_tokens(items)

        def estimate_compaction(items, instructions):
            return estimate_response_request_tokens(
                items,
                tools=None,
                extra_instructions=instructions,
                max_output_tokens=1024,
            )

        async def create(items, instructions):
            # Compaction model returns a short dense checkpoint.
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Checkpoint: archived large tool "
                                    "and chat payloads; continue the "
                                    "active coding task."
                                ),
                            }
                        ],
                    }
                ]
            }

        before = estimate_active(
            context.build_active_context(self.session_id)
        )
        self.assertGreater(
            before,
            budget * 2,
            msg=(
                f"precondition failed: before={before} "
                f"should greatly exceed budget={budget}"
            ),
        )

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(
                context,
                "context_input_budget",
                lambda: budget,
            ),
        ):
            work = await context.ensure_context_under_budget(
                self.session_id,
                estimate_active_fn=estimate_active,
                estimate_compaction_request=estimate_compaction,
                create_compaction_response=create,
                reason="threshold",
                log_source="agent",
            )

        after = estimate_active(work)
        self.assertLess(
            after,
            budget,
            msg=(
                f"post-compact request still oversized: "
                f"after={after} budget={budget} before={before}"
            ),
        )
        self.assertLess(after, before // 4)
        cp = chat_store.latest_checkpoint(self.session_id)
        self.assertIsNotNone(cp)
        self.assertTrue(
            any(
                "Checkpoint:" in str(item.get("content", ""))
                for item in work
            )
        )

    async def test_single_new_turn_escalates_to_whole_window_compaction(
        self,
    ) -> None:
        """Regression for /new + chat_history exceeding the threshold.

        A recent-tail policy cannot split a sole user turn. The threshold
        pass therefore makes no progress and must retry as forced overflow
        compaction instead of raising immediately.
        """
        huge = "new-session-tool-context-" * 500
        chat_store.append_item(
            self.session_id,
            {"role": "user", "content": huge},
            source="user",
        )
        create_calls = 0

        def estimate_active(items):
            return 100 + sum(context.estimate_tokens(item) for item in items)

        async def create(items, instructions):
            nonlocal create_calls
            create_calls += 1
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Take over the archived chat.",
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 20_000),
            patch.object(context, "context_input_budget", lambda: 1_000),
        ):
            work = await context.ensure_context_under_budget(
                self.session_id,
                estimate_active_fn=estimate_active,
                estimate_compaction_request=lambda *_: 100,
                create_compaction_response=create,
                reason="threshold",
                log_source="agent",
            )

        self.assertEqual(create_calls, 1)
        self.assertLess(estimate_active(work), 1_000)
        self.assertIn("Take over", work[0]["content"])

    async def test_oversized_existing_checkpoint_is_recompacted(
        self,
    ) -> None:
        event = chat_store.append_item(
            self.session_id,
            {"role": "user", "content": "archived source"},
            source="user",
        )
        chat_store.append_checkpoint(
            self.session_id,
            summary="oversized checkpoint " * 500,
            through_seq=event["seq"],
        )

        def estimate_active(items):
            return 100 + sum(context.estimate_tokens(item) for item in items)

        async def create(items, instructions):
            self.assertIn("oversized checkpoint", items[0]["content"])
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Recompacted oversized checkpoint.",
                            }
                        ],
                    }
                ]
            }

        with patch.object(
            context,
            "context_input_budget",
            lambda: 1_000,
        ):
            work = await context.ensure_context_under_budget(
                self.session_id,
                estimate_active_fn=estimate_active,
                estimate_compaction_request=lambda *_: 100,
                create_compaction_response=create,
                reason="threshold",
                log_source="agent",
            )

        self.assertLess(estimate_active(work), 1_000)
        self.assertTrue(
            any(
                "Recompacted" in str(item.get("content", ""))
                for item in work
            )
        )

    async def test_model_cannot_return_unbounded_checkpoint_summary(
        self,
    ) -> None:
        chat_store.append_item(
            self.session_id,
            {"role": "user", "content": "source " * 5_000},
            source="user",
        )

        def estimate_active(items):
            return 2_500 + sum(
                context.estimate_tokens(item) for item in items
            )

        async def create(items, instructions):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "SUMMARY-START\n"
                                    + ("unbounded " * 50_000)
                                    + "\nSUMMARY-END"
                                ),
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(context, "COMPACTION_OUTPUT_TOKENS", 500),
            patch.object(context, "context_input_budget", lambda: 3_000),
        ):
            work = await context.ensure_context_under_budget(
                self.session_id,
                estimate_active_fn=estimate_active,
                estimate_compaction_request=lambda *_: 100,
                create_compaction_response=create,
                reason="threshold",
                log_source="agent",
            )

        summary = chat_store.latest_checkpoint(self.session_id)["summary"]
        self.assertLessEqual(context.estimate_tokens(summary), 500)
        self.assertIn("SUMMARY-START", summary)
        self.assertIn("SUMMARY-END", summary)
        self.assertIn("checkpoint middle truncated", summary)
        self.assertLess(estimate_active(work), 3_000)
        self.assertEqual(
            chat_store.latest_checkpoint(self.session_id)["reason"],
            "budget_fit",
        )

    async def test_oversized_single_record_compacts_via_truncated_render(
        self,
    ) -> None:
        """One archive blob too big for full compaction input still retires.

        Rendering truncates for the checkpoint request only; JSONL stays full.
        """
        from raptor.model.responses import estimate_response_request_tokens
        budget = 10_000
        huge = ("OVERSIZED-RECORD-" + ("Q" * 200) + "\n") * 200
        event = chat_store.append_item(
            self.session_id,
            {"role": "user", "content": huge},
            source="user",
        )

        def estimate_active(items):
            return estimate_response_request_tokens(items)

        def estimate_compaction(items, instructions):
            return estimate_response_request_tokens(
                items,
                tools=None,
                extra_instructions=instructions,
                max_output_tokens=1024,
            )

        async def create(items, instructions):
            content = items[0]["content"]
            self.assertIn(
                "truncated for checkpoint generation",
                content,
            )
            self.assertNotIn(huge, content)
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Checkpoint: oversized archive "
                                    "record summarized."
                                ),
                            }
                        ],
                    }
                ]
            }

        before = estimate_active(
            context.build_active_context(self.session_id)
        )
        self.assertGreater(before, budget)

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(
                context,
                "context_input_budget",
                lambda: budget,
            ),
            patch.object(
                chat_store,
                "COMPACTION_MAX_RECORD_CHARS",
                4_000,
            ),
        ):
            work = await context.ensure_context_under_budget(
                self.session_id,
                estimate_active_fn=estimate_active,
                estimate_compaction_request=estimate_compaction,
                create_compaction_response=create,
                reason="threshold",
                log_source="agent",
            )

        after = estimate_active(work)
        self.assertLess(after, budget)
        stored = next(
            e["item"]["content"]
            for e in chat_store.item_events(self.session_id)
            if e.get("seq") == event["seq"]
        )
        self.assertEqual(stored, huge)
        self.assertEqual(len(stored), len(huge))

    async def test_raises_when_still_over_budget(self) -> None:
        chat_store.append_item(
            self.session_id,
            {"role": "user", "content": "only"},
            source="user",
        )

        async def create(items, instructions):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "summary",
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0),
            patch.object(
                context,
                "context_input_budget",
                lambda: 50,
            ),
            self.assertRaises(RuntimeError),
        ):
            await context.ensure_context_under_budget(
                self.session_id,
                estimate_active_fn=lambda _items: 9999,
                estimate_compaction_request=lambda *_: 10,
                create_compaction_response=create,
                reason="threshold",
                max_passes=2,
            )


class RecentTailTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        ContextTests.setUp(self)

    def _user(self, text: str) -> dict:
        return chat_store.append_item(
            self.session_id,
            {"role": "user", "content": text},
            source="user",
        )

    def _assistant(self, text: str) -> dict:
        return chat_store.append_item(
            self.session_id,
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            },
            source="assistant",
        )

    def _call(self, call_id: str) -> dict:
        return chat_store.append_item(
            self.session_id,
            {
                "type": "function_call",
                "name": "shell",
                "call_id": call_id,
                "arguments": "{}",
            },
            source="assistant",
        )

    def _result(self, call_id: str, output: str) -> dict:
        return chat_store.append_item(
            self.session_id,
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
            source="tool",
        )

    def _texts(self, items: list[dict]) -> list[str]:
        found: list[str] = []
        for item in items:
            content = item.get("content")
            if isinstance(content, str):
                found.append(content)
            elif isinstance(content, list):
                found.extend(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict)
                )
        return found

    async def _compact(self, budget: int, **kwargs) -> bool:
        async def create(_items, _instructions):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "summary"}
                        ],
                    }
                ]
            }
        with patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", budget):
            return await context.compact_session(
                self.session_id,
                estimate_compaction_request=lambda *_args: 10,
                create_compaction_response=create,
                **kwargs,
            )

    def _tail(self, active: list[dict]) -> list[dict]:
        for index, item in enumerate(active):
            content = str(item.get("content") or "")
            if content.startswith("Raptor conversation checkpoint "):
                return active[index + 1:]
        return active

    async def test_newest_groups_stay_chronological(self) -> None:
        self._user("user-0")
        self._assistant("assistant-0")
        self._user("user-1")
        newest_user = {"role": "user", "content": "user-1"}
        newest_assistant = {
            "type": "message",
            "content": [{"type": "output_text", "text": "assistant-1"}],
        }
        self._assistant("assistant-1")
        budget = (
            context.estimate_tokens(newest_user)
            + context.estimate_tokens(newest_assistant)
        )
        self.assertTrue(await self._compact(budget))
        active = context.build_active_context(self.session_id)
        self.assertEqual(active[0]["content"], "user-0")
        self.assertEqual(
            self._texts(self._tail(active)),
            ["user-1", "assistant-1"],
        )
        checkpoint = chat_store.latest_checkpoint(self.session_id)
        self.assertNotIn("assistant_anchor", checkpoint)
        self.assertNotIn("assistant_reply", checkpoint)
        self.assertNotIn("tail", checkpoint)

    async def test_boundary_message_stays_exact_in_the_archive(self) -> None:
        from raptor.model.model_providers import ModelTarget
        from raptor.state import session
        from raptor.tools import chat_history_tool
        previous_target = session._default_model_target
        previous_chat = copy.deepcopy(
            session.CHAT_DEFAULT_STATE.get("model_target")
        )
        previous_default = copy.deepcopy(
            session.DEFAULT_STATE.get("model_target")
        )

        def restore_target() -> None:
            session._default_model_target = previous_target
            session.CHAT_DEFAULT_STATE["model_target"] = previous_chat
            session.DEFAULT_STATE["model_target"] = previous_default

        self.addCleanup(restore_target)
        target = ModelTarget("local", "test-model")
        session.set_default_model_target(target)
        with session.bound_chat("telegram:tail"):
            runtime = session.current_runtime()
            self.session_id = str(runtime.state["current_session_id"])
            huge = (
                "USER_START " + ("u" * 1800) + "USER_MIDDLE"
                + ("u" * 1800) + " USER_END"
            )
            self._user("anchor-me")
            self._user(huge)
            assistant = {
                "type": "message",
                "content": [{"type": "output_text", "text": "small-note"}],
            }
            self._assistant("small-note")
            later = {"role": "user", "content": "drop-turn"}
            self._user("drop-turn")
            budget = (
                context.estimate_tokens(later)
                + context.estimate_tokens(assistant)
                + 80
            )
            self.assertTrue(await self._compact(budget))
            active = json.dumps(
                context.build_active_context(self.session_id)
            )
            self.assertIn("tokens truncated", active)
            self.assertIn("USER_START", active)
            self.assertIn("USER_END", active)
            self.assertNotIn("USER_MIDDLE", active)
            self.assertIn("anchor-me", active)
            archived = [
                event["item"] for event in chat_store.item_events(self.session_id)
            ]
            self.assertIn(huge, json.dumps(archived))
            found = chat_history_tool(
                {
                    "action": "search",
                    "query": "USER_MIDDLE",
                    "session_id": self.session_id,
                },
                execution_context={"session_id": self.session_id},
            )
            self.assertTrue(found["ok"])
            self.assertGreaterEqual(len(found["hits"]), 1)
            forked, _copied = chat_store.fork_session_before_last_user_turns(
                self.session_id,
                turns=1,
                chat_key=runtime.key,
                model_target=target.to_dict(),
            )
            forked_items = [
                event["item"] for event in chat_store.item_events(forked)
            ]
            self.assertIn(huge, json.dumps(forked_items))
            self.assertNotIn("tokens truncated", json.dumps(forked_items))
            forked_active = json.dumps(context.build_active_context(forked))
            self.assertIn("USER_MIDDLE", forked_active)
            self.assertNotIn("drop-turn", forked_active)
            checkpoint = chat_store.latest_checkpoint(self.session_id)
            self.assertIn("tail", checkpoint)
            self.assertNotIn("assistant_anchor", checkpoint)

    async def test_interleaved_tool_groups_stay_whole(self) -> None:
        self._user("old-request " + ("o" * 500))
        self._call("alpha")
        self._call("beta")
        self._result("alpha", "alpha-out")
        self._result("beta", "beta-out")
        keep = {"role": "user", "content": "keep-me"}
        self._user("keep-me")
        tool_cost = sum(
            context.estimate_tokens(event["item"])
            for event in chat_store.item_events(self.session_id)[1:5]
        )
        budget = context.estimate_tokens(keep) + tool_cost
        self.assertTrue(await self._compact(budget))
        tail = self._tail(context.build_active_context(self.session_id))
        self.assertEqual(
            [item.get("call_id") for item in tail if item.get("call_id")],
            ["alpha", "beta", "alpha", "beta"],
        )
        self.assertEqual(tail[-1]["content"], "keep-me")
        self.assertNotIn("old-request", json.dumps(tail))
        calls = [
            item.get("type") for item in tail
            if str(item.get("type") or "").startswith("function_")
        ]
        self.assertEqual(calls.count("function_call"), 2)
        self.assertEqual(calls.count("function_call_output"), 2)

    async def test_oversized_tool_group_is_dropped_whole(self) -> None:
        self._call("alpha")
        self._call("beta")
        self._result("beta", "beta-out")
        self._result("alpha", "alpha-out")
        keep = {"role": "user", "content": "keep-me"}
        self._user("keep-me")
        self.assertTrue(await self._compact(context.estimate_tokens(keep)))
        tail = self._tail(context.build_active_context(self.session_id))
        self.assertEqual([item.get("content") for item in tail], ["keep-me"])
        archived = json.dumps(
            [event["item"] for event in chat_store.item_events(self.session_id)]
        )
        self.assertIn("alpha-out", archived)
        self.assertIn("beta-out", archived)

    async def test_manual_and_forced_compaction_keep_no_tail(self) -> None:
        self._user("old-bulk " + ("b" * 800))
        self._user("manual-user")
        self._assistant("manual-note")
        self._call("keep")
        self._result("keep", "KEEP_ME")
        keep_user = {"role": "user", "content": "manual-user"}
        note = {
            "type": "message",
            "content": [{"type": "output_text", "text": "manual-note"}],
        }
        call = {
            "type": "function_call",
            "name": "shell",
            "call_id": "keep",
            "arguments": "{}",
        }
        result = {
            "type": "function_call_output",
            "call_id": "keep",
            "output": "KEEP_ME",
        }
        budget = (
            context.estimate_tokens(keep_user)
            + context.estimate_tokens(note)
            + context.estimate_tokens(call)
            + context.estimate_tokens(result)
        )
        self.assertTrue(await self._compact(budget))
        self.assertIn(
            "KEEP_ME",
            json.dumps(self._tail(
                context.build_active_context(self.session_id)
            )),
        )
        self.assertTrue(await self._compact(budget, force=True, reason="overflow"))
        self.assertEqual(
            self._tail(context.build_active_context(self.session_id)),
            [],
        )
        self._user("after")
        self.assertTrue(
            await self._compact(20_000, force=True, reason="manual")
        )
        checkpoint = chat_store.latest_checkpoint(self.session_id)
        self.assertEqual(checkpoint["reason"], "manual")
        self.assertNotIn("tail", checkpoint)
        self.assertNotIn(
            "after",
            json.dumps(self._tail(
                context.build_active_context(self.session_id)
            )),
        )
        self.assertIn("manual-user", json.dumps(
            context.build_active_context(self.session_id)
        ))

    async def test_summary_prefix_does_not_split_tool_groups(self) -> None:
        self._call("first")
        self._result("first", "first-out")
        self._call("second")
        self._result("second", "second-out")
        self._user("later-user")

        def estimate(items, _instructions):
            text = items[0]["content"]
            if text.count("FUNCTION_CALL") > 1 or "later-user" in text:
                return 10_000
            return 10

        async def create(items, _instructions):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "summary"}
                        ],
                    }
                ]
            }

        with patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 0):
            self.assertTrue(
                await context.compact_session(
                    self.session_id,
                    estimate_compaction_request=estimate,
                    create_compaction_response=create,
                    reason="overflow",
                    force=True,
                )
            )
        tail = self._tail(context.build_active_context(self.session_id))
        ids = [item.get("call_id") for item in tail if item.get("call_id")]
        self.assertEqual(ids, ["second", "second"])
        self.assertEqual(tail[-1]["content"], "later-user")
        self.assertNotIn("first-out", json.dumps(tail))

    def _tool_ids(self, items: list[dict]) -> tuple[list[str], list[str]]:
        calls: list[str] = []
        outputs: list[str] = []
        for item in items:
            call_id = str(item.get("call_id") or "")
            if item.get("type") == "function_call" and call_id:
                calls.append(call_id)
            elif item.get("type") == "function_call_output" and call_id:
                outputs.append(call_id)
        return calls, outputs

    async def test_incomplete_tool_fragments_are_summarized(self) -> None:
        rendered: list[str] = []
        self._result("ghost", "ghost-only")
        self._call("orphan")
        self._call("paired")
        self._result("paired", "paired-out")
        keep = {"role": "user", "content": "keep-me"}
        self._user("keep-me")
        stored = chat_store.item_events(self.session_id)
        pair_cost = (
            context.estimate_tokens(stored[2]["item"])
            + context.estimate_tokens(stored[3]["item"])
        )
        budget = context.estimate_tokens(keep) + pair_cost

        async def create(items, _instructions):
            rendered.append(items[0]["content"])
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "summary"}
                        ],
                    }
                ]
            }

        with patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", budget):
            self.assertTrue(
                await context.compact_session(
                    self.session_id,
                    estimate_compaction_request=lambda *_args: 10,
                    create_compaction_response=create,
                    reason="threshold",
                )
            )
        active = context.build_active_context(self.session_id)
        calls, outputs = self._tool_ids(active)
        self.assertEqual(calls, ["paired"])
        self.assertEqual(outputs, ["paired"])
        self.assertNotIn("orphan", json.dumps(self._tail(active)))
        self.assertNotIn("ghost-only", json.dumps(self._tail(active)))
        self.assertIn("call_id=orphan", rendered[0])
        self.assertIn("ghost-only", rendered[0])
        archived = json.dumps(
            [event["item"] for event in chat_store.item_events(self.session_id)]
        )
        self.assertIn("orphan", archived)
        progress = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        chat_store.append_item(
            progress,
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "solo",
                "arguments": "{}",
            },
            source="assistant",
        )

        async def create_progress(items, _instructions):
            rendered.append(items[0]["content"])
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "summary"}
                        ],
                    }
                ]
            }

        with patch.object(context, "COMPACT_KEEP_RECENT_TOKENS", 20_000):
            self.assertTrue(
                await context.compact_session(
                    progress,
                    estimate_compaction_request=lambda *_args: 100_000,
                    create_compaction_response=create_progress,
                    reason="threshold",
                )
            )
        progress_active = context.build_active_context(progress)
        progress_calls, progress_outputs = self._tool_ids(progress_active)
        self.assertEqual(progress_calls, [])
        self.assertEqual(progress_outputs, [])
        self.assertIn("call_id=solo", rendered[-1])

    async def test_multipart_boundary_keeps_fitting_parts(self) -> None:
        image = {
            "type": "input_image",
            "image_url": "data:image/png;base64,QQ==",
            "detail": "low",
        }
        first = {
            "type": "output_text",
            "text": "PART_A",
            "annotations": [{"type": "file_citation"}],
        }
        second_text = (
            "PART_B_START " + ("b" * 1800) + "PART_B_MIDDLE"
            + ("b" * 1800) + " PART_B_END"
        )
        message = {
            "type": "message",
            "role": "assistant",
            "content": [first, image, {"type": "output_text", "text": second_text}],
        }
        self._user("older-anchor")
        chat_store.append_item(self.session_id, message, source="assistant")
        later = {"role": "user", "content": "later-user"}
        self._user("later-user")
        prefix = {
            "type": "message",
            "role": "assistant",
            "content": [first, image],
        }
        budget = (
            context.estimate_tokens(later)
            + context.estimate_tokens(prefix)
            + 100
        )
        self.assertLess(
            budget,
            context.estimate_tokens(later) + context.estimate_tokens(message),
        )
        self.assertTrue(await self._compact(budget))
        active = context.build_active_context(self.session_id)
        tail = self._tail(active)
        bounded = next(item for item in tail if item.get("type") == "message")
        parts = bounded["content"]
        self.assertEqual(
            [part.get("type") for part in parts],
            ["output_text", "input_image", "output_text"],
        )
        self.assertEqual(parts[0]["text"], "PART_A")
        self.assertEqual(parts[0]["annotations"], [{"type": "file_citation"}])
        self.assertEqual(parts[1]["detail"], "low")
        self.assertEqual(parts[1]["image_url"], image["image_url"])
        self.assertIn("PART_B_START", parts[2]["text"])
        self.assertIn("PART_B_END", parts[2]["text"])
        self.assertIn("tokens truncated", parts[2]["text"])
        self.assertNotIn("PART_B_MIDDLE", parts[2]["text"])
        self.assertLessEqual(
            sum(context.estimate_tokens(item) for item in tail),
            budget,
        )
        archived = [
            event["item"] for event in chat_store.item_events(self.session_id)
        ]
        self.assertIn(second_text, json.dumps(archived))
        self.assertNotIn("tokens truncated", json.dumps(archived))


if __name__ == "__main__":
    unittest.main()
