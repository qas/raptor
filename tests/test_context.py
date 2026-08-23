"""Active context and checkpoint compaction tests."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

import chat_store
import context
from responses import ContextLengthError, TransientResponsesError


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
        self.session_id = chat_store.create_session(kind="main")

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
        from responses import parse_context_length_error
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
        from responses import parse_context_length_error
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
        from responses import parse_context_length_error
        response = self._response(
            500,
            {"error": {"message": "internal server error"}},
        )
        self.assertIsNone(parse_context_length_error(response))

    def test_accepts_413(self) -> None:
        from responses import parse_context_length_error
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
        self.session_id = chat_store.create_session(kind="main")

    async def test_real_estimator_compacts_huge_archive_under_budget(
        self,
    ) -> None:
        """Regression: rebuilt request must actually shrink under budget.

        Guards the 314k→314k failure where compaction ran but the next
        request still used an oversized active context.
        """
        from responses import estimate_response_request_tokens
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
        from responses import estimate_response_request_tokens
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


if __name__ == "__main__":
    unittest.main()
