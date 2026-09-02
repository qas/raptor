import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-chat-store-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from raptor.state import chat_store

TEST_MODEL_TARGET = {"provider_id": "local", "model": "test-model"}


class ChatStoreTests(unittest.TestCase):
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

    def test_session_creation_creates_jsonl(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        self.assertTrue(path.is_file())
        events = chat_store.read_events(sid)
        self.assertEqual(events[0]["type"], "session_start")
        self.assertEqual(events[0]["kind"], "main")
        self.assertEqual(events[0]["seq"], 1)

    def test_transcript_and_runtime_directories_are_private(self) -> None:
        home = Path(tempfile.mkdtemp(prefix="private-home-"))
        chats = home / "chats"
        home.chmod(0o777)
        old_umask = os.umask(0)
        try:
            with (
                patch.object(chat_store, "RAPTOR_HOME", home),
                patch.object(chat_store, "CHAT_DIR", chats),
            ):
                sid = chat_store.create_session(model_target=TEST_MODEL_TARGET,
                    kind="main",
                    chat_key="local",
                )
                transcript = chat_store.chat_path(sid)
                self.assertEqual(home.stat().st_mode & 0o777, 0o700)
                self.assertEqual(chats.stat().st_mode & 0o777, 0o700)
                self.assertEqual(
                    transcript.stat().st_mode & 0o777,
                    0o600,
                )
        finally:
            os.umask(old_umask)

    def test_boot_repair_tightens_existing_transcript_mode(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        self._chat_dir.chmod(0o755)
        path.chmod(0o644)

        chat_store.repair_all_chat_files()

        self.assertEqual(self._chat_dir.stat().st_mode & 0o777, 0o700)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_context_reset_preserves_archive_and_discards_old_checkpoint(
        self,
    ) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        earlier = chat_store.append_item(
            sid,
            {"role": "user", "content": "earlier"},
            source="user",
        )
        chat_store.append_checkpoint(
            sid,
            summary="safe context",
            through_seq=int(earlier["seq"]),
        )
        failed = chat_store.append_item(
            sid,
            {"role": "user", "content": "rejected task"},
            source="user",
        )
        chat_store.append_checkpoint(
            sid,
            summary="contaminated context",
            through_seq=int(failed["seq"]),
        )
        outcome = chat_store.append_item(
            sid,
            {"role": "assistant", "content": "failed"},
            source="assistant",
        )

        chat_store.reset_model_context(
            sid,
            through_seq=int(outcome["seq"]),
        )

        self.assertIsNone(chat_store.active_checkpoint(sid))
        self.assertEqual(chat_store.active_item_events(sid), [])
        self.assertEqual(len(chat_store.item_events(sid)), 3)

    def test_sequence_numbers_monotonic(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        a = chat_store.append_item(
            sid,
            {"role": "user", "content": "one"},
            source="user",
        )
        b = chat_store.append_item(
            sid,
            {"role": "user", "content": "two"},
            source="user",
        )
        self.assertEqual(a["seq"], 2)
        self.assertEqual(b["seq"], 3)

    def test_append_never_rewrites_previous_lines(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        before = path.read_text(encoding="utf-8")
        chat_store.append_item(
            sid,
            {"role": "user", "content": "x"},
            source="user",
        )
        after = path.read_text(encoding="utf-8")
        self.assertTrue(after.startswith(before))
        self.assertGreater(len(after), len(before))

    def test_raw_responses_items_round_trip(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        item = {
            "type": "function_call",
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"ls"}',
        }
        written = chat_store.append_item(
            sid,
            item,
            source="assistant",
        )
        loaded = chat_store.item_events(sid)[-1]
        self.assertEqual(loaded["item"], item)
        self.assertEqual(written["item"], item)

    def test_checkpoint_and_session_end_persist(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        chat_store.append_item(
            sid,
            {"role": "user", "content": "hi"},
            source="user",
        )
        cp = chat_store.append_checkpoint(
            sid,
            summary="done so far",
            through_seq=2,
            input_from_seq=2,
            input_to_seq=2,
            reason="manual",
            anchors=[
                {
                    "seq": 2,
                    "item": {"role": "user", "content": "hi"},
                }
            ],
        )
        end = chat_store.end_session(
            sid,
            reason="new_session",
            todos=[{"id": 1, "text": "t"}],
        )
        self.assertEqual(cp["type"], "checkpoint")
        self.assertEqual(end["type"], "session_end")
        self.assertEqual(
            chat_store.latest_checkpoint(sid)["summary"],
            "done so far",
        )
        self.assertEqual(
            chat_store.latest_checkpoint(sid)["anchors"][0]["item"]["content"],
            "hi",
        )

    def test_main_and_subagent_sessions_isolated(self) -> None:
        main = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        child = chat_store.create_session(model_target=TEST_MODEL_TARGET,
            kind="subagent",
            chat_key="local",
            agent_id="abcd1234",
            parent_session_id=main,
        )
        chat_store.append_item(
            main,
            {"role": "user", "content": "parent"},
            source="user",
        )
        chat_store.append_item(
            child,
            {"role": "user", "content": "child"},
            source="delegation",
        )
        self.assertEqual(
            [e["item"]["content"] for e in chat_store.item_events(main)],
            ["parent"],
        )
        self.assertEqual(
            [e["item"]["content"] for e in chat_store.item_events(child)],
            ["child"],
        )
        start = chat_store.read_events(child)[0]
        self.assertEqual(start["parent_session_id"], main)
        self.assertEqual(start["agent_id"], "abcd1234")

    def test_session_listing_and_search_are_transcript_scoped(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        chat_store.append_item(
            sid,
            {"role": "user", "content": "Project Firefly"},
            source="user",
        )
        (self._chat_dir / "not-a-session.jsonl").write_text("{}\n")

        sessions = chat_store.list_sessions()

        self.assertEqual([row["session_id"] for row in sessions], [sid])
        self.assertTrue(chat_store.session_contains_text(sid, "firefly"))
        self.assertFalse(chat_store.session_contains_text(sid, "saturn"))

    def test_session_listing_summarizes_only_requested_newest(self) -> None:
        for _ in range(5):
            chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        original = chat_store._session_summary
        seen: list[Path] = []

        def summarize(path: Path):
            seen.append(path)
            return original(path)

        with patch.object(chat_store, "_session_summary", summarize):
            sessions = chat_store.list_sessions(limit=2)

        self.assertEqual(len(sessions), 2)
        self.assertEqual(len(seen), 2)

    def test_filtered_listing_is_not_starved_by_noisy_chats(self) -> None:
        session_ids = iter(
            f"20260824-000000-{index:08x}"
            for index in range(102)
        )
        with patch.object(
            chat_store,
            "new_session_id",
            side_effect=lambda: next(session_ids),
        ):
            quiet = chat_store.create_session(model_target=TEST_MODEL_TARGET,
                kind="main",
                chat_key="quiet",
            )
            for _ in range(101):
                chat_store.create_session(model_target=TEST_MODEL_TARGET,
                    kind="main",
                    chat_key="noisy",
                )

        sessions = chat_store.list_sessions(
            limit=20,
            chat_key="quiet",
            kinds={"main"},
        )

        self.assertEqual(
            [row["session_id"] for row in sessions],
            [quiet],
        )

    def test_exact_session_summary_does_not_discover_archive(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        with patch.object(
            Path,
            "glob",
            side_effect=AssertionError("archive discovery is not allowed"),
        ):
            summary = chat_store.session_summary(sid)

        self.assertEqual(summary["session_id"], sid)

    def test_invalid_session_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chat_store.chat_path("../escape")
        with self.assertRaises(ValueError):
            chat_store.append_item(
                "not-a-session",
                {"role": "user", "content": "x"},
                source="user",
            )

    def test_partial_final_line_repaired_before_append(
        self,
    ) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        path.write_bytes(
            path.read_bytes() + b'{"v":1,"seq":99,"partial"'
        )
        self.assertTrue(chat_store.repair_chat_file(sid))
        chat_store.append_item(
            sid,
            {"role": "user", "content": "after-repair"},
            source="user",
        )
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for line in lines:
            json.loads(line)
        self.assertIn(
            "after-repair",
            lines[-1],
        )

    def test_large_partial_tail_is_repaired_without_full_file_read(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        valid_size = path.stat().st_size
        with path.open("ab") as handle:
            handle.write(b"x" * (2 * 1024 * 1024))

        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("full-file read"),
        ):
            self.assertTrue(chat_store.repair_chat_file(sid))

        self.assertEqual(path.stat().st_size, valid_size)

    def test_active_projection_does_not_materialize_archive(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        retired = chat_store.append_item(
            sid,
            {"role": "user", "content": "retired"},
            source="user",
        )
        chat_store.append_checkpoint(
            sid,
            summary="old",
            through_seq=int(retired["seq"]),
        )
        chat_store.reset_model_context(
            sid,
            through_seq=int(retired["seq"]),
        )
        active = chat_store.append_item(
            sid,
            {"role": "user", "content": "active"},
            source="user",
        )
        chat_store.append_checkpoint(
            sid,
            summary="current",
            through_seq=int(active["seq"]),
        )
        chat_store.append_item(
            sid,
            {"role": "user", "content": "tail"},
            source="user",
        )

        with patch.object(
            chat_store,
            "read_events",
            side_effect=AssertionError("archive materialized"),
        ):
            projection = chat_store.active_projection(sid)

        self.assertEqual(
            [event["item"]["content"] for event in projection.items],
            ["tail"],
        )
        self.assertEqual(projection.checkpoint["summary"], "current")
        self.assertEqual(projection.archive_events, 7)

    def test_ending_session_releases_sequence_cache_entry(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        self.assertIn(sid, chat_store._SEQ_CACHE)

        chat_store.end_session(sid, reason="archived")

        self.assertNotIn(sid, chat_store._SEQ_CACHE)

    def test_append_repairs_crash_tail_automatically(
        self,
    ) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        good = path.read_bytes()
        path.write_bytes(good + b'{"broken":')
        chat_store.append_item(
            sid,
            {"role": "user", "content": "ok"},
            source="user",
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)

    def test_owner_lookup_does_not_parse_a_partial_tail(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="owner")
        with chat_store.chat_path(sid).open("ab") as handle:
            handle.write(b'{"type":"item"')

        self.assertEqual(chat_store.session_chat_key(sid), "owner")

    def test_complete_corrupt_event_is_reported(self) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("not-json\n")

        with self.assertRaisesRegex(
            RuntimeError,
            rf"{re.escape(str(path))}:2",
        ):
            chat_store.read_events(sid)

    def test_healthy_append_does_not_reread_whole_file(
        self,
    ) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        path = chat_store.chat_path(sid)
        reads = {"n": 0}
        real_read = Path.read_bytes

        def counted_read(self):
            reads["n"] += 1
            return real_read(self)

        with patch.object(Path, "read_bytes", counted_read):
            chat_store.append_item(
                sid,
                {"role": "user", "content": "cheap"},
                source="user",
            )
        self.assertEqual(reads["n"], 0)
        self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_compaction_render_truncates_oversized_record(
        self,
    ) -> None:
        huge = "H" * 50_000
        records = [
            {
                "type": "item",
                "seq": 2,
                "item": {
                    "role": "user",
                    "content": huge,
                },
            }
        ]
        rendered = chat_store.render_compaction_records(
            records,
            max_record_chars=2_000,
        )
        self.assertLessEqual(len(rendered), 2_000)
        self.assertIn(
            "truncated for checkpoint generation",
            rendered,
        )
        self.assertIn("USER", rendered)
        self.assertNotIn(huge, rendered)

    def test_compaction_render_leaves_archive_untouched(
        self,
    ) -> None:
        sid = chat_store.create_session(model_target=TEST_MODEL_TARGET, kind="main", chat_key="local")
        huge = "CANONICAL-BLOB-" + ("Z" * 40_000)
        event = chat_store.append_item(
            sid,
            {"role": "user", "content": huge},
            source="user",
        )
        before = chat_store.chat_path(sid).read_text(
            encoding="utf-8"
        )
        rendered = chat_store.render_compaction_records(
            [event],
            max_record_chars=1_500,
        )
        after = chat_store.chat_path(sid).read_text(
            encoding="utf-8"
        )
        self.assertEqual(before, after)
        stored = chat_store.item_events(sid)[-1]["item"]["content"]
        self.assertEqual(stored, huge)
        self.assertLessEqual(len(rendered), 1_500)
        self.assertIn(
            "truncated for checkpoint generation",
            rendered,
        )

    def test_fork_copies_items_before_last_turn_with_tool_items(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        for item, item_source in (
            ({"role": "user", "content": "one"}, "user"),
            ({"type": "function_call", "call_id": "c1"}, "assistant"),
            ({"type": "function_call_output", "call_id": "c1"}, "tool"),
            ({"role": "assistant", "content": "done"}, "assistant"),
            ({"role": "user", "content": "two"}, "user"),
        ):
            chat_store.append_item(source, item, source=item_source)

        forked, copied = chat_store.fork_session_before_last_user_turns(
            source,
            turns=1,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )

        self.assertEqual(copied, 4)
        self.assertEqual(
            [event["item"] for event in chat_store.item_events(forked)],
            [event["item"] for event in chat_store.item_events(source)[:4]],
        )
        self.assertEqual(chat_store.session_chat_key(forked), "local")
        self.assertEqual(
            chat_store.session_summary(forked)["model_target"],
            TEST_MODEL_TARGET,
        )

    def test_fork_remaps_checkpoint_before_active_cutoff(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        first = chat_store.append_item(
            source,
            {"role": "user", "content": "first"},
            source="user",
        )
        chat_store.append_item(
            source,
            {"role": "assistant", "content": "answer"},
            source="assistant",
        )
        chat_store.append_checkpoint(
            source,
            summary="stale summary",
            through_seq=int(first["seq"]) + 1,
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "last"},
            source="user",
        )
        forked, _ = chat_store.fork_session_before_last_user_turns(
            source,
            turns=1,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )

        self.assertEqual(
            chat_store.latest_checkpoint(forked)["summary"],
            "stale summary",
        )
        self.assertEqual(chat_store.latest_checkpoint(forked)["through_seq"], 1)
        self.assertEqual(len(chat_store.item_events(forked)), 0)
        self.assertEqual(chat_store.read_events(forked)[-1]["type"], "checkpoint")

    def test_fork_keeps_late_checkpoint_that_precedes_active_cutoff(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        retired = chat_store.append_item(
            source,
            {"role": "user", "content": "retired"},
            source="user",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "retained active turn"},
            source="user",
        )
        chat_store.append_checkpoint(
            source,
            summary="retired prefix only",
            through_seq=int(retired["seq"]),
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "final turn"},
            source="user",
        )

        forked, _ = chat_store.fork_session_before_last_user_turns(
            source,
            turns=2,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )

        self.assertEqual(chat_store.item_events(forked), [])
        self.assertEqual(
            chat_store.active_projection(forked).checkpoint["summary"],
            "retired prefix only",
        )

    def test_fork_keeps_context_reset_boundary_effective(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        retired = chat_store.append_item(
            source,
            {"role": "user", "content": "retired"},
            source="user",
        )
        chat_store.reset_model_context(
            source,
            through_seq=int(retired["seq"]),
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "current"},
            source="user",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "last"},
            source="user",
        )

        forked, _ = chat_store.fork_session_before_last_user_turns(
            source,
            turns=1,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )

        self.assertEqual(
            [
                event["item"]["content"]
                for event in chat_store.item_events(forked)
            ],
            ["current"],
        )
        self.assertEqual(
            chat_store.active_item_events(forked)[0]["item"]["content"],
            "current",
        )

    def test_repeated_fork_cannot_cross_checkpoint_boundary(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        retired = chat_store.append_item(
            source,
            {"role": "user", "content": "retired"},
            source="user",
        )
        chat_store.append_checkpoint(
            source,
            summary="bounded summary",
            through_seq=int(retired["seq"]),
        )
        for text, role in (
            ("active one", "user"),
            ("answer", "assistant"),
            ("active two", "user"),
        ):
            chat_store.append_item(
                source,
                {"role": role, "content": text},
                source=role,
            )

        first, _ = chat_store.fork_session_before_last_user_turns(
            source,
            turns=1,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )
        second, _ = chat_store.fork_session_before_last_user_turns(
            first,
            turns=1,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )

        self.assertEqual(chat_store.item_events(second), [])
        self.assertEqual(
            chat_store.active_projection(second).checkpoint["summary"],
            "bounded summary",
        )
        sessions_before = {row["session_id"] for row in chat_store.list_sessions()}
        with self.assertRaisesRegex(ValueError, "0 active user turn"):
            chat_store.fork_session_before_last_user_turns(
                second,
                turns=1,
                chat_key="local",
                model_target=TEST_MODEL_TARGET,
            )
        self.assertEqual(
            sessions_before,
            {row["session_id"] for row in chat_store.list_sessions()},
        )

    def test_fork_rejects_too_many_turns_without_mutation_or_destination(
        self,
    ) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "only"},
            source="user",
        )
        before = chat_store.chat_path(source).read_text(encoding="utf-8")
        sessions_before = {row["session_id"] for row in chat_store.list_sessions()}

        with self.assertRaises(ValueError):
            chat_store.fork_session_before_last_user_turns(
                source,
                turns=2,
                chat_key="local",
                model_target=TEST_MODEL_TARGET,
            )

        self.assertEqual(
            chat_store.chat_path(source).read_text(encoding="utf-8"),
            before,
        )
        sessions_after = {row["session_id"] for row in chat_store.list_sessions()}
        self.assertEqual(sessions_after, sessions_before)

    def test_fork_counts_thread_merge_users_but_not_steer_or_runtime(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "base"},
            source="user",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "steering"},
            source="steer",
            data={"steer_id": "s1"},
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "runtime"},
            source="runtime",
        )
        chat_store.append_event(
            source,
            {
                "type": "item",
                "source": "thread_merge",
                "origin": {"source": "user"},
                "item": {"role": "user", "content": "merged"},
            },
        )
        chat_store.append_item(
            source,
            {"role": "assistant", "content": "answer"},
            source="assistant",
        )

        forked, copied = chat_store.fork_session_before_last_user_turns(
            source,
            turns=1,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )

        self.assertEqual(copied, 3)
        self.assertEqual(
            [event["item"]["content"] for event in chat_store.item_events(forked)],
            ["base", "steering", "runtime"],
        )

    def test_fork_preserves_and_copies_item_metadata(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        item = {"role": "assistant", "content": [{"text": "old"}]}
        metadata = {"nested": ["value"]}
        event = chat_store.append_event(
            source,
            {
                "type": "item",
                "source": "thread_merge",
                "origin": {"session_id": "branch", "seq": 8},
                "data": metadata,
                "item": item,
            },
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "last"},
            source="user",
        )

        forked, _ = chat_store.fork_session_before_last_user_turns(
            source,
            turns=1,
            chat_key="local",
            model_target=TEST_MODEL_TARGET,
        )

        copied = chat_store.item_events(forked)[0]
        self.assertEqual(copied["data"], event["data"])
        self.assertEqual(copied["origin"], event["origin"])
        self.assertIsNot(copied["data"], metadata)
        self.assertIsNot(copied["item"], item)
        metadata["nested"].append("mutated")
        item["content"][0]["text"] = "mutated"
        self.assertEqual(copied["data"], {"nested": ["value"]})
        self.assertEqual(copied["item"]["content"][0]["text"], "old")

    def test_fork_write_failure_archives_destination_and_keeps_source(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "first"},
            source="user",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "last"},
            source="user",
        )
        before = chat_store.chat_path(source).read_text(encoding="utf-8")
        original_append = chat_store.append_event
        calls = {"count": 0}

        def fail_copy(session_id, event, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError("simulated copy failure")
            return original_append(session_id, event, **kwargs)

        with patch.object(chat_store, "append_event", fail_copy):
            with self.assertRaises(OSError):
                chat_store.fork_session_before_last_user_turns(
                    source,
                    turns=1,
                    chat_key="local",
                    model_target=TEST_MODEL_TARGET,
                )

        self.assertEqual(
            chat_store.chat_path(source).read_text(encoding="utf-8"),
            before,
        )
        archived = [
            row
            for row in chat_store.list_sessions()
            if row["session_id"] != source
        ]
        self.assertEqual(len(archived), 1)
        self.assertTrue(chat_store.session_is_ended(archived[0]["session_id"]))

    def test_fork_rejects_invalid_source_and_turn_values_before_creation(self) -> None:
        source = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        chat_store.append_item(
            source,
            {"role": "user", "content": "request"},
            source="user",
        )
        cases = (
            (True, "local", TEST_MODEL_TARGET),
            ("1", "local", TEST_MODEL_TARGET),
            (1, "other", TEST_MODEL_TARGET),
            (1, "local", {"provider_id": "other", "model": "m"}),
        )
        for turns, owner, target in cases:
            with self.subTest(turns=turns, owner=owner, target=target):
                before = {row["session_id"] for row in chat_store.list_sessions()}
                with self.assertRaises(ValueError):
                    chat_store.fork_session_before_last_user_turns(
                        source,
                        turns=turns,
                        chat_key=owner,
                        model_target=target,
                    )
                self.assertEqual(
                    before,
                    {row["session_id"] for row in chat_store.list_sessions()},
                )

    def test_fork_rejects_ended_source_and_no_user_history(self) -> None:
        ended = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        chat_store.append_item(
            ended,
            {"role": "assistant", "content": "orphan"},
            source="assistant",
        )
        chat_store.end_session(ended, reason="test")
        empty = chat_store.create_session(
            model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key="local",
        )
        for session_id in (ended, empty):
            with self.subTest(session_id=session_id):
                with self.assertRaises(ValueError):
                    chat_store.fork_session_before_last_user_turns(
                        session_id,
                        turns=1,
                        chat_key="local",
                        model_target=TEST_MODEL_TARGET,
                    )


if __name__ == "__main__":
    unittest.main()
