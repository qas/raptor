import asyncio
import json
import unittest
from unittest.mock import patch

from config import MAX_TOOL_OUTPUT
from engine import function_call_output, run_agent


class AgentEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_tool_arguments_report_terminal_result(self) -> None:
        responses = [
            {
                "output": [{
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call-1",
                    "arguments": "{",
                }]
            },
            {
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }]
            },
        ]
        executed = False
        reported: list[tuple[dict, dict]] = []

        async def create_response(_work):
            return responses.pop(0)

        async def execute_call(_call):
            nonlocal executed
            executed = True
            return {"ok": True}

        async def report_tool_result(call, result):
            reported.append((call, result))

        await run_agent(
            work=[],
            create_response=create_response,
            execute_call=execute_call,
            source="test",
            max_tool_rounds=1,
            report_tool_result=report_tool_result,
        )

        self.assertFalse(executed)
        self.assertEqual(reported[0][0]["call_id"], "call-1")
        self.assertFalse(reported[0][1]["ok"])
        self.assertIn("bad JSON arguments", reported[0][1]["error"])

    async def test_terminal_output_uses_distinct_recording_boundary(
        self,
    ) -> None:
        output = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        }
        ordinary: list[dict] = []
        terminal: list[tuple[list[dict], str]] = []

        async def create_response(_work):
            return {"output": [output]}

        result = await run_agent(
            work=[],
            create_response=create_response,
            execute_call=lambda _call: None,
            source="test",
            max_tool_rounds=1,
            record_items=lambda items, _source: ordinary.extend(items),
            record_terminal_items=lambda items, text: terminal.append(
                (items, text)
            ),
        )

        self.assertEqual(result["text"], "done")
        self.assertEqual(ordinary, [])
        self.assertEqual(terminal, [([output], "done")])

    def test_function_output_bounds_oversized_result_as_valid_json(self) -> None:
        item = function_call_output(
            {"call_id": "call-1"},
            {"ok": True, "content": "x" * (MAX_TOOL_OUTPUT * 2)},
        )

        self.assertLessEqual(len(item["output"]), MAX_TOOL_OUTPUT)
        payload = json.loads(item["output"])
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["truncated"])
        self.assertGreater(payload["original_chars"], MAX_TOOL_OUTPUT)

    async def test_tool_result_log_excludes_result_payload(self) -> None:
        responses = [
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "read_file",
                        "call_id": "call-1",
                        "arguments": '{"path":"secret"}',
                    }
                ]
            },
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "done"}
                        ],
                    }
                ]
            },
        ]

        async def create_response(_work):
            return responses.pop(0)

        async def execute_call(_call):
            return {"ok": True, "text": "private tool payload"}

        with patch("engine.log_event") as logged:
            await run_agent(
                work=[],
                create_response=create_response,
                execute_call=execute_call,
                source="test",
                max_tool_rounds=1,
            )

        tool_log = next(
            call
            for call in logged.call_args_list
            if call.args[1] == "tool_result"
        )
        self.assertNotIn("result", tool_log.args[2])
        self.assertNotIn("private tool payload", json.dumps(tool_log.args[2]))

    async def test_interruption_records_matching_tool_output(self) -> None:
        started = asyncio.Event()
        recorded: list[dict] = []

        async def create_response(_work):
            return {
                "id": "response-1",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "name": "shell",
                        "call_id": "call-1",
                        "arguments": '{"command":"sleep 60"}',
                    }
                ],
            }

        async def execute_call(_call):
            started.set()
            await asyncio.Event().wait()

        def record_items(items, _source):
            recorded.extend(items)

        with patch("engine.log_event"):
            task = asyncio.create_task(
                run_agent(
                    work=[],
                    create_response=create_response,
                    execute_call=execute_call,
                    source="test",
                    max_tool_rounds=1,
                    record_items=record_items,
                )
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(recorded[0]["type"], "function_call")
        self.assertEqual(recorded[1]["type"], "function_call_output")
        self.assertEqual(recorded[1]["call_id"], "call-1")
        output = json.loads(recorded[1]["output"])
        self.assertEqual(output["status"], "interrupted")

    async def test_tool_exception_becomes_matching_failure_output(self) -> None:
        recorded: list[dict] = []
        responses = [
            {
                "output": [{
                    "type": "function_call",
                    "name": "read_file",
                    "call_id": "call-1",
                    "arguments": "{}",
                }]
            },
            {
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "recovered"}],
                }]
            },
        ]

        async def create_response(_work):
            return responses.pop(0)

        async def execute_call(_call):
            raise RuntimeError("approval transport failed")

        def record_items(items, _source):
            recorded.extend(items)

        result = await run_agent(
            work=[],
            create_response=create_response,
            execute_call=execute_call,
            source="test",
            max_tool_rounds=1,
            record_items=record_items,
        )

        self.assertEqual(result["text"], "recovered")
        self.assertEqual(recorded[1]["type"], "function_call_output")
        failure = json.loads(recorded[1]["output"])
        self.assertFalse(failure["ok"])
        self.assertIn("approval transport failed", failure["error"])

    async def test_malformed_tool_arguments_never_reach_execution(self) -> None:
        recorded: list[dict] = []
        responses = [
            {
                "output": [{
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call-1",
                    "arguments": '{"command":"unterminated}',
                }]
            },
            {
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "recovered"}],
                }]
            },
        ]

        async def create_response(_work):
            return responses.pop(0)

        async def execute_call(_call):
            self.fail("malformed tool arguments reached execution")

        def record_items(items, _source):
            recorded.extend(items)

        result = await run_agent(
            work=[],
            create_response=create_response,
            execute_call=execute_call,
            source="test",
            max_tool_rounds=1,
            record_items=record_items,
        )

        self.assertEqual(result["text"], "recovered")
        failure = json.loads(recorded[1]["output"])
        self.assertFalse(failure["ok"])
        self.assertIn("bad JSON arguments", failure["error"])


if __name__ == "__main__":
    unittest.main()
