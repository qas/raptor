import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import skills
from config import TOOLS
from tools import execute_tool


class SkillsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        skills.reset_skill_cache_for_tests()

    async def asyncTearDown(self) -> None:
        skills.reset_skill_cache_for_tests()

    async def test_discovers_metadata_then_reads_full_skill_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".skills"
            skill_dir = root / "audit"
            skill_dir.mkdir(parents=True)
            body = (
                "---\n"
                "name: audit\n"
                "description: Review code carefully.\n"
                "---\n\n"
                "FULL SECRET INSTRUCTIONS\n"
            )
            (skill_dir / "SKILL.md").write_text(body)
            with patch.object(skills, "SKILLS_ROOT", root), patch.object(
                skills, "AGENT_WORKDIR", Path(directory)
            ):
                snapshot = await skills.refresh_skills()
                self.assertEqual([item.name for item in snapshot.skills], ["audit"])

                catalog = await skills.skill_catalog_instructions()
                self.assertIn("Review code carefully.", catalog)
                self.assertNotIn("FULL SECRET INSTRUCTIONS", catalog)

                loaded = await skills.read_skill_tool({"name": "AUDIT"})
                self.assertTrue(loaded["ok"])
                self.assertEqual(loaded["contents"], body)

    async def test_start_discovery_is_non_blocking_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(skills, "SKILLS_ROOT", Path(directory) / ".skills"):
                task = skills.start_skill_discovery()
                self.assertIsInstance(task, asyncio.Task)
                await task

    async def test_rejects_skill_symlink_that_escapes_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / ".skills"
            root.mkdir()
            outside = base / "outside.md"
            outside.write_text(
                "---\nname: escaped\ndescription: bad\n---\n"
            )
            link_dir = root / "escaped"
            link_dir.mkdir()
            (link_dir / "SKILL.md").symlink_to(outside)
            with patch.object(skills, "SKILLS_ROOT", root):
                snapshot = await skills.refresh_skills()
            self.assertEqual(snapshot.skills, ())
            self.assertEqual(len(snapshot.errors), 1)
            self.assertIn("escapes .skills", snapshot.errors[0])

    async def test_unknown_name_refreshes_and_lists_available_skills(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".skills"
            skill_dir = root / "known"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: known\ndescription: Known workflow\n---\n"
            )
            with patch.object(skills, "SKILLS_ROOT", root):
                result = await skills.read_skill_tool({"name": "missing"})
            self.assertFalse(result["ok"])
            self.assertEqual(result["available"], ["known"])

    async def test_read_skill_is_exposed_through_agent_tools(self) -> None:
        schema = next(tool for tool in TOOLS if tool.get("name") == "read_skill")
        self.assertEqual(schema["parameters"]["required"], ["name"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".skills"
            skill_dir = root / "known"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: known\ndescription: Known workflow\n---\nbody\n"
            )
            with patch.object(skills, "SKILLS_ROOT", root):
                result = await execute_tool(
                    {
                        "name": "read_skill",
                        "arguments": '{"name":"known"}',
                    }
                )
            self.assertTrue(result["ok"])
            self.assertIn("body", result["contents"])


if __name__ == "__main__":
    unittest.main()
