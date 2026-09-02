import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-skills-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)

import skills
from raptor.shell.filesystem_permissions import FileAccessPolicy
from config import TOOLS
from tools import execute_tool


class SkillsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        skills.reset_skill_cache_for_tests()

    async def asyncTearDown(self) -> None:
        skills.reset_skill_cache_for_tests()

    async def test_discovers_metadata_then_reads_full_skill_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"
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
            with patch.object(skills, "SKILLS_ROOTS", (root,)), patch.object(
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

    async def test_initializes_create_skill_once_without_replacing_edits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"

            created = skills.initialize_builtin_skills(root)
            path = root / "create-skill" / "SKILL.md"

            self.assertEqual(created, (str(path),))
            self.assertEqual(
                path.read_bytes(),
                skills.BUILTIN_CREATE_SKILL.read_bytes(),
            )
            with patch.object(skills, "SKILLS_ROOTS", (root,)):
                snapshot = await skills.refresh_skills()
            self.assertEqual(
                [item.name for item in snapshot.skills],
                ["create-skill"],
            )

            path.write_text("operator version")
            self.assertEqual(skills.initialize_builtin_skills(root), ())
            self.assertEqual(path.read_text(), "operator version")

    async def test_start_discovery_is_non_blocking_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"
            with patch.object(skills, "SKILLS_ROOTS", (root,)):
                task = skills.start_skill_discovery()
                self.assertIsInstance(task, asyncio.Task)
                await task

    async def test_discovers_both_supported_roots_and_ignores_old_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raptor_root = base / ".raptor" / "skills"
            agent_root = base / ".agent" / "skills"
            old_root = base / ".skills"
            for root, name in (
                (raptor_root, "raptor-skill"),
                (agent_root, "agent-skill"),
                (old_root, "old-skill"),
            ):
                skill_dir = root / name
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(
                    f"---\nname: {name}\ndescription: {name}\n---\n"
                )

            with patch.object(
                skills,
                "SKILLS_ROOTS",
                (raptor_root, agent_root),
            ):
                snapshot = await skills.refresh_skills()

            self.assertEqual(
                [item.name for item in snapshot.skills],
                ["agent-skill", "raptor-skill"],
            )

    async def test_rejects_skill_symlink_that_escapes_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / ".raptor" / "skills"
            root.mkdir(parents=True)
            outside = base / "outside.md"
            outside.write_text(
                "---\nname: escaped\ndescription: bad\n---\n"
            )
            link_dir = root / "escaped"
            link_dir.mkdir()
            (link_dir / "SKILL.md").symlink_to(outside)
            with patch.object(skills, "SKILLS_ROOTS", (root,)):
                snapshot = await skills.refresh_skills()
            self.assertEqual(snapshot.skills, ())
            self.assertEqual(len(snapshot.errors), 1)
            self.assertIn("escapes configured root", snapshot.errors[0])

    async def test_unknown_name_refreshes_and_lists_available_skills(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"
            skill_dir = root / "known"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: known\ndescription: Known workflow\n---\n"
            )
            with patch.object(skills, "SKILLS_ROOTS", (root,)):
                result = await skills.read_skill_tool({"name": "missing"})
            self.assertFalse(result["ok"])
            self.assertEqual(result["available"], ["known"])

    async def test_denied_skill_is_not_advertised_or_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / ".raptor" / "skills"
            for name in ("private", "public"):
                skill_dir = root / name
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(
                    f"---\nname: {name}\ndescription: {name} workflow\n"
                    "---\nSECRET BODY\n"
                )
            policy = FileAccessPolicy.create(
                base, [".raptor/skills/private"]
            )
            with (
                patch.object(skills, "SKILLS_ROOTS", (root,)),
                patch.object(skills, "AGENT_WORKDIR", base),
                patch.object(skills, "FILESYSTEM_POLICY", policy),
            ):
                snapshot = await skills.refresh_skills()
                catalog = await skills.skill_catalog_instructions()
                result = await skills.read_skill_tool({"name": "private"})

            self.assertEqual(
                [item.name for item in snapshot.skills], ["public"]
            )
            self.assertNotIn("private workflow", catalog)
            self.assertFalse(result["ok"])
            self.assertNotIn("SECRET BODY", str(result))

    async def test_read_skill_is_exposed_through_agent_tools(self) -> None:
        schema = next(tool for tool in TOOLS if tool.get("name") == "read_skill")
        self.assertEqual(schema["parameters"]["required"], ["name"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"
            skill_dir = root / "known"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: known\ndescription: Known workflow\n---\nbody\n"
            )
            with patch.object(skills, "SKILLS_ROOTS", (root,)):
                result = await execute_tool(
                    {
                        "name": "read_skill",
                        "arguments": '{"name":"known"}',
                    }
                )
            self.assertTrue(result["ok"])
            self.assertIn("body", result["contents"])

    async def test_discovery_reads_frontmatter_without_loading_large_body(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"
            skill_dir = root / "large"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: large\ndescription: Large workflow\n---\n"
                + "x" * 20_000
            )
            with patch.object(skills, "SKILLS_ROOTS", (root,)):
                snapshot = await skills.refresh_skills()

            self.assertEqual([item.name for item in snapshot.skills], ["large"])

    async def test_read_skill_rejects_content_above_tool_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"
            skill_dir = root / "large"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: large\ndescription: Large workflow\n---\nbody"
            )
            with (
                patch.object(skills, "SKILLS_ROOTS", (root,)),
                patch.object(skills, "MAX_TOOL_OUTPUT", 16),
            ):
                result = await skills.read_skill_tool({"name": "large"})

            self.assertFalse(result["ok"])
            self.assertIn("tool-output limit", result["error"])

    async def test_discovery_rejects_oversized_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".raptor" / "skills"
            skill_dir = root / "large"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\ndescription: " + "x" * 20_000
            )
            with patch.object(skills, "SKILLS_ROOTS", (root,)):
                snapshot = await skills.refresh_skills()

            self.assertEqual(snapshot.skills, ())
            self.assertIn("frontmatter exceeds", snapshot.errors[0])


if __name__ == "__main__":
    unittest.main()
