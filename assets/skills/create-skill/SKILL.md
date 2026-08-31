---
name: create-skill
description: >-
  Create or revise reusable Raptor skills. Use when authoring a new skill,
  changing an existing SKILL.md, or designing a reusable agent workflow.
---

# Create a Raptor skill

Create focused workspace skills that add useful, non-obvious instructions for
repeated tasks.

## Location and structure

Store each skill at `.raptor/skills/<skill-name>/SKILL.md`:

```text
.raptor/skills/
  skill-name/
    SKILL.md
```

A skill may also contain `scripts/`, `references/`, or `assets/` when those
resources materially improve the workflow. Do not create placeholder files or
extra documentation without a concrete use.

Every `SKILL.md` starts with YAML frontmatter:

```markdown
---
name: skill-name
description: What the skill does and when it should be used.
---
```

Names use lowercase letters, digits, and hyphens and must be no longer than 64
characters. Descriptions should be concise and specific enough to distinguish
the skill from nearby workflows.

## Authoring workflow

1. Infer the purpose, trigger scenarios, constraints, and expected result from
   the request and workspace context. Ask only for information that cannot be
   safely inferred and would materially change the skill.
2. Inspect an existing skill before editing it. Preserve the user's intent and
   exact requested wording, and avoid unrelated changes.
3. Assume the agent is already capable. Include only guidance that changes its
   decisions, local conventions it cannot know, and fragile steps needed for
   correctness or safety.
4. Keep the main file short. Move substantial conditional details to directly
   linked files under `references/`. Add scripts only for repeated operations
   where deterministic execution improves reliability.
5. Match specificity to risk: allow judgment for open-ended work, but make
   permissions, validation, and dangerous operations explicit.
6. Verify the frontmatter, referenced paths, and any scripts. Confirm that the
   skill is discoverable and that no unfinished placeholders remain.

Do not add compatibility aliases, alternate storage locations, or generalized
rules for isolated examples unless the user explicitly requires them.
