# Contributing to Raptor

Thank you for helping improve Raptor. This project favors small, complete
changes, explicit contracts, and evidence over cleverness. Contributions must
leave the runtime easier to understand and at least as reliable as before.

## Before you start

For substantial behavior, architecture, or interface changes, open a design
discussion before implementation. State the user problem, relevant invariants,
alternatives considered, and the narrowest complete solution.

Keep each change focused on one concern. Do not combine unrelated cleanup,
formatting, or renaming with functional work.

## Engineering principles

Contributions must preserve these invariants:

- One root controller owns root-turn scheduling.
- The append-only JSONL transcript remains the canonical conversation record.
- Compaction may retire model context but never rewrite transcript history.
- Replies retain the provider and conversation that originated their input.
- Retained queues, records, output, retries, and recovery state are bounded.
- Process ownership is established before mutable runtime state is loaded.
- Main-agent and subagent backend settings remain independent.
- Provider-specific behavior stays behind the chat-provider contract.

Prefer the simplest design that fully enforces the required behavior. Add an
abstraction only when it establishes a real ownership or dependency boundary.
Avoid speculative extension points, duplicated policy, hidden global
initialization, and function-local imports used merely to conceal dependency
cycles.

Raptor is a greenfield project. Do not add compatibility aliases, migration
shims, deprecated paths, silent coercions, or fallback behavior unless the
project explicitly adopts a compatibility requirement first. Change the
canonical interface cleanly and update every caller, test, and document in the
same contribution.

## Implementation requirements

- Use clear names that describe ownership and behavior.
- Keep domain policy independent from transport adapters.
- Make task, process, file, and network-resource ownership explicit.
- Bound every retry loop and retained in-memory collection.
- Enforce memory and queue limits at acquisition; truncating only the returned
  representation does not establish a bound.
- Preserve cancellation; never convert `CancelledError` into an ordinary
  failure.
- Use crash-safe persistence for durable control state.
- Record operational failures as structured events with useful context.
- Do not silently swallow unexpected exceptions.
- Avoid broad exception handling unless the boundary can recover, report, or
  re-raise deterministically.
- Redact credentials, authorization headers, and private model payloads from
  automatic diagnostics. Deliberate audit events may preserve an operator's
  exact action only when the payload is already bounded and the local log is
  access-restricted.
- Keep source lines at or below 88 characters unless syntax makes that less
  readable.
- Do not leave dead code, commented-out implementations, TODO markers, or
  unrelated generated artifacts.

## Tests

Every behavioral change requires focused tests at the lowest useful boundary.
Test observable behavior and invariants, not source text or implementation
spelling. Include failure, cancellation, restart, and concurrency cases when
they are part of the changed behavior.

Run the complete suite from this directory:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

A contribution is not ready while tests are flaky, skipped without a tracked
reason, dependent on execution order, or reliant on real external services.
Use minimal deterministic fakes at provider and transport boundaries.

## Documentation and configuration

Update `README.md` whenever a user-visible command, environment variable,
default, storage path, operational behavior, or extension contract changes.
Document every environment variable defined by `raptor/config.py`, including its
default and purpose. Examples must be runnable and must not contain real
credentials.

Architecture documentation must describe the implementation that ships in the
same change. Do not document proposals as current behavior.

## Review standard

Reviewers should be able to answer all of the following from the contribution:

1. What user-visible or operational problem does this solve?
2. Which invariant owns the behavior?
3. Are state transitions and failure modes explicit?
4. Are concurrency and cancellation safe?
5. Are persistence and retained memory bounded?
6. Do tests prove the behavior rather than mirror the implementation?
7. Are configuration and operator documentation complete?
8. Did the change remove all superseded code and naming?

Address review comments by fixing the underlying design or explaining the
tradeoff with concrete evidence. Avoid patching symptoms with special cases.

## Submitting a change

Provide a concise description of the problem and solution, list the tests you
ran, and call out any storage, configuration, protocol, security, or operational
impact. Keep the change reviewable as one complete vertical slice.

Use Conventional Commits for commit messages, without scopes:

```text
feat: add background shell sessions
fix: preserve cancellation during retries
docs: clarify provider configuration
```

Choose the type that describes the user-visible purpose of the change. Keep
commits focused, independently understandable, and free of unrelated cleanup.

By contributing, you agree that your contribution is licensed under the MIT
License included with this project.
