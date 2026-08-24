# Raptor

![Raptor](assets/banner.png)

Raptor is a persistent, tool-using agent runtime built on the OpenAI Responses
API. It provides durable conversations, bounded context compaction, goals,
steering, approvals, managed shell sessions, skills, and isolated subagents
behind a provider-neutral chat interface.

Raptor ships with Telegram and an inbound Responses-compatible HTTP API. Both
providers can run at the same time. Every provider conversation owns an
isolated durable main-agent chat; replies and transport state never cross chat
boundaries.

## Key properties

- **Durable by default.** An append-only JSONL transcript is the source of
  truth. Checkpoint compaction changes the model window, not the archive.
- **Provider-neutral.** The core operates on normalized events and opaque
  identifiers; adapters own transport behavior.
- **Recoverable.** Interrupted work, goals, shell completions, and subagent
  completions retain enough state to continue safely.
- **Bounded.** Context, tool output, subagent records, and recovery events have
  explicit limits.
- **Observable.** Structured lifecycle events record requests, compaction,
  retries, tools, and completion delivery without logging prompt or tool-result
  bodies.
- **Extensible.** Providers and workspace-local skills use small explicit
  contracts.

## Quick start

Raptor requires Python 3.11 or newer and a Responses-compatible model backend.
The entry point declares its Python dependency for `uv`.

```bash
export RESPONSES_BASE_URL=http://127.0.0.1:8000/v1
export RESPONSES_MODEL=your-model
export CHAT_PROVIDERS=telegram,responses_api
export TG_BOT_TOKEN=your-telegram-token
export TG_USER_ID=123456789
export TG_CHAT_ID=123456789

uv run raptor.py
```

The inbound Responses API listens on `0.0.0.0:8787` by default:

```bash
curl http://127.0.0.1:8787/v1/responses \
  -H 'Authorization: Bearer strong-secret' \
  -H 'Content-Type: application/json' \
  -d '{"conversation":"project-a","input":"Hello"}'
```

Model discovery is available at `GET /v1/models`. Live status surfaces are
available at `GET /v1/status` for `default`, or at
`GET /v1/status?conversation=project-a` for a named conversation.

Process commands:

```bash
uv run raptor.py --status
uv run raptor.py --stop-daemon
uv run raptor.py --daemon
```

Raptor uses an atomic process-lifetime lock. A second instance cannot start
against the same `RAPTOR_HOME`. Status and stop commands use that lock's owner
PID, so they still work while the application is starting or when session
metadata is unavailable. Run process commands with the same `RAPTOR_HOME` as
the daemon (the launch workspace's `.raptor` directory by default).

## Commands

| Command | Behavior |
|---|---|
| `/new` | Archive the current session and create a clean one |
| `/chats [term]` | List recent chats, or find chats containing a term |
| `/resume <session-id>` | Resume a prior main chat by its full ID |
| `/ask <message>` | Run a stateless side query without changing conversation history |
| `/thread` | Fork a temporary conversation branch |
| `/thread clear` | Discard the branch and return to its parent |
| `/thread merge` | Merge branch-native conversation items into its parent |
| `/status` | Show runtime, context, goal, subagent, and shell status |
| `/stop` | Interrupt the current root turn; background work continues |
| `/stop all` | Interrupt and cancel background work in the current main chat |
| `/compact` | Create a durable context checkpoint |
| `/model` | List or switch backend models |
| `/approval` | Toggle tool approval |
| `/todos` | Show the active execution plan |
| `/goal` | Inspect or manage the persistent goal |
| `/help` | Show available commands |

`/ask` has no prior conversation, goal, or base instructions. It may use normal
tools for multiple in-memory rounds, but neither its model exchange nor answer
is added to the canonical transcript. Tool side effects remain real.

`/thread` creates a separate crash-safe transcript. Clearing a thread restores
the parent unchanged. Merging copies only post-fork user, assistant,
function-call, and function-result items. Filesystem, network, and other tool
side effects are not reversible.

## Architecture

| Path | Responsibility |
|---|---|
| `raptor.py` | Ownership-first process entry point and CLI |
| `application.py` | Long-running provider and agent application lifecycle |
| `process_lock.py` | State-independent atomic process ownership |
| `runtime.py` | Process metadata and daemon controls |
| `runtime_paths.py` | Dependency-free runtime filesystem locations |
| `agent.py` | Root turn execution and compaction integration |
| `engine.py` | Tool-round engine and response parsing |
| `controller.py` | Single-owner root scheduling and goal continuation |
| `turn_runtime.py` | Root-turn ownership, identity, and bounded interruption |
| `runtime_events.py` | Typed background-completion events delivered to the root |
| `observability.py` | Structured runtime events and activity labels |
| `responses.py` | Outbound Responses client, streaming, and retry policy |
| `response_errors.py` | Shared Responses protocol errors |
| `chat_provider.py` | Provider protocol and normalized event types |
| `chat_runtime.py` | Provider loading, binding, and deferred delivery context |
| `multi_provider.py` | Concurrent provider routing |
| `telegram.py` | Telegram adapter |
| `responses_provider.py` | Inbound Responses-compatible HTTP adapter |
| `presentation.py` | Provider-neutral status, controls, and activity policy |
| `chat_store.py` | Append-only transcript storage |
| `storage.py` | Crash-safe atomic local file replacement |
| `context.py` | Active-context construction and checkpoint compaction |
| `session.py` | Durable chat registry and context-bound per-chat runtimes |
| `thread_state.py` | Temporary-thread state queries |
| `thread_status.py` | Temporary-thread status projection |
| `subagents.py` | Isolated foreground and background subagents |
| `shell_sessions.py` | Managed shell processes, PTYs, polling, and completion |
| `skills.py` | Progressive `.skills` discovery and loading |
| `commands.py` | Provider-neutral slash commands |
| `threads.py` | Temporary branch lifecycle and merge policy |

### Main chats, scheduling, and delivery

Each provider conversation owns one main-chat runtime: transcript selection,
goal, todos, thread, approvals, steering queue, background resources, status
surface, and root controller. One root turn runs at a time within a chat;
different chats can run concurrently. Model selection and process limits remain
process-wide.

User input, steering, background shell completion, subagent completion, and
goal continuation enter the owning chat's controller instead of starting
competing runs. Completion events carry their chat owner and cannot be consumed
by another chat.

The root agent can cancel one background subagent or managed shell owned by its
chat using the returned identifier. Targeted cancellation suppresses completion
delivery; `/stop all` cancels the current chat's root turn, queued work,
subagents, and shells without disturbing other chats.

Request-only providers never discard an out-of-band completion. If no request
is open when background work finishes, delivery remains pending and is retried
when the user next addresses that conversation.

Each queued request retains its originating conversation and provider delivery
context. A Responses HTTP request that becomes steering remains open and
receives the answer produced for that specific queued input. Supply the
`conversation` field to select a named API chat; omitting it selects `default`.
Telegram replies remain in their originating topic.

The persistent status slot has this priority:

```text
approval > thread > goal > empty
```

Steering is transient and is never pinned.

### Context and compaction

Raptor maintains three distinct layers:

1. **Canonical transcript:** lossless append-only JSONL.
2. **Active context:** the latest checkpoint, preserved user anchors, and the
   native tail after the checkpoint boundary.
3. **Checkpoint:** a model-generated continuation summary that retires older
   items from the model window only.

Compaction sends a plain rendered record set rather than malformed native tool
call fragments. A completed compaction with no visible summary is retried once;
if it remains empty, the failure is classified as transient and an active goal
is paused instead of blocked. Providers may show a temporary animated
`Compacting.` / `Compacting..` / `Compacting...` indicator.

### Retries and continuation

Transient transport errors include connection failures, disconnects such as
`RemoteProtocolError`, incomplete streams, and retryable HTTP statuses (`408`,
`429`, `500`, `502`, `503`, and `504`). The initial request is followed by up
to `RESPONSES_MAX_RETRIES` exponential-backoff retries.

Retry exhaustion ends only the current attempt:

- an ordinary root turn releases the controller, so the user can message again;
- an active goal pauses and can continue with `/goal resume`;
- a subagent keeps its transcript and can be continued by agent ID.

There is no unbounded automatic retry loop.

### Shell sessions and subagents

Shell commands wait for `yield_time_ms` and then return a managed session ID if
still running. `write_stdin` polls output or writes to a PTY. Detached
completions re-enter through the root controller and retry delivery if the
controller temporarily fails. `/stop all` terminates every live process group
owned by the current main chat.

Subagents have isolated transcripts and independent backend, reasoning, retry,
and context-window configuration. Completed record retention and recovery tool
events are bounded; running, interrupted, and undelivered completion records
are protected from pruning. Their private tool history is never projected into
the parent. A subagent compacts lazily when its next model request needs room,
so finishing a child does not trigger speculative compaction or delay its
result.

The background-subagent limit is process-wide. Providers may project safe,
bounded activity without receiving the child's transcript or tool payloads. In
a Telegram forum, Raptor creates a temporary `Subagent: <id>` topic that shows
the same public reasoning summaries, assistant output, and tool activity as the
main agent. The topic is read-only because the parent owns steering; the child
transcript and tool payloads remain isolated. Raptor coalesces updates, treats
duplicates as no-ops, deletes the topic when the subagent ends, and returns the
bounded completion to the parent chat.

### Telegram forum mode

Set `TG_CHAT_ID` to a private chat for one main chat, or to a forum-enabled
supergroup for multichat. In forum mode, the General topic and every normal
forum topic are independent main-agent chats. The bot must be an administrator
with **Manage Topics** permission so it can manage temporary subagent activity
topics. Only `TG_USER_ID` is accepted as interactive input.

Create and name normal topics with Telegram's standard UI. Raptor discovers a
topic on its first message and persists its runtime. Activity topics are
presentation-only and cannot steer their subagent; steer from the parent topic.

## Storage

All durable runtime data lives under `RAPTOR_HOME`:

```text
$AGENT_WORKDIR/.raptor/
  runtime.lock
  state.json
  raptor.log                 # daemon mode
  chats/
    <session-id>.jsonl
```

`/new` creates a new transcript without deleting the old one. Archived sessions
remain searchable with the `chat_history` tool, but only from their owning main
chat. Transcript `session_start` records carry the main-chat key, and recovery
rejects cross-chat references. State schema mismatches fail explicitly instead
of guessing or silently discarding durable state.

## Skills

Raptor discovers skills recursively under `$AGENT_WORKDIR/.skills` using this
layout:

```text
.skills/
  skill-name/
    SKILL.md
```

Discovery loads only frontmatter metadata into normal instructions. The full
`SKILL.md` is loaded when a user names the skill or the task matches its
description. Referenced resources remain unloaded until the skill needs them.
Root agents and subagents share the catalog.

## Add a chat provider

Set `CHAT_PROVIDERS` to a built-in name or `module:attribute`. The attribute may
be a provider instance, a zero-argument class, or a zero-argument factory. The
result must satisfy `ChatProvider`.

```python
from chat_provider import PollResult, ProviderCapabilities


class MatrixProvider:
    name = "matrix"
    authorized_user_id = "@operator:example.org"
    primary_conversation_id = "!raptor:example.org"
    capabilities = ProviderCapabilities(
        drafts=False,
        reasoning_summaries=False,
        pins=True,
        controls=False,
        typing=True,
    )

    def encode_conversation_id(self, conversation_id): ...
    def decode_conversation_id(self, value): ...
    async def initialize(self, commands): ...
    async def close(self): ...
    async def poll(self, cursor, *, timeout) -> PollResult: ...
    async def send_text(self, conversation_id, text): ...
    async def send_draft(self, conversation_id, draft_id, text): ...
    async def send_reasoning_summary(self, conversation_id, delta): ...
    async def create_message(self, conversation_id, text, controls=()): ...
    async def edit_message(
        self, conversation_id, message_id, text, controls=()
    ): ...
    async def delete_message(self, conversation_id, message_id): ...
    async def pin_message(self, conversation_id, message_id): ...
    async def unpin_message(self, conversation_id, message_id): ...
    async def set_typing(self, conversation_id, active): ...
    async def reject_busy_message(self, conversation_id): ...
    async def acknowledge_queued_message(self, conversation_id): ...
    async def finish_event(self, event): ...
    def prepare_event(self, event): ...
    def capture_delivery_context(self, conversation_id): ...
    def activate_delivery_context(self, conversation_id, context): ...
    def restore_delivery_context(self, token): ...
    async def answer_action(self, action_id, text="", *, alert=False): ...


def create_provider():
    return MatrixProvider()
```

Adapters normalize inbound payloads, own authentication and transport
lifecycle, and declare capability degradation explicitly. Application policy
stays in the provider-neutral core.

## Configuration

Subagent backend settings are independent; they never inherit main-agent
backend settings. Invalid numbers, booleans, duplicate providers, and values
outside the documented ranges stop startup with a configuration error.

### Runtime and providers

| Variable | Default | Purpose |
|---|---:|---|
| `AGENT_WORKDIR` | launch directory | Workspace, shell working directory, and `.skills` parent |
| `RAPTOR_HOME` | `$AGENT_WORKDIR/.raptor` | Durable state and transcript directory |
| `RAPTOR_LOG` | `$RAPTOR_HOME/raptor.log` | Daemon stdout/stderr event log |
| `CHAT_PROVIDERS` | `telegram,responses_api` | Comma-separated built-ins or `module:attribute` providers |
| `CHAT_STREAMING` | `1` | Enable streamed draft previews |
| `CHAT_STREAM_INTERVAL` | `0.35` | Minimum seconds between draft snapshots |
| `MAX_TOOL_ROUNDS` | `0` | Tool-round cap; `0` is uncapped |
| `SHELL_TIMEOUT` | `120` | Shell hard timeout in seconds |
| `MAX_TOOL_OUTPUT` | `30000` | Retained tool-output characters |
| `MAX_PENDING_STEERS` | `64` | Maximum queued root steering inputs |

### Telegram

| Variable | Default | Purpose |
|---|---:|---|
| `TG_BOT_TOKEN` | empty | Bot token; required when Telegram is enabled |
| `TG_USER_ID` | `0` | Authorized Telegram user ID |
| `TG_CHAT_ID` | `TG_USER_ID` | Private chat or forum group served by the bot |
| `TELEGRAM_MARKDOWN` | `1` | Enable Telegram Markdown rendering |

### Inbound Responses API

| Variable | Default | Purpose |
|---|---:|---|
| `RESPONSES_SERVER_HOST` | `0.0.0.0` | Bind address |
| `RESPONSES_SERVER_PORT` | `8787` | Bind port |
| `RESPONSES_SERVER_API_KEY` | `strong-secret` | Bearer token required off loopback |
| `RESPONSES_SERVER_MAX_BODY` | `1048576` | Maximum request body bytes; minimum 1024 |

### Main model backend

| Variable | Default | Purpose |
|---|---:|---|
| `RESPONSES_BASE_URL` | `http://127.0.0.1:8000/v1` | Responses-compatible backend |
| `RESPONSES_API_KEY` | empty | Backend bearer token |
| `RESPONSES_MODEL` | empty | Initial model; otherwise discovered from the backend |
| `RESPONSES_REASONING_EFFORT` | empty | Main-agent reasoning effort; empty uses the model default |
| `RESPONSES_REASONING_SUMMARY` | `auto` | Public reasoning-summary mode; empty omits it |
| `RESPONSES_MAX_RETRIES` | `3` | Retries after the initial transiently failed request |
| `RESPONSES_RETRY_BASE_SECONDS` | `0.5` | Initial exponential-backoff delay |

### Subagent backend

| Variable | Default | Purpose |
|---|---:|---|
| `SUBAGENT_RESPONSES_BASE_URL` | `http://127.0.0.1:8000/v1` | Independent subagent backend |
| `SUBAGENT_RESPONSES_API_KEY` | empty | Independent backend bearer token |
| `SUBAGENT_RESPONSES_MODEL` | empty | Required subagent model |
| `SUBAGENT_RESPONSES_REASONING_EFFORT` | empty | Independent reasoning effort |
| `SUBAGENT_RESPONSES_REASONING_SUMMARY` | `auto` | Public reasoning-summary mode |
| `SUBAGENT_RESPONSES_MAX_RETRIES` | `3` | Independent retries after the initial request |
| `SUBAGENT_RESPONSES_RETRY_BASE_SECONDS` | `0.5` | Independent initial backoff delay |
| `MAX_SUBAGENT_DEPTH` | `3` | Maximum recursive delegation depth; minimum 1 |
| `MAX_SUBAGENT_RECORDS` | `100` | Completed records retained in addition to protected records |
| `MAX_SUBAGENT_TOOL_EVENTS` | `500` | Recent recovery tool events retained per record |
| `MAX_SUBAGENT_PENDING_INPUTS` | `64` | Maximum queued inputs for a running subagent |
| `MAX_BACKGROUND_SUBAGENTS` | `16` | Maximum concurrently running background subagents |

### Context and compaction

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_CONTEXT_TOKENS` | `0` | Main context window; `0` disables proactive checks |
| `SUBAGENT_MODEL_CONTEXT_TOKENS` | `0` | Independent subagent context window |
| `CONTEXT_COMPACT_RATIO` | `0.82` | Proactive compaction ratio; must be 0.50–0.95 |
| `CONTEXT_SAFETY_TOKENS` | `4096` | Tokens reserved below each window |
| `COMPACT_KEEP_RECENT_TOKENS` | `20000` | Native tail retained by normal compaction |
| `COMPACTION_USER_ANCHOR_TOKENS` | `20000` | Original-user-request anchor budget |
| `COMPACTION_MAX_RECORD_CHARS` | `12000` | Per-record compaction rendering cap; minimum 1024 |
| `COMPACTION_OUTPUT_TOKENS` | `4000` | Durable checkpoint token cap; minimum 256 |
| `COMPACTION_GENERATION_TOKENS` | output cap + `4096` | Generation allowance including hidden reasoning |
| `COMPACTION_REASONING_EFFORT` | `low` | Checkpoint-generation reasoning effort |

## Development

Run the complete test suite from this directory:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Run the provider contract separately while developing an adapter:

```bash
uv run python -m unittest tests.test_chat_provider tests.test_multi_provider
```

Changes should preserve the core invariants: one root controller per main chat,
append-only owner-tagged conversation history, provider-affine delivery,
bounded retained state, atomic process ownership, and explicit recovery after
transient failure.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the design, implementation, testing,
and review requirements.

## License

Raptor is licensed under the [MIT License](LICENSE).
