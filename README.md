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
- **Recoverable.** Interrupted turns, goals, and subagent work retain enough
  durable state to continue safely. Managed shells are process-owned and are
  terminated if the daemon exits.
- **Bounded.** Context, tool output, subagent records, and recovery events have
  explicit limits.
- **Observable.** Structured lifecycle events record requests, compaction,
  retries, tools, and completion delivery. Automatic diagnostics redact
  credentials; deliberate shell audit events preserve the exact command.
- **Extensible.** Providers and workspace-local skills use small explicit
  contracts.

## Quick start

Linux and macOS binaries are published on GitHub Releases. Intel and ARM
builds are separate artifacts. Windows is not supported. macOS binaries are
unsigned; Gatekeeper may require a one-time override for a downloaded
executable.

```bash
curl -fsSL \
  https://github.com/qas/raptor/releases/latest/download/install.sh |
  sh
```

The installer places `raptor` in `~/.local/bin`. Add that directory to `PATH`
if the command is not found. Pin a release with `RAPTOR_VERSION=v0.1.0`, or
override the install locations with `RAPTOR_INSTALL_ROOT` and
`RAPTOR_BIN_DIR`.

Remove the installed binary without touching workspace data:

```bash
curl -fsSL \
  https://github.com/qas/raptor/releases/latest/download/install.sh |
  sh -s -- --uninstall
```

`raptor` uses the current directory as `AGENT_WORKDIR` unless configured.
On first startup, Raptor creates missing `AGENTS.md` and `MEMORY.md` templates
in that workspace without replacing existing files. Their bounded UTF-8
contents become persistent instructions and context for the main agent and
subagents for the lifetime of the process; restart Raptor after editing them.

Create `.raptor/config.toml` in the workspace:

```toml
model_provider = "local"
model = "your-model"

[model_providers.local]
base_url = "http://127.0.0.1:8000/v1"
default_model = "your-model"
context_window = 131072

[chat]
providers = ["telegram", "responses_api"]

[telegram]
user_id = 123456789
chat_ids = [123456789]
subagent_topics_silent = true
```

Then provide the Telegram secret and start Raptor:

```bash
export TG_BOT_TOKEN=your-telegram-token

raptor
```

The inbound Responses API listens on `127.0.0.1:8787` by default:

```bash
curl http://127.0.0.1:8787/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"conversation":"project-a","input":"Hello"}'
```

To accept remote connections, set both a non-loopback bind address and a
secret bearer token. Raptor refuses a non-loopback bind without a token.

Model discovery is available at `GET /v1/models`. Live status surfaces and
bounded asynchronous completion messages are available at `GET /v1/status`
for `default`, or at `GET /v1/status?conversation=project-a` for a named
conversation. Asynchronous records have `asynchronous: true`; retention is
bounded independently per conversation and process lifetime. The canonical
assistant turn remains durable in the conversation transcript.

Process commands:

```bash
raptor --version
raptor --status
raptor --check-proxy
raptor --stop-daemon
raptor --daemon
```

`raptor --version` prints the version from `pyproject.toml` and exits before
acquiring the runtime lock or initializing the application.

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
| `/truncate <n>` | Fork history before the last `n` user turns |
| `/models [provider]` | List models served by a configured provider |
| `/model` | Show the current model target and configured providers |
| `/model <provider> <model>` | Start a fresh chat session on that target |
| `/approval` | Toggle tool approval |
| `/todos` | Show the active execution plan |
| `/subagents` | Show subagent status, provider, and model |
| `/goal` | Inspect or manage the persistent goal |
| `/help` | Show available commands |

`/ask` has no prior conversation, goal, or base instructions. It may use normal
tools for multiple in-memory rounds, but neither its model exchange nor answer
is added to the canonical transcript. Tool side effects remain real. The query
runs as the chat's owned turn, so provider polling stays responsive and `/stop`
can cancel it.

`/thread` creates a separate crash-safe transcript. Clearing a thread restores
the parent unchanged. Merging copies only post-fork user, assistant,
function-call, and function-result items. Filesystem, network, and other tool
side effects are not reversible.

`/truncate <n>` creates a new main transcript from the active history before
the last `n` user turns, archives the old transcript with reason
`history_truncated`, and switches only the current session. Model target,
goals, todos, approvals, subagent records, and interrupted state are retained.
Turn starts are direct user messages and user messages merged from a thread;
steering, runtime, goal, and internal inputs are not counted. Eligible
checkpoint/reset state before the cutoff is preserved without expanding the
retired raw prefix. Only user turns in the active native tail can be removed;
Raptor rejects a request that would cross the effective checkpoint/reset
boundary and reports how many active turns are available. The complete raw
history remains in the old full archive.
For provider messages that expose persistent IDs, Raptor also deletes the
user and agent chat messages linked to the removed turns. Telegram records
every split response message and deletes each one individually. Messages from
transcripts created before message-reference tracking cannot be deleted
safely, and provider permission/API failures are reported without guessing
message ranges.
The operation is rejected during an active thread, a session transition, or
while a response for the current session is awaiting delivery. Invalid,
oversized, or failed storage operations leave the current session unchanged.
Raptor durably records a preparing transition before creating the candidate
transcript; restart recovery aborts partial candidates or finishes committed
switches deterministically.
Transcript truncation does not undo files or tool side effects; the old
transcript remains available as audit history.

## Architecture

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Python runtime requirements for development and tests |
| `raptor.spec` | Frozen onedir release bundle for GitHub Releases |
| `install.sh` | Unix installer and uninstaller for published Linux and macOS binaries |
| `raptor.py` | Ownership-first process entry point and CLI |
| `application.py` | Long-running provider and agent application lifecycle |
| `workspace_identity.py` | Workspace identity bootstrap and bounded loading |
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
| `model_providers.py` | `.raptor/config.toml` model-provider registry and target selection |
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
| `shell_supervisor.py` | Child process-group ownership and exit enforcement |
| `shell_sessions.py` | Managed shell state, PTYs, polling, and completion |
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

Managed shells run beneath a supervisor-owned process group. Cancellation,
timeout, normal command exit, and daemon loss all terminate remaining child
processes; a command cannot start until its exact audit event is recorded.

Request-only providers never discard an out-of-band completion. The inbound
Responses adapter retains it in the conversation's bounded asynchronous inbox,
where clients can retrieve it through `GET /v1/status`.

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
to the selected provider's `request_max_retries` exponential-backoff retries.
A valid HTTP
`Retry-After` delay takes precedence over a shorter local delay. Once a stream
has exposed public output, Raptor does not replay it automatically after a
disconnect because doing so could duplicate visible output or tool effects.

If a provider rejects malformed model-generated tool arguments, Raptor
executes nothing, archives the rejected turn and terminal outcome, and retires
the failed context epoch. An active goal pauses, and later messages start with
fresh model context instead of regenerating the same call. The archived chat
remains available to history search.

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

Subagents have isolated transcripts and immutable model targets. A new child
inherits its parent's provider and model unless the parent selects another
configured provider/model in the subagent call. Continuations keep the target
stored with the child transcript. Provider-specific reasoning, retry, and
context-window settings follow that target. Completed record retention and
recovery tool events are bounded; running, interrupted, and undelivered
completion records are protected from pruning. Their private tool history is
never projected into the parent. A subagent compacts lazily when its next
model request needs room, so finishing a child does not trigger speculative
compaction or delay its result.

Calling the agent's `subagent` tool without arguments returns a bounded,
running-first roster with authoritative total, running, and pending-result
counts. Large task text is reduced to a preview; query a specific agent ID for
its detailed public status and final result. If retained history cannot fit in one
tool result, the response reports how many records were returned and omitted
instead of falling through to generic head/tail output truncation.

The background-subagent limit is process-wide. Providers may project safe,
bounded activity without receiving the child's transcript or tool payloads. In
a Telegram forum, Raptor creates a persistent `Subagent: <id>` topic for both
foreground and background children; scheduling mode does not change their
visible activity surface. The delegated task, public reasoning summary,
streamed reply, and final assistant message appear as ordinary messages without
exposing the child's transcript or tool payloads. Raptor removes user input
from that topic because the parent owns steering. The topic remains open across
completed runs and is reused when the same subagent is continued. Deleting a
stopped subagent removes its topic and durable runtime record.

### Telegram chats and forums

Set `TG_CHAT_IDS` to an ordered, comma-separated list of private chats, groups,
or supergroups. The first entry is the default chat. Every configured chat is
an independent main-agent chat; in a forum, the General topic and every normal
topic are independent chats as well. The bot must be an administrator with
**Manage Topics** and **Delete Messages** permissions in each configured
forum so it can manage subagent topics and remove input from them. Only
`TG_USER_ID` is accepted as interactive input.

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

In daemon mode, `raptor.log` is an operator-owned audit record created with
mode `0600`; foreground event output inherits the destination chosen by the
operator. Shell start events include the exact command that Raptor executed,
including any values the operator intentionally placed in it. Automatic
transport and exception events continue to redact recognized credentials.

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
    async def send_text(self, conversation_id, text):
        return ("persistent-message-id",)
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
stays in the provider-neutral core. `send_text` returns every persistent
message ID it created (for example, every part of a split response), or an
empty tuple when the provider has no deletable chat artifact.

## Configuration

Subagents inherit their parent's model target by default. An explicit provider
or model selects a different configured target for that new child, and the
child keeps it for every continuation. Invalid fields, numbers, booleans,
duplicate providers, and values outside the documented ranges stop startup
with a configuration error.

Non-secret settings live in `$RAPTOR_HOME/config.toml`. Existing environment
variables remain supported and override the corresponding TOML value. Bot
tokens, API keys, and credential-bearing proxy URLs remain environment-only.
Unspecified values use the defaults shown below.

```toml
[network]
proxy = "http://proxy.example:8080"
no_proxy = ["models.internal", "*.example.net"]

[chat]
providers = ["telegram", "responses_api"]
streaming = true
stream_interval = 0.35
max_pending_steers = 64
max_runtimes = 1024

[telegram]
user_id = 123456
chat_ids = [123456, -1001234567890]
max_retries = 3
markdown = true
subagent_topics_silent = true

[responses_server]
host = "127.0.0.1"
port = 8787
max_body = 1048576
max_connections = 128
max_pending = 64
max_status_messages = 256
max_stream_events = 256
read_timeout = 10.0

[subagents]
max_depth = 3
max_records = 100
max_tool_events = 500
max_pending_inputs = 64
max_background = 16

[tools]
max_rounds = 0
max_output = 30000

[shell]
timeout = 0

[state]
max_load_bytes = 16777216

[compaction]
output_tokens = 4000
generation_tokens = 8096
reasoning_effort = "low"
keep_recent_tokens = 20000
user_anchor_tokens = 20000
max_record_chars = 12000
context_ratio = 0.82
context_safety_tokens = 4096
```

### Runtime and providers

| Variable | Default | Purpose |
|---|---:|---|
| `AGENT_WORKDIR` | launch directory | Workspace, shell working directory, and `.skills` parent |
| `RAPTOR_HOME` | `$AGENT_WORKDIR/.raptor` | Durable state and transcript directory |
| `RAPTOR_CONFIG` | `$RAPTOR_HOME/config.toml` | Raptor TOML configuration file |
| `RAPTOR_LOG` | `$RAPTOR_HOME/raptor.log` | Daemon stdout/stderr event log |
| `RAPTOR_PROXY` | empty | Outbound `http`, `https`, or remote-DNS `socks5h` proxy |
| `RAPTOR_NO_PROXY` | empty | Comma-separated exact hosts or `*.` subdomain patterns routed directly |
| `CHAT_PROVIDERS` | `telegram,responses_api` | Comma-separated built-ins or `module:attribute` providers |
| `CHAT_STREAMING` | `1` | Enable streamed draft previews |
| `CHAT_STREAM_INTERVAL` | `0.35` | Minimum seconds between draft snapshots |
| `MAX_TOOL_ROUNDS` | `0` | Tool-round cap; `0` is uncapped |
| `SHELL_TIMEOUT` | `0` | Default shell deadline in seconds; `0` disables it |
| `MAX_TOOL_OUTPUT` | `30000` | Tool text, output, shell-input, and audit-command character budget |
| `MAX_PENDING_STEERS` | `64` | Maximum queued root steering inputs |
| `MAX_CHAT_RUNTIMES` | `1024` | Maximum provider conversations admitted per process |
| `MAX_STATE_LOAD_BYTES` | `16777216` | Maximum state file bytes accepted at startup |

`AGENT_WORKDIR`, `RAPTOR_HOME`, `RAPTOR_CONFIG`, and `RAPTOR_LOG` are bootstrap
paths and remain environment-only because they determine where configuration
and runtime files are found.

When `RAPTOR_PROXY` is set, Raptor routes built-in outbound HTTP traffic
through that proxy and fails non-bypassed requests if it is unavailable.
`RAPTOR_NO_PROXY` routes matching destination hosts directly. Exact entries
match only that host; `*.example.com` matches subdomains such as
`api.example.com`, but not `example.com`. List both forms to bypass both.
Bypassed hosts use local DNS. Ambient proxy variables and `NO_PROXY` are
ignored. Managed shell commands and custom provider implementations retain
their own network configuration and are outside this routing guarantee.
Run `raptor --check-proxy` to make an explicit bounded request through the
proxy to `api.ipify.org`; it ignores `RAPTOR_NO_PROXY` and prints the observed
public egress IP without displaying the configured proxy address.

```bash
RAPTOR_PROXY=https://proxy.example:8443 raptor
RAPTOR_PROXY=socks5h://proxy.example:1080 raptor
RAPTOR_PROXY=socks5h://proxy.example:1080 \
  RAPTOR_NO_PROXY='models.internal,google.com,*.google.com' \
  raptor
```

### Telegram

| Variable | Default | Purpose |
|---|---:|---|
| `TG_BOT_TOKEN` | empty | Bot token; required when Telegram is enabled |
| `TG_USER_ID` | `0` | Authorized Telegram user ID |
| `TG_CHAT_IDS` | `TG_USER_ID` | Ordered, comma-separated chats served by the bot |
| `TG_MAX_RETRIES` | `3` | Retries after a Telegram flood-control response |
| `TELEGRAM_MARKDOWN` | `1` | Enable Telegram Markdown rendering |
| `TELEGRAM_SUBAGENT_TOPICS_SILENT` | `1` | Send subagent-topic messages without notifications |

### Inbound Responses API

| Variable | Default | Purpose |
|---|---:|---|
| `RESPONSES_SERVER_HOST` | `127.0.0.1` | Bind address |
| `RESPONSES_SERVER_PORT` | `8787` | Bind port |
| `RESPONSES_SERVER_API_KEY` | empty | Bearer token; required off loopback |
| `RESPONSES_SERVER_MAX_BODY` | `1048576` | Maximum request body bytes; minimum 1024 |
| `RESPONSES_SERVER_MAX_CONNECTIONS` | `128` | Maximum simultaneous HTTP connections |
| `RESPONSES_SERVER_MAX_PENDING` | `64` | Maximum queued or active requests |
| `RESPONSES_SERVER_MAX_STATUS_MESSAGES` | `256` | Live and asynchronous records retained per conversation |
| `RESPONSES_SERVER_MAX_STREAM_EVENTS` | `256` | Buffered SSE events per active response |
| `RESPONSES_SERVER_READ_TIMEOUT` | `10.0` | Header and body read deadline in seconds |

### Model providers

Outbound model configuration lives in `.raptor/config.toml`, or the file set
by `RAPTOR_CONFIG`. Environment variables are used only for secrets named by
`api_key_env`; secret values are never persisted in transcripts or state.

```toml
model_provider = "openai"
model = "gpt-5"

[model_providers.openai]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-5"
request_max_retries = 3
retry_base_seconds = 0.5
context_window = 400000
reasoning_effort = "high"
reasoning_summary = "auto"

[model_providers.local]
base_url = "http://127.0.0.1:8000/v1"
default_model = "local-model"
context_window = 131072

[model_providers.local.models."small-model"]
context_window = 32768
reasoning_effort = "low"
```

Each provider must expose Responses-compatible `/responses` and `/models`
endpoints. Model tables override provider defaults. Omitting top-level `model`
uses the selected provider's `default_model`; with no config file, Raptor
probes the local provider and selects its first served model.

Changing `/model` archives the current root transcript and starts a fresh one,
so provider-private response items never cross provider boundaries.

### Subagents

| Variable | Default | Purpose |
|---|---:|---|
| `MAX_SUBAGENT_DEPTH` | `3` | Maximum recursive delegation depth; minimum 1 |
| `MAX_SUBAGENT_RECORDS` | `100` | Retained subagent records and persistent activity surfaces |
| `MAX_SUBAGENT_TOOL_EVENTS` | `500` | Recent recovery tool events retained per record |
| `MAX_SUBAGENT_PENDING_INPUTS` | `64` | Maximum queued inputs for a running subagent |
| `MAX_BACKGROUND_SUBAGENTS` | `16` | Maximum concurrently running background subagents |

### Context and compaction

| Variable | Default | Purpose |
|---|---:|---|
| `CONTEXT_COMPACT_RATIO` | `0.82` | Proactive compaction ratio; must be 0.50–0.95 |
| `CONTEXT_SAFETY_TOKENS` | `4096` | Tokens reserved below each window |
| `COMPACT_KEEP_RECENT_TOKENS` | `20000` | Native tail retained by normal compaction |
| `COMPACTION_USER_ANCHOR_TOKENS` | `20000` | Original-user-request anchor budget |
| `COMPACTION_MAX_RECORD_CHARS` | `12000` | Per-record compaction rendering cap; minimum 1024 |
| `COMPACTION_OUTPUT_TOKENS` | `4000` | Durable checkpoint token cap; minimum 256 |
| `COMPACTION_GENERATION_TOKENS` | output cap + `4096` | Generation allowance including hidden reasoning |
| `COMPACTION_REASONING_EFFORT` | `low` | Checkpoint-generation reasoning effort |

## Run from source

Raptor requires Python 3.11 or newer and a Responses-compatible model backend.
Project metadata declares its Python dependency for `uv`. Custom
`module:attribute` chat providers also require a source checkout; frozen
releases include the built-in Telegram and Responses API providers.

```bash
uv run raptor.py
uv run raptor.py --status
uv run raptor.py --stop-daemon
uv run raptor.py --daemon
```

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

To publish Linux and macOS binaries, set `project.version` in
`pyproject.toml`, then push a matching `v*` tag. GitHub Actions tests, freezes
`raptor`, and uploads the release assets.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the design, implementation, testing,
and review requirements.

## License

Raptor is licensed under the [MIT License](LICENSE).
