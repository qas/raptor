# Raptor

![Raptor](assets/banner.png)

Raptor is a persistent runtime for tool-using agents. It uses the OpenAI
Responses API and provides durable conversations, bounded context compaction,
goals, steering, approvals, managed shell sessions, skills, and isolated
subagents through a provider-neutral chat interface.

Raptor includes Telegram and an inbound, Responses-compatible HTTP API. You can
run both chat providers at the same time. Each chat-provider conversation owns
an isolated, durable main chat. Replies and transport state never cross chat
boundaries.

## Key properties

- **Durable by default.** An append-only JSONL transcript is the source of
  truth. Checkpoint compaction changes the model window, not the archive.
- **Provider-neutral.** The core processes normalized events and opaque
  identifiers. Adapters own transport-specific behavior.
- **Recoverable.** Interrupted turns, goals, and subagent work retain enough
  durable state to continue safely. Raptor owns managed shell processes and
  terminates them when the daemon exits.
- **Bounded.** Context, tool output, subagent records, and recovery events have
  explicit limits.
- **Observable.** Structured lifecycle events record requests, compaction,
  retries, tools, and completion delivery. Automatic diagnostics redact
  credentials; deliberate shell audit events preserve the exact command.
- **Extensible.** Chat providers and workspace-local skills use small,
  explicit contracts.

## Install and start Raptor

GitHub Releases provides binaries for Linux and macOS, with separate artifacts
for Intel and ARM. Raptor does not support Windows. The macOS binaries are
unsigned, so Gatekeeper might require a one-time override after download.

```bash
curl -fsSL \
  https://github.com/qas/raptor/releases/latest/download/install.sh |
  sh
```

The installer writes `raptor` to `~/.local/bin`. If your shell cannot find the
command, add that directory to `PATH`. By default, the installer downloads the
latest stable release. To install a published stable or prerelease tag, set
`RAPTOR_VERSION`. To install the latest successful build of `main`, run:

```bash
curl -fsSL \
  https://github.com/qas/raptor/releases/download/nightly/install.sh |
  RAPTOR_VERSION=nightly sh
```

Nightly builds use their commit SHA as the build identifier and are intended
for early testing. To change the installation locations, set
`RAPTOR_INSTALL_ROOT` and `RAPTOR_BIN_DIR`.

On Linux, the installer probes Bubblewrap and reports whether restricted shell
commands can start. It does not silently install packages, add AppArmor
exceptions, or weaken kernel policy. If Bubblewrap is missing, the installer
prints the package-manager command. On Ubuntu 24.04 and later, it identifies
the AppArmor user-namespace restriction separately so a system administrator
can make an explicit policy decision.

Remove the installed binary without touching workspace data:

```bash
curl -fsSL \
  https://github.com/qas/raptor/releases/latest/download/install.sh |
  sh -s -- --uninstall
```

Unless you configure it explicitly, Raptor uses the current directory as
`AGENT_WORKDIR`. On first startup, Raptor creates any missing `AGENTS.md` and
`MEMORY.md` templates and a `.raptor/skills/create-skill/SKILL.md` starter. It
does not replace existing files. Raptor loads the bounded UTF-8 contents of the
identity files as persistent instructions and context for the main agent and
subagents. Restart Raptor after you edit these files.

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
tool_activity = true

[telegram]
user_id = 123456789
chat_ids = [123456789]
subagent_topics_silent = true
```

Set the Telegram bot token, then start Raptor:

```bash
export TG_BOT_TOKEN=your-telegram-token

raptor
```

To run Raptor in the background, use daemon mode:

```bash
raptor -d
```

`-d` is the short form of `--daemon`. By default, Raptor writes daemon output
to `$RAPTOR_HOME/raptor.log`.

The inbound Responses API listens on `127.0.0.1:8787` by default:

```bash
curl http://127.0.0.1:8787/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"conversation":"project-a","input":"Hello"}'
```

To accept remote connections, configure both a non-loopback bind address and a
secret bearer token. Raptor rejects a non-loopback configuration that does not
include a token.

Use `GET /v1/models` to discover models. Use `GET /v1/status` to retrieve live
status and bounded asynchronous completion messages for the `default`
conversation. For a named conversation, add the query parameter, for example
`GET /v1/status?conversation=project-a`. Asynchronous records include
`asynchronous: true`. Raptor bounds retention separately for each conversation,
and the records last only for the process lifetime. The canonical assistant
turn remains durable in the conversation transcript.

Process commands:

```bash
raptor --version
raptor --status
raptor --check-proxy
raptor --stop-daemon
raptor --daemon
```

`raptor --version` prints the selected version and commit for a tagged build.
For a nightly build, it prints the commit and build date. The command exits
before it acquires the runtime lock or initializes the application. Source
checkouts use the development version from `pyproject.toml`.

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
| `/task <message>` | Run isolated tool-capable work outside conversation history |
| `/thread` | Fork a temporary conversation branch |
| `/thread clear` | Discard the branch and return to its parent |
| `/thread merge` | Merge branch-native conversation items into its parent |
| `/status` | Show runtime, context, goal, subagent, and shell status |
| `/stop` | Interrupt the current root turn; background work continues |
| `/stop all` | Interrupt and cancel background work in the current main chat |
| `/compact` | Force a durable checkpoint of available native history |
| `/truncate <n>` | Fork history before the last `n` user turns |
| `/models [provider]` | List models served by a configured model provider |
| `/model` | Show the current model target and configured model providers |
| `/model <provider> <model>` | Start a new chat session on that target |
| `/approval` | Toggle tool approval |
| `/todos` | Show the active execution plan |
| `/subagents` | Show subagent status, model provider, and model |
| `/console <command>` | Run one bounded command in the managed shell sandbox |
| `/console -f <command>` | Follow live output until exit or Stop |
| `/shutdown` | Clean up owned work and stop the Raptor process |
| `/restart` | Clean up owned work and replace the Raptor process |
| `/goal` | Inspect or manage the persistent goal |
| `/help` | Show available commands |

`/task` runs without prior conversation or active-goal context. It uses the same
base instructions, workspace `AGENTS.md` and `MEMORY.md` files, and discovered
skill catalog as a normal agent turn. The task can use standard tools for
multiple in-memory rounds, but Raptor does not add its model exchange or answer
to the canonical transcript. Tool side effects still apply. The task runs as
the chat's owned turn, so chat-provider polling remains responsive and `/stop`
can cancel it.

`/console` uses the same authorized-user check as every other interactive
command. It applies the configured filesystem policy, records the exact
command in the shell audit log, and bounds retained output. Sandbox preparation
has a 60-second limit. After the command starts, it has a 20-second execution
limit. Results use a fenced Bash block containing the command and its output.
`/console -f` (or `--follow`) instead edits one live terminal message until the
command exits or the operator presses Stop. Raptor owns at most one followed
console, bounds its retained screen, and stops it during shutdown. Neither mode
bypasses the configured sandbox or filesystem policy.

`/shutdown` pauses an active goal, cleans up owned turns, subagents, and shell
processes, and then exits. A stopped process cannot receive chat commands. To
start it again, run `raptor --daemon` on the host. `/restart` performs the same
cleanup, preserves an active goal, clears published runtime state, and replaces
the process by using its original command line.

`/thread` creates a separate crash-safe transcript. Clearing a thread restores
the parent unchanged. Merging copies only post-fork user, assistant,
function-call, and function-result items. Filesystem, network, and other tool
side effects are not reversible.

`/truncate <n>` creates a new main transcript from the history before the last
`n` user turns. It archives the old transcript with the reason
`history_truncated` and switches only the current session. Raptor retains the
model target, goals, todos, approvals, subagent records, and interrupted state.

Raptor counts direct user messages and user messages merged from a thread as
turn starts. It does not count steering, runtime, goal, or internal inputs.
Only user turns in the active native tail can be removed. Raptor rejects a
request that crosses the effective checkpoint or reset boundary and reports
the number of available active turns. It preserves eligible checkpoint and
reset state before the cutoff without expanding the retired raw prefix.

When a chat provider exposes persistent message IDs, Raptor also deletes the
user and agent messages associated with the removed turns. Telegram records
and deletes each part of a split response individually. Raptor cannot safely
delete messages from transcripts created before message-reference tracking.
It reports provider permission and API failures without guessing message
ranges.

Raptor rejects truncation during an active thread or session transition, or
while a response for the current session awaits delivery. Invalid, oversized,
or failed storage operations leave the current session unchanged. Before it
creates the candidate transcript, Raptor records a durable preparing
transition. During restart recovery, it deterministically aborts a partial
candidate or finishes a committed switch.

Truncation does not undo filesystem or tool side effects. The complete raw
history remains available in the archived transcript as an audit record.

## Architecture

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Python runtime requirements for development and tests |
| `raptor.spec` | Frozen onedir release bundle for GitHub Releases |
| `install.sh` | Unix installer and uninstaller for published Linux and macOS binaries |
| `raptor.py` | Thin source and frozen executable launcher |
| `raptor/entrypoint.py` | Ownership-first process entry point and CLI |
| `raptor/app/application.py` | Provider and agent application lifecycle |
| `raptor/app/application_control.py` | Shutdown and restart requests |
| `raptor/app/workspace_identity.py` | Workspace identity bootstrap and loading |
| `raptor/app/process_lock.py` | State-independent atomic process ownership |
| `raptor/app/runtime.py` | Process metadata and daemon controls |
| `raptor/runtime_paths.py` | Dependency-free runtime filesystem locations |
| `raptor/agent/agent.py` | Root turn execution and compaction integration |
| `raptor/agent/engine.py` | Tool-round engine and response parsing |
| `raptor/agent/controller.py` | Single-owner root scheduling and goal continuation |
| `raptor/agent/turn_runtime.py` | Root-turn ownership and bounded interruption |
| `raptor/agent/runtime_events.py` | Typed background-completion events |
| `raptor/observability.py` | Structured runtime events and activity labels |
| `raptor/model/responses.py` | Outbound Responses client, streaming, and retry policy |
| `raptor/model/model_providers.py` | Model-provider registry and target selection |
| `raptor/model/response_errors.py` | Shared Responses protocol errors |
| `raptor/chat/chat_provider.py` | Provider protocol and normalized event types |
| `raptor/chat/chat_runtime.py` | Provider loading, binding, and delivery context |
| `raptor/chat/providers/multi_provider.py` | Concurrent provider routing |
| `raptor/chat/providers/telegram.py` | Telegram adapter |
| `raptor/chat/providers/responses_provider.py` | Responses-compatible HTTP adapter |
| `raptor/chat/presentation.py` | Provider-neutral status and activity policy |
| `raptor/chat/tool_activity.py` | Streamed tool-call status and lifecycle projection |
| `raptor/state/chat_store.py` | Append-only transcript storage |
| `raptor/state/storage.py` | Crash-safe atomic local file replacement |
| `raptor/agent/context.py` | Active-context construction and checkpoint compaction |
| `raptor/state/session.py` | Durable chat registry and context-bound per-chat runtimes |
| `raptor/agent/thread_state.py` | Temporary-thread state queries |
| `raptor/agent/thread_status.py` | Temporary-thread status projection |
| `raptor/agent/subagents.py` | Isolated foreground and background subagents |
| `raptor/shell/shell_supervisor.py` | Child process-group ownership and exit enforcement |
| `raptor/shell/shell_sessions.py` | Managed shell state, PTYs, polling, and completion |
| `raptor/agent/skills.py` | Progressive workspace skill discovery and loading |
| `raptor/chat/commands.py` | Provider-neutral slash commands |
| `raptor/agent/threads.py` | Temporary branch lifecycle and merge policy |

### Main chats, scheduling, and delivery

Each chat-provider conversation owns one main-chat runtime. The runtime
contains the transcript selection, goal, todos, thread, approvals, steering
queue, background resources, status surface, and root controller. A chat runs
one root turn at a time, but different chats can run concurrently. Model
selection and process limits apply to the entire process.

The owning chat's controller receives user input, steering, background shell
completions, subagent completions, and goal continuations. These events do not
start competing runs. Each completion event identifies its chat owner, so
another chat cannot consume it.

An immediately applied queued steer interrupts only the root turn. If the root
was waiting for a foreground subagent, that subagent moves to background
ownership, continues its durable run, and reports its result asynchronously.
An ordinary queued steer applied at the next safe point does not interrupt
either task.

The root agent can cancel one background subagent or managed shell owned by its
chat using the returned identifier. Targeted cancellation suppresses completion
delivery; `/stop all` cancels the current chat's root turn, queued work,
subagents, and shells without disturbing other chats.

Managed shells run beneath a supervisor-owned process group. Cancellation,
timeout, normal command exit, and daemon loss all terminate remaining child
processes; a command cannot start until its exact audit event is recorded.

Request-only chat providers never discard an out-of-band completion. The
inbound Responses adapter retains it in the conversation's bounded
asynchronous inbox, where clients can retrieve it through `GET /v1/status`.

Each queued request retains its originating conversation and chat-provider
delivery context. A Responses HTTP request that becomes steering remains open
and receives the answer produced for that specific queued input. Set the
`conversation` field to select a named API chat. If you omit it, Raptor selects
`default`. Telegram replies remain in their originating topic.

The persistent status slot has this priority:

```text
thread > goal > empty
```

Each root-agent tool call receives a new bounded, unpinned status bubble. The
bubble streams arguments while Raptor prepares the call, then changes to
`Running` and finally to `Completed`, `Failed`, `Denied`, or `Interrupted`.
When approval is enabled, Raptor adds Approve and Deny controls before
execution. Terminal bubbles remain visible during the turn. Raptor removes
them before it sends the final answer. Tool activity never replaces an active
thread or goal pin. `/task` uses the same lifecycle after its non-streaming model
response produces a tool call. On Telegram, shell bubbles open in `Console`,
which streams the latest seven output lines; one button toggles between that
view and `Info`. Info renders arguments as labeled fields, with multiline and
nested values in code blocks. Empty `write_stdin` polls edit one waiting bubble
every five seconds instead of exposing polling arguments. Steering is transient
and is never pinned. Set `chat.tool_activity = false` to hide nonessential
transient tool bubbles. Approval prompts remain visible because they require an
operator decision.

### Context and compaction

Raptor maintains three distinct layers:

1. **Canonical transcript:** lossless append-only JSONL.
2. **Active context:** the latest checkpoint, preserved user anchors, and the
   native tail after the checkpoint boundary.
3. **Checkpoint:** a model-generated continuation summary that retires older
   items from the model window only.

Compaction sends a plain rendered record set instead of incomplete native tool
call fragments. If a completed compaction has no visible summary, Raptor
retries it once. If the second result is also empty, Raptor classifies the
failure as transient and pauses an active goal instead of blocking it. Chat
providers can show a temporary animated `Compacting.`, `Compacting..`, or
`Compacting...` indicator. `/status` reports the configured context limit and
threshold compaction state even when the model provider's live model-list
request is unavailable.

When compaction occurs during an in-progress tool turn, Raptor marks the
rebuilt request as a continuation. The model does not repeat completed or
already communicated work and resumes only unresolved actions. Manual and
post-response compaction do not add this transient continuation input.

### Retries and continuation

Transient transport errors include connection failures, disconnects such as
`RemoteProtocolError`, incomplete streams, and retryable HTTP statuses: `408`,
`429`, `500`, `502`, `503`, and `504`. After the initial request, Raptor makes
up to the selected model provider's `request_max_retries` retry attempts with
exponential backoff. A valid HTTP `Retry-After` delay takes precedence over a
shorter local delay. After a stream exposes public output, Raptor does not
automatically replay it following a disconnect because replay could duplicate
visible output or tool effects.

If a model provider rejects malformed model-generated tool arguments, Raptor
executes nothing, archives the rejected turn and terminal outcome, and retires
the failed context epoch. An active goal pauses, and later messages start with
fresh model context instead of regenerating the same call. The archived chat
remains available to history search.

Retry exhaustion ends only the current attempt:

- An ordinary root turn releases the controller, so the user can send another
  message.
- An active goal pauses and can continue with `/goal resume`.
- A subagent retains its transcript and can continue by agent ID.

There is no unbounded automatic retry loop.

### Shell sessions and subagents

Shell commands wait for `yield_time_ms`. If a command is still running after
that interval, the tool returns a managed session ID. `write_stdin` polls
output or writes to a PTY. Detached completions re-enter through the root
controller and retry delivery after a temporary controller failure. `/stop all`
terminates every live process group owned by the current main chat.

Subagents have isolated transcripts and immutable model targets. A new child
inherits its parent's model provider and model unless the parent selects
another configured target in the subagent call. Continuations use the target
stored with the child transcript. Provider-specific reasoning, retry, and
context-window settings follow that target. Raptor bounds completed-record and
recovery-tool-event retention. It does not prune running, interrupted, or
undelivered completion records. A subagent's private tool history never appears
in its parent. Subagents compact lazily when the next model request needs room,
so completing a child does not trigger speculative compaction or delay its
result.

Calling the agent's `subagent` tool without arguments returns a bounded roster
with running agents first and authoritative total, running, and pending-result
counts. Raptor reduces long task text to a preview. Query a specific agent ID
for detailed public status and its final result. If retained history does not
fit in one tool result, the response reports the number of returned and omitted
records instead of applying generic head-and-tail output truncation.

The background-subagent limit applies to the entire process. Chat providers
can show safe, bounded activity without receiving the child's transcript or
tool payloads. In a Telegram forum, Raptor creates a persistent
`Subagent: <id>` topic for foreground and background children. Scheduling mode
does not change the visible activity surface. The delegated task, public
reasoning summary, streamed reply, and final assistant message appear as
ordinary messages without exposing the child's transcript or tool payloads.

Root and child tools use the same status-bubble implementation. With approval
enabled, the child's bubble includes the same preview and Approve and Deny
controls in its topic. With approval disabled, the bubble streams the same
`Preparing tool`, `Running`, and terminal lifecycle as a root tool. Each call
uses one bubble. Raptor does not add a second status message for the same call.

Raptor removes user input from the subagent topic because the parent owns
steering. The topic remains open after a run completes, and Raptor reuses it
when the same subagent continues. Deleting a stopped subagent removes both its
topic and durable runtime record. Each parent-authored steer starts a new reply
segment, so later streaming appears below the steer instead of editing an
earlier reply above it.

### Telegram chats and forums

Set `TG_CHAT_IDS` to an ordered, comma-separated list of private chats, groups,
or supergroups. The first entry is the default chat. Every configured chat is
an independent main chat. In a forum, the General topic and every normal topic
are also independent chats. The bot must be an administrator with
**Manage Topics** and **Delete Messages** permissions in each configured
forum so it can manage subagent topics and remove input from them. Raptor
accepts interactive input only from `TG_USER_ID`.

Create and name normal topics in the Telegram UI. Raptor discovers a topic when
it receives the first message and then persists the topic's runtime. Activity
topics are presentation-only and cannot steer their subagent. Send steering
messages from the parent topic. `subagent_topics_silent` maps to Telegram's
silent-message flag. It suppresses notification sounds, but Telegram clients
can still show notifications and unread badges. Bots cannot mute a topic in a
user's Telegram settings.

## Storage

All durable runtime data lives under `RAPTOR_HOME`:

```text
$AGENT_WORKDIR/.raptor/
  runtime.lock
  state.json
  raptor.log                 # daemon mode
  providers/
    telegram.cursor          # finalized Telegram update offset
  chats/
    <session-id>.jsonl
```

In daemon mode, Raptor creates `raptor.log` with mode `0600` as an
operator-owned audit record. In foreground mode, event output uses the
destination selected by the operator. Shell start events include the exact
command that Raptor executed, including values that the operator deliberately
included. Automatic transport and exception events redact recognized
credentials.

Telegram advances `providers/telegram.cursor` only after an update finishes
core handling and transport finalization. After a restart, polling resumes from
that durable offset, so a completed lifecycle command is not replayed.

`/new` creates a transcript without deleting the previous one. The
`chat_history` tool can search archived sessions, but only from their owning
main chat. Transcript `session_start` records contain the main-chat key, and
recovery rejects cross-chat references. When a state schema does not match,
Raptor reports an explicit failure instead of guessing or silently discarding
durable state.

## Skills

Raptor discovers skills recursively under `$AGENT_WORKDIR/.raptor/skills` and
`$AGENT_WORKDIR/.agent/skills` using this layout:

```text
.raptor/skills/
  skill-name/
    SKILL.md

.agent/skills/
  skill-name/
    SKILL.md
```

During discovery, Raptor loads only frontmatter metadata into the standard
instructions. It loads the complete `SKILL.md` when a user names the skill or
the task matches its description. Referenced resources remain unloaded until
the skill needs them. Root agents and subagents share the catalog. On startup,
Raptor creates `.raptor/skills/create-skill/SKILL.md` if it is missing and does
not replace an existing file.

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
    async def publish_process_output(self, conversation_id, chunk): ...
    async def create_message(self, conversation_id, text, controls=()): ...
    async def edit_message(
        self, conversation_id, message_id, text, controls=()
    ): ...
    async def delete_message(self, conversation_id, message_id): ...
    async def delete_messages(self, conversation_id, message_ids): ...
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

Adapters normalize inbound payloads, own authentication and the transport
lifecycle, and explicitly declare unavailable capabilities. Application
policy remains in the provider-neutral core. `send_text` returns every
persistent message ID that it creates, including every part of a split
response. It returns an empty tuple if the provider creates no deletable chat
artifact.

`publish_process_output` is optional. When implemented, it receives bounded,
decoded stdout and stderr chunks with their shell-session and tool-call IDs.
The adapter decides whether to stream, batch, render, or ignore them. Delivery
backpressure is bounded by the subprocess pipe and a core timeout. Adapter
failures do not change the command result.

## Configuration

Store non-secret settings in `$RAPTOR_HOME/config.toml`. Environment variables
override the corresponding TOML values. Keep bot tokens, API keys, and proxy
URLs that contain credentials in environment variables. Settings that you do
not specify use the defaults shown below. Invalid fields, numbers, booleans,
duplicate providers, and values outside the documented ranges cause a startup
configuration error.

```toml
[permissions.filesystem]
# Paths are relative to AGENT_WORKDIR unless absolute.
deny_read = [".env", "**/.env", "**/*.pem"]
glob_scan_max_depth = 32

[network]
proxy = "http://proxy.example:8080"
no_proxy = ["models.internal", "*.example.net"]

[chat]
providers = ["telegram", "responses_api"]
streaming = true
stream_interval = 0.35
tool_activity = true
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
# Optional: omit both to use each agent's active model target.
# model_provider = "economy"
# model = "economy-model"
output_tokens = 4000
generation_tokens = 8096
reasoning_effort = "low"
keep_recent_tokens = 20000
user_anchor_tokens = 20000
max_record_chars = 12000
context_ratio = 0.82
context_safety_tokens = 4096
```

### Filesystem permissions

`permissions.filesystem.deny_read` prevents Raptor's filesystem tools and
managed shell commands from accessing matching files and directories. An exact
path protects the complete subtree when it names a directory. Patterns support
`*`, `?`, `[]`, and recursive `**` glob segments. Relative patterns resolve
from `AGENT_WORKDIR`. Raptor rejects root-wide globs such as `/**/*.pem` so a
misconfiguration cannot trigger an unbounded full-disk scan.

When you configure shell enforcement, it fails closed. Linux requires a
root-owned `bwrap` (Bubblewrap) executable on `PATH` that is not writable by
the group or other users. Bubblewrap also requires access to unprivileged user
namespaces. Ubuntu 24.04 and later restrict that access with AppArmor by
default. Raptor's installer probes the complete requirement and reports the
host-policy failure without disabling the protection globally. The installer
runs that probe through the installed Raptor executable so application-specific
AppArmor policy is applied. macOS uses `/usr/bin/sandbox-exec`.

#### Ubuntu AppArmor

Ubuntu 24.04 and later can deny Bubblewrap's user-namespace setup even when
`kernel.unprivileged_userns_clone` is enabled. To authorize installed Raptor
versions without disabling the restriction globally, create a named AppArmor
profile:

```bash
RAPTOR_BIN="$(readlink -f "$(command -v raptor)")"
RAPTOR_VERSIONS="$(dirname "$(dirname "$RAPTOR_BIN")")"

sudo tee /etc/apparmor.d/raptor >/dev/null <<EOF
abi <abi/4.0>,
include <tunables/global>

profile raptor ${RAPTOR_VERSIONS}/*/raptor flags=(unconfined) {
  userns,
}
EOF

sudo apparmor_parser -r /etc/apparmor.d/raptor
raptor --check-sandbox
```

This is an explicit host-security decision. The installer detects the denial
and links to these commands, but does not write system policy or disable
AppArmor itself. Restart Raptor after loading the profile.

Raptor limits glob expansion to 1,024 matches and 250,000 scanned entries. The
`glob_scan_max_depth` setting limits recursive traversal and defaults to `32`.
If permissions prevent enumeration or traversal reaches the depth limit,
shell enforcement treats the directory as an opaque, denied subtree. It does
not allow a possible match beneath that directory.

Recursive globs do not cross directory symlinks. A symlinked repository or
environment is a separate scan boundary and does not make unrelated shell
commands fail. If an exact pattern, a matching file entry, or a glob's fixed
prefix itself goes through a symlink, enforcement fails closed because a
writable logical symlink cannot be securely represented by a physical-target
mount. Patterns with the same fixed directory prefix share one traversal.

Raptor expands shell patterns immediately before each managed command.
Filesystem tools evaluate paths at the time of each tool call. If you do not
configure `deny_read`, it defaults to an empty list and does not change shell
behavior.

### Compaction model

Use `compaction.model_provider` and `compaction.model` to route checkpoint
summarization through another configured model target, such as a less expensive
model. If you omit both settings, compaction uses the active chat or subagent
target. If you set only `model_provider`, compaction uses that provider's
`default_model`. If you set only `model`, compaction uses that model with the
active target's provider. These settings do not change the model assigned to
the conversation or subagent.

### Runtime and chat providers

| Variable | Default | Purpose |
|---|---:|---|
| `AGENT_WORKDIR` | launch directory | Workspace, shell working directory, and skill-root parent |
| `RAPTOR_HOME` | `$AGENT_WORKDIR/.raptor` | Durable state and transcript directory |
| `RAPTOR_CONFIG` | `$RAPTOR_HOME/config.toml` | Raptor TOML configuration file |
| `RAPTOR_LOG` | `$RAPTOR_HOME/raptor.log` | Daemon stdout/stderr event log |
| `RAPTOR_PROXY` | empty | Outbound `http`, `https`, or remote-DNS `socks5h` proxy |
| `RAPTOR_NO_PROXY` | empty | Comma-separated exact hosts or `*.` subdomain patterns routed directly |
| `CHAT_PROVIDERS` | `telegram,responses_api` | Comma-separated built-ins or `module:attribute` providers |
| `CHAT_STREAMING` | `1` | Enable streamed draft previews |
| `CHAT_STREAM_INTERVAL` | `0.35` | Minimum seconds between draft snapshots |
| `CHAT_TOOL_ACTIVITY` | `1` | Show transient tool activity bubbles |
| `MAX_TOOL_ROUNDS` | `0` | Tool-round cap; `0` is uncapped |
| `SHELL_TIMEOUT` | `0` | Default shell deadline in seconds; `0` disables it |
| `MAX_TOOL_OUTPUT` | `30000` | Tool text, output, shell-input, and audit-command character budget |
| `MAX_PENDING_STEERS` | `64` | Maximum queued root steering inputs |
| `MAX_CHAT_RUNTIMES` | `1024` | Maximum provider conversations admitted per process |
| `MAX_STATE_LOAD_BYTES` | `16777216` | Maximum state file bytes accepted at startup |

Configure `AGENT_WORKDIR`, `RAPTOR_HOME`, `RAPTOR_CONFIG`, and `RAPTOR_LOG`
through environment variables. These bootstrap paths determine where Raptor
finds configuration and runtime files, so they are not TOML settings.

When you set `RAPTOR_PROXY`, Raptor routes built-in outbound HTTP traffic
through that proxy. A non-bypassed request fails if the proxy is unavailable.
Use `RAPTOR_NO_PROXY` to route matching destination hosts directly. An exact
entry matches only that host. For example, `*.example.com` matches
`api.example.com` but not `example.com`; list both forms to bypass both.
Bypassed hosts use local DNS.

Raptor ignores ambient proxy variables and `NO_PROXY`. Managed shell commands
and custom chat-provider implementations use their own network configuration
and are outside this routing guarantee. Run `raptor --check-proxy` to send a
bounded request through the proxy to `api.ipify.org`. The command ignores
`RAPTOR_NO_PROXY` and prints the observed public egress IP without displaying
the configured proxy address.

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
| `TELEGRAM_SUBAGENT_TOPICS_SILENT` | `1` | Send subagent-topic messages without notification sound |

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

Store outbound model configuration in `.raptor/config.toml` or in the file
selected by `RAPTOR_CONFIG`. Use environment variables for the secrets named
by `api_key_env`. Raptor does not persist secret values in transcripts or
state.

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

Each model provider must expose Responses-compatible `/responses` and
`/models` endpoints. Per-model tables override provider defaults. If you omit
the top-level `model`, Raptor uses the selected provider's `default_model`. If
no configuration file exists, Raptor probes the local provider and selects the
first model that it serves.

The `/model` command archives the current root transcript and starts a new one.
As a result, provider-private response items never cross model-provider
boundaries.

### Subagents

By default, a subagent inherits its parent's model target. An explicit model
provider or model selects a different configured target for the new child. The
child retains that target for every continuation.

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

Raptor requires Python 3.11 or later and a Responses-compatible model backend.
The project metadata declares the Python dependency for `uv`. Custom
`module:attribute` chat providers require a source checkout. Frozen releases
include the built-in Telegram and Responses API providers.

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

Preserve these core invariants when you make a change: one root controller per
main chat, append-only owner-tagged conversation history, provider-affine
delivery, bounded retained state, atomic process ownership, and explicit
recovery after transient failure.

Every push to `main` runs tests and publishes a rolling `nightly` build
identified by its commit SHA. Human-selected tags publish releases
independently of the development version in `pyproject.toml`. Supported tags
include stable `vMAJOR.MINOR.PATCH` versions and numbered `alpha`, `beta`, or
`rc` prereleases. GitHub Actions uses the same test and freeze workflow for
nightly and tagged builds. Prerelease tags create GitHub prereleases. Stable
tags create standard releases that the default installer uses.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the design, implementation, testing,
and review requirements.

## License

Raptor is licensed under the [MIT License](LICENSE).
