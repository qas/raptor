"""Durable chat registry and context-bound runtime state."""

import asyncio
import copy
import json
import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from chat_provider import ConversationId
from chat_store import (
    append_item,
    create_session,
    ensure_chat_dirs,
    event_at_seq,
    next_event_seq,
    repair_all_chat_files,
    repair_chat_file,
    session_chat_key,
    session_exists,
    session_is_ended,
    steer_is_recorded,
)
from config import (
    MAX_CHAT_RUNTIMES,
    MAX_PENDING_STEERS,
    MAX_STATE_LOAD_BYTES,
    MAX_SUBAGENT_PENDING_INPUTS,
    MAX_SUBAGENT_RECORDS,
    RESPONSES_MODEL,
    STATE_PATH,
)
from runtime_events import RuntimeEvent
from storage import write_text_atomic
from todos import validate_plan

STATE_SCHEMA_VERSION = 2
_RECOVERY_BYTES_PER_CHAT = 512

GLOBAL_DEFAULT_STATE: dict[str, Any] = {
    "schema_version": STATE_SCHEMA_VERSION,
    "model": None,
    "runtime": {},
    "chats": {},
}

CHAT_DEFAULT_STATE: dict[str, Any] = {
    "current_session_id": None,
    "todos": [],
    "approval_mode": "off",
    "pending_inputs": [],
    "pending_delivery": None,
    "active_root_turn": None,
    "interrupted_subagents": [],
    "subagents": {},
    "goal": None,
    "thread": None,
}


class StateCapacityError(RuntimeError):
    """Raised when a durable state update would exceed its load bound."""

# Logical state exposed to domain code and focused tests.
DEFAULT_STATE: dict[str, Any] = {
    "model": None,
    **copy.deepcopy(CHAT_DEFAULT_STATE),
    "runtime": {},
}


def _new_turn_coordinator() -> Any:
    from turn_runtime import TurnCoordinator

    return TurnCoordinator()


@dataclass
class ChatRuntime:
    """All mutable state owned by one interactive main-agent chat."""

    key: str
    conversation_id: ConversationId
    state: dict[str, Any]
    steer_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=asyncio.Queue
    )
    pending_steers: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_event_queue: asyncio.Queue[RuntimeEvent] = field(
        default_factory=asyncio.Queue
    )
    subagent_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    pending_approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    turns: Any = field(default_factory=_new_turn_coordinator)
    goal_pin_message_id: int | str | None = None
    goal_pin_goal_id: str | None = None
    pinned_status_conversation_id: int | str | None = None
    pinned_status_message_id: int | str | None = None
    pinned_status_owner: str | None = None
    goal_creation_authorized: bool = False
    presentation_lock: asyncio.Lock | None = None
    presentation_loop: asyncio.AbstractEventLoop | None = None

    @property
    def subagent_records(self) -> dict[str, dict[str, Any]]:
        return self.state["subagents"]


_root_state: dict[str, Any]
_runtimes: dict[str, ChatRuntime] = {}
_default_runtime_key: str | None = None
_current_runtime: ContextVar[ChatRuntime | None] = ContextVar(
    "raptor_chat_runtime",
    default=None,
)


def conversation_key(conversation_id: ConversationId) -> str:
    """Return the stable persisted key for a provider conversation."""
    value = str(conversation_id)
    if not value:
        raise ValueError("conversation ID must not be empty")
    return value


def _load_plan(value: Any, owner: str) -> list[dict[str, str]]:
    try:
        return validate_plan(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid persisted {owner} plan: {exc}") from exc


def _prune_subagent_mapping(
    records: dict[str, Any],
    interrupted: list[Any],
) -> int:
    protected_ids = {
        str(item.get("id"))
        for item in interrupted
        if isinstance(item, dict) and item.get("id") is not None
    }
    protected_ids.update(
        str(agent_id)
        for agent_id, record in records.items()
        if isinstance(record, dict)
        and (
            record.get("status") == "running"
            or record.get("completion_pending")
            or record.get("activity_surface_id")
        )
    )
    removable = sorted(
        (
            (str(agent_id), record)
            for agent_id, record in records.items()
            if str(agent_id) not in protected_ids and isinstance(record, dict)
        ),
        key=lambda pair: (
            float(pair[1].get("completed_at") or 0),
            float(pair[1].get("started_at") or 0),
        ),
        reverse=True,
    )
    remove_ids = {
        agent_id
        for agent_id, _record in removable[MAX_SUBAGENT_RECORDS:]
    }
    for agent_id in remove_ids:
        records.pop(agent_id, None)
    return len(remove_ids)


def bounded_interrupted_subagents(items: list[Any]) -> list[dict[str, Any]]:
    """Retain the newest unique interrupted-subagent checkpoints."""
    newest_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        agent_id = str(item["id"])
        current = newest_by_id.get(agent_id)
        if current is None or float(item.get("interrupted_at") or 0) >= float(
            current.get("interrupted_at") or 0
        ):
            newest_by_id[agent_id] = item
    return sorted(
        newest_by_id.values(),
        key=lambda item: float(item.get("interrupted_at") or 0),
        reverse=True,
    )[:MAX_SUBAGENT_RECORDS]


def _normalize_chat_state(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Persisted chat state must be an object: {owner}")
    result = copy.deepcopy(CHAT_DEFAULT_STATE)
    for key in CHAT_DEFAULT_STATE:
        if key in value:
            result[key] = value[key]

    if not isinstance(result.get("subagents"), dict):
        raise RuntimeError(f"Persisted subagents must be an object: {owner}")
    interrupted = result.get("interrupted_subagents")
    result["interrupted_subagents"] = bounded_interrupted_subagents(
        interrupted if isinstance(interrupted, list) else []
    )
    pending = result.get("pending_inputs")
    pending_items = (
        pending[:MAX_PENDING_STEERS] if isinstance(pending, list) else []
    )
    result["pending_inputs"] = [
        {
            "id": str(item["id"]),
            "text": str(item["text"]),
        }
        for item in pending_items
        if isinstance(item, dict) and item.get("id") and item.get("text")
    ]
    pending_delivery = result.get("pending_delivery")
    if pending_delivery is not None:
        if not isinstance(pending_delivery, dict):
            raise RuntimeError(
                f"Persisted pending delivery must be an object: {owner}"
            )
        session_id = str(pending_delivery.get("session_id") or "")
        seq = pending_delivery.get("seq")
        try:
            delivery_owned = (
                bool(session_id)
                and session_exists(session_id)
                and session_chat_key(session_id) == owner
            )
        except ValueError:
            delivery_owned = False
        if (
            not delivery_owned
            or not isinstance(seq, int)
            or seq <= 0
        ):
            raise RuntimeError(
                f"Invalid persisted pending delivery: {owner}"
            )
        repair_chat_file(session_id)
        delivery_event = event_at_seq(session_id, seq)
        delivery_item = (
            delivery_event.get("item")
            if isinstance(delivery_event, dict)
            else None
        )
        prepared = (
            delivery_event is None
            and seq == next_event_seq(session_id)
        )
        recorded = (
            isinstance(delivery_event, dict)
            and delivery_event.get("source") == "assistant"
            and isinstance(delivery_item, dict)
            and delivery_item.get("type") == "message"
        )
        if not prepared and not recorded:
            raise RuntimeError(
                f"Pending delivery does not reference an answer: {owner}"
            )
        result["pending_delivery"] = {
            "session_id": session_id,
            "seq": seq,
        }
    result["todos"] = _load_plan(result.get("todos"), f"{owner} root")

    goal = result.get("goal")
    if goal is not None and not isinstance(goal, dict):
        raise RuntimeError(f"Persisted goal must be an object: {owner}")
    if isinstance(goal, dict):
        goal.setdefault("blocked_reason", None)
        goal.setdefault("notified_status", None)
        goal["todos"] = _load_plan(goal.get("todos"), f"{owner} goal")

    thread = result.get("thread")
    if thread is not None and not isinstance(thread, dict):
        raise RuntimeError(f"Persisted thread must be an object: {owner}")
    if isinstance(thread, dict):
        parent_id = str(thread.get("parent_session_id") or "")
        branch_id = str(thread.get("session_id") or "")
        parent_owned = (
            session_exists(parent_id) and session_chat_key(parent_id) == owner
        )
        branch_owned = (
            session_exists(branch_id) and session_chat_key(branch_id) == owner
        )
        if not parent_owned or not branch_owned:
            parent_interrupted = thread.get("parent_interrupted_subagents")
            result["thread"] = None
            if parent_owned:
                result["current_session_id"] = parent_id
                result["interrupted_subagents"] = (
                    parent_interrupted
                    if isinstance(parent_interrupted, list)
                    else []
                )
            else:
                result["current_session_id"] = None
                result["interrupted_subagents"] = []
        else:
            result["current_session_id"] = branch_id

    for record in result["subagents"].values():
        if not isinstance(record, dict):
            raise RuntimeError(f"Persisted subagent must be an object: {owner}")
        if str(record.get("chat_key") or "") != owner:
            raise RuntimeError(
                f"Persisted subagent belongs to another chat: {owner}"
            )
        chat_id = record.get("chat_id")
        if chat_id is None or conversation_key(chat_id) != owner:
            raise RuntimeError(
                f"Persisted subagent conversation does not match: {owner}"
            )
        record["todos"] = _load_plan(
            record.get("todos"),
            f"{owner} subagent {record.get('id') or 'unknown'}",
        )
        subagent_pending = record.get("pending_inputs")
        record["pending_inputs"] = (
            [
                str(item)
                for item in subagent_pending[:MAX_SUBAGENT_PENDING_INPUTS]
                if str(item)
            ]
            if isinstance(subagent_pending, list)
            else []
        )
        record.setdefault("recovery_context", None)
        record.setdefault("completion_pending", False)
        record.setdefault("completion_notified_at", None)
        record.setdefault("completion_attempts", 0)
        record.setdefault("parent_session_id", None)
        record.setdefault("root_session_id", record.get("parent_session_id"))
        record["run_generation"] = max(
            1,
            int(record.get("run_generation") or 1),
        )
        record.setdefault("activity_surface_id", None)
        record["activity_finished_generation"] = max(
            0,
            int(record.get("activity_finished_generation") or 0),
        )
        record.setdefault("activity_result_delivered", False)
        record["task_count"] = max(1, int(record.get("task_count") or 1))
        if record.get("status") == "running":
            record["status"] = "interrupted"
            record["error"] = "Process exited while subagent was running"
            record["completed_at"] = int(time.time())
            checkpoint = {
                "id": record.get("id"),
                "session_id": record.get("session_id"),
                "interrupted_at": time.time(),
                "tool_events": list(record.get("tool_events") or []),
            }
            result["interrupted_subagents"] = [
                item
                for item in result["interrupted_subagents"]
                if item.get("id") != checkpoint.get("id")
            ]
            result["interrupted_subagents"].append(checkpoint)

    result["interrupted_subagents"] = bounded_interrupted_subagents(
        result["interrupted_subagents"]
    )
    _prune_subagent_mapping(
        result["subagents"],
        result["interrupted_subagents"],
    )
    if result.get("approval_mode") not in {"on", "off"}:
        raise RuntimeError(f"Invalid persisted approval mode: {owner}")
    current_session_id = str(result.get("current_session_id") or "")
    if (
        current_session_id
        and session_exists(current_session_id)
        and session_chat_key(current_session_id) != owner
    ):
        raise RuntimeError(
            f"Persisted session belongs to another chat: {owner}"
        )
    for record in result["subagents"].values():
        child_session_id = str(record.get("session_id") or "")
        if (
            child_session_id
            and session_exists(child_session_id)
            and session_chat_key(child_session_id) != owner
        ):
            raise RuntimeError(
                f"Persisted subagent session belongs to another chat: {owner}"
            )
    return result


def load_state() -> dict[str, Any]:
    result = copy.deepcopy(GLOBAL_DEFAULT_STATE)
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("rb") as handle:
                encoded = handle.read(MAX_STATE_LOAD_BYTES + 1)
        except OSError as exc:
            raise RuntimeError(f"Could not load state: {STATE_PATH}") from exc
        if len(encoded) > MAX_STATE_LOAD_BYTES:
            raise RuntimeError(
                f"State file exceeds MAX_STATE_LOAD_BYTES: {STATE_PATH}"
            )
        try:
            loaded = json.loads(encoded)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load state: {STATE_PATH}") from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(f"State root must be an object: {STATE_PATH}")
        if loaded.get("schema_version") != STATE_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported Raptor state schema; start with a fresh "
                f"RAPTOR_HOME (expected {STATE_SCHEMA_VERSION})"
            )
        result["model"] = loaded.get("model")
        runtime = loaded.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("Persisted runtime metadata must be an object")
        result["runtime"] = runtime
        chats = loaded.get("chats")
        if not isinstance(chats, dict):
            raise RuntimeError("Persisted chats must be an object")
        if len(chats) > MAX_CHAT_RUNTIMES:
            raise RuntimeError(
                "Persisted chats exceed MAX_CHAT_RUNTIMES"
            )
        normalized_chats: dict[str, Any] = {}
        for key, entry in chats.items():
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Persisted chat entry must be an object: {key}"
                )
            conversation_id = entry.get("conversation_id")
            if conversation_id is None:
                raise RuntimeError(
                    f"Persisted chat has no conversation ID: {key}"
                )
            normalized_chats[str(key)] = {
                "conversation_id": conversation_id,
                "state": _normalize_chat_state(entry.get("state"), str(key)),
            }
        result["chats"] = normalized_chats
    if RESPONSES_MODEL:
        result["model"] = RESPONSES_MODEL
    return result


def _register_loaded_chats() -> None:
    for key, entry in _root_state["chats"].items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"Persisted chat entry must be an object: {key}")
        conversation_id = entry.get("conversation_id")
        if conversation_id is None:
            raise RuntimeError(f"Persisted chat has no conversation ID: {key}")
        normalized = entry["state"]
        _runtimes[str(key)] = ChatRuntime(
            key=str(key),
            conversation_id=conversation_id,
            state=normalized,
        )


_root_state = load_state()
_register_loaded_chats()


def _state_recovery_reserve(root_state: dict[str, Any]) -> int:
    return len(root_state.get("chats", {})) * _RECOVERY_BYTES_PER_CHAT


def _state_admission_limit(root_state: dict[str, Any]) -> int:
    return MAX_STATE_LOAD_BYTES - _state_recovery_reserve(root_state)


def _state_without_recovery_markers(
    root_state: dict[str, Any],
) -> dict[str, Any]:
    chats: dict[str, Any] = {}
    for key, entry in root_state.get("chats", {}).items():
        chat_state = dict(entry.get("state", {}))
        chat_state["active_root_turn"] = None
        chat_state["pending_delivery"] = None
        chats[str(key)] = {**entry, "state": chat_state}
    return {**root_state, "chats": chats}


def _write_root_state(root_state: dict[str, Any]) -> None:
    admitted = json.dumps(
        _state_without_recovery_markers(root_state),
        indent=2,
        ensure_ascii=False,
    )
    admitted_bytes = len(admitted.encode("utf-8"))
    reserve = _state_recovery_reserve(root_state)
    if admitted_bytes > _state_admission_limit(root_state):
        raise StateCapacityError(
            "State update would consume reserved recovery capacity"
        )
    serialized = json.dumps(root_state, indent=2, ensure_ascii=False)
    serialized_bytes = len(serialized.encode("utf-8"))
    if serialized_bytes - admitted_bytes > reserve:
        raise StateCapacityError("Recovery markers exceed reserved capacity")
    if serialized_bytes > MAX_STATE_LOAD_BYTES:
        raise StateCapacityError(
            "State update exceeds MAX_STATE_LOAD_BYTES"
        )
    write_text_atomic(
        STATE_PATH,
        serialized,
        mode=0o600,
    )


def _write_state() -> None:
    _write_root_state(_root_state)


def _persist_chat_candidate(
    candidate_state: dict[str, Any],
) -> None:
    """Commit one chat-state replacement only after its write succeeds."""
    runtime = current_runtime()
    current_entry = _root_state["chats"][runtime.key]
    candidate_entry = {
        **current_entry,
        "state": candidate_state,
    }
    candidate_root = {
        **_root_state,
        "chats": {
            **_root_state["chats"],
            runtime.key: candidate_entry,
        },
    }
    _write_root_state(candidate_root)
    runtime.state.clear()
    runtime.state.update(candidate_state)


def queue_pending_steer(steer_id: str, text: str) -> None:
    """Durably admit one steer without mutating live state on rejection."""
    candidate = copy.deepcopy(current_runtime().state)
    pending = candidate.setdefault("pending_inputs", [])
    if not isinstance(pending, list):
        raise RuntimeError("Pending steering state is invalid")
    if len(pending) >= MAX_PENDING_STEERS:
        raise StateCapacityError("Steering queue is full")
    pending.append({"id": str(steer_id), "text": str(text)})
    _persist_chat_candidate(candidate)


def set_pending_delivery(session_id: str, seq: int) -> None:
    """Persist the transcript reference awaiting provider delivery."""
    candidate = copy.deepcopy(current_runtime().state)
    reference = {"session_id": str(session_id), "seq": int(seq)}
    existing = candidate.get("pending_delivery")
    if existing is not None and existing != reference:
        raise RuntimeError("A response is already awaiting delivery")
    candidate["pending_delivery"] = reference
    _persist_chat_candidate(candidate)


def clear_pending_delivery(session_id: str, seq: int) -> None:
    """Clear a delivered transcript reference if it is still current."""
    expected = {"session_id": str(session_id), "seq": int(seq)}
    if current_runtime().state.get("pending_delivery") != expected:
        return
    candidate = copy.deepcopy(current_runtime().state)
    candidate["pending_delivery"] = None
    _persist_chat_candidate(candidate)


def set_active_root_turn(marker: dict[str, Any]) -> None:
    """Persist the bounded crash-recovery marker from reserved capacity."""
    candidate = copy.deepcopy(current_runtime().state)
    candidate["active_root_turn"] = copy.deepcopy(marker)
    _persist_chat_candidate(candidate)


def ensure_chat(conversation_id: ConversationId) -> ChatRuntime:
    """Return the runtime owned by a provider conversation, creating it."""
    key = conversation_key(conversation_id)
    runtime = _runtimes.get(key)
    if runtime is not None:
        if runtime.conversation_id != conversation_id:
            raise RuntimeError(f"Conversation key collision: {key}")
        return runtime
    if len(_runtimes) >= MAX_CHAT_RUNTIMES:
        raise RuntimeError("Chat runtime capacity reached")
    chat_state = copy.deepcopy(CHAT_DEFAULT_STATE)
    ensure_chat_dirs()
    chat_state["current_session_id"] = create_session(
        kind="main",
        chat_key=key,
    )
    runtime = ChatRuntime(
        key=key,
        conversation_id=conversation_id,
        state=chat_state,
    )
    entry = {
        "conversation_id": conversation_id,
        "state": chat_state,
    }
    candidate = dict(_root_state)
    candidate["chats"] = {
        **_root_state["chats"],
        key: entry,
    }
    _write_root_state(candidate)
    _root_state["chats"][key] = entry
    _runtimes[key] = runtime
    global _default_runtime_key
    if _default_runtime_key is None:
        _default_runtime_key = key
    return runtime


def current_runtime() -> ChatRuntime:
    runtime = _current_runtime.get()
    if runtime is not None:
        return runtime
    if _default_runtime_key is not None:
        default_runtime = _runtimes.get(_default_runtime_key)
        if default_runtime is not None:
            return default_runtime
    raise RuntimeError("No chat runtime is bound to this task")


def set_default_chat(conversation_id: ConversationId) -> ChatRuntime:
    """Set the runtime used by lifecycle work outside an event context."""
    runtime = ensure_chat(conversation_id)
    global _default_runtime_key
    _default_runtime_key = runtime.key
    return runtime


@contextmanager
def bound_chat(conversation_id: ConversationId) -> Iterator[ChatRuntime]:
    runtime = ensure_chat(conversation_id)
    token = _current_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _current_runtime.reset(token)


@contextmanager
def bound_runtime(runtime: ChatRuntime) -> Iterator[ChatRuntime]:
    token = _current_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _current_runtime.reset(token)


def all_chat_runtimes() -> tuple[ChatRuntime, ...]:
    return tuple(_runtimes.values())


GLOBAL_STATE_KEYS = frozenset({"model", "runtime"})


class StateView(MutableMapping[str, Any]):
    """Mutable mapping over global state and the context-bound chat state."""

    def _chat(self) -> dict[str, Any]:
        return current_runtime().state

    def __getitem__(self, key: str) -> Any:
        if key in GLOBAL_STATE_KEYS:
            return _root_state[key]
        return self._chat()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in GLOBAL_STATE_KEYS:
            _root_state[key] = value
        else:
            self._chat()[key] = value

    def __delitem__(self, key: str) -> None:
        if key in GLOBAL_STATE_KEYS:
            del _root_state[key]
        else:
            del self._chat()[key]

    def __iter__(self) -> Iterator[str]:
        yield from GLOBAL_STATE_KEYS
        yield from self._chat()

    def __len__(self) -> int:
        return len(GLOBAL_STATE_KEYS) + len(self._chat())

    def clear(self) -> None:
        runtime = current_runtime()
        if runtime.turns.is_running() or runtime.subagent_tasks:
            raise RuntimeError("Cannot clear a chat runtime while work is active")
        _root_state["model"] = None
        _root_state["runtime"] = {}
        runtime.state.clear()
        runtime.state.update(copy.deepcopy(CHAT_DEFAULT_STATE))
        runtime.pending_steers.clear()
        runtime.pending_approvals.clear()
        runtime.steer_queue = asyncio.Queue()
        runtime.runtime_event_queue = asyncio.Queue()
        runtime.goal_pin_message_id = None
        runtime.goal_pin_goal_id = None
        runtime.pinned_status_conversation_id = None
        runtime.pinned_status_message_id = None
        runtime.pinned_status_owner = None
        runtime.goal_creation_authorized = False
        runtime.presentation_lock = None
        runtime.presentation_loop = None
        runtime.turns.finish()


state: MutableMapping[str, Any] = StateView()


T = TypeVar("T")


class ContextMapping(MutableMapping[str, T], Generic[T]):
    def __init__(self, attribute: str) -> None:
        self.attribute = attribute

    def _target(self) -> dict[str, T]:
        return getattr(current_runtime(), self.attribute)

    def __getitem__(self, key: str) -> T:
        return self._target()[key]

    def __setitem__(self, key: str, value: T) -> None:
        self._target()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._target()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._target())

    def __len__(self) -> int:
        return len(self._target())


class ContextQueue(Generic[T]):
    def __init__(self, attribute: str) -> None:
        self.attribute = attribute

    def _target(self) -> asyncio.Queue[T]:
        return getattr(current_runtime(), self.attribute)

    async def put(self, item: T) -> None:
        await self._target().put(item)

    def put_nowait(self, item: T) -> None:
        self._target().put_nowait(item)

    async def get(self) -> T:
        return await self._target().get()

    def get_nowait(self) -> T:
        return self._target().get_nowait()

    def task_done(self) -> None:
        self._target().task_done()

    async def join(self) -> None:
        await self._target().join()

    def empty(self) -> bool:
        return self._target().empty()

    def qsize(self) -> int:
        return self._target().qsize()


steer_queue: ContextQueue[dict[str, Any]] = ContextQueue("steer_queue")
pending_steers: ContextMapping[dict[str, Any]] = ContextMapping(
    "pending_steers"
)
runtime_event_queue: ContextQueue[RuntimeEvent] = ContextQueue(
    "runtime_event_queue"
)
subagent_tasks: ContextMapping[asyncio.Task[Any]] = ContextMapping(
    "subagent_tasks"
)
subagent_records: ContextMapping[dict[str, Any]] = ContextMapping(
    "subagent_records"
)
pending_approvals: ContextMapping[dict[str, Any]] = ContextMapping(
    "pending_approvals"
)

APPROVAL_TOOLS = {"shell", "write_file", "edit_file"}

DAEMON_MODE = False
responses: Any


def save_state() -> None:
    _write_state()


def consume_pending_steer(entry: dict[str, Any]) -> None:
    """Retire one accepted steer from runtime and durable pending state."""
    entry["status"] = "applied"
    pending_steers.pop(str(entry.get("id") or ""), None)
    pending = state.get("pending_inputs")
    if isinstance(pending, list):
        steer_id = str(entry.get("id") or "")
        state["pending_inputs"] = [
            item
            for item in pending
            if not isinstance(item, dict) or str(item.get("id") or "") != steer_id
        ]
    save_state()


def persist_steer_handoff(entry: dict[str, Any]) -> None:
    """Transfer one accepted steer to the transcript before retiring it."""
    session_id = str(state.get("current_session_id") or "")
    if not session_id:
        raise RuntimeError("No current session")
    steer_id = str(entry.get("id") or "")
    text = str(entry.get("text") or "")
    if not steer_id or not text:
        raise ValueError("invalid pending steer")
    if not steer_is_recorded(session_id, steer_id):
        append_item(
            session_id,
            {"role": "user", "content": text},
            source="steer",
            data={"steer_id": steer_id},
        )
    consume_pending_steer(entry)


def prune_subagent_records() -> int:
    runtime = current_runtime()
    return _prune_subagent_mapping(
        runtime.subagent_records,
        runtime.state.get("interrupted_subagents", []),
    )


def rehydrate_pending_inputs(chat_id: ConversationId) -> int:
    """Restore one chat's persisted steering texts into its runtime queue."""
    runtime = ensure_chat(chat_id)
    pending = runtime.state.get("pending_inputs")
    if not isinstance(pending, list) or not pending:
        return 0
    count = 0
    with bound_runtime(runtime):
        for item in pending:
            if not isinstance(item, dict):
                continue
            steer_id = str(item.get("id") or "")
            content = str(item.get("text") or "")
            if not steer_id or not content:
                continue
            entry = {
                "id": steer_id,
                "chat_id": chat_id,
                "text": content,
                "message_id": None,
                "status": "queued",
                "rehydrated": True,
            }
            runtime.pending_steers[steer_id] = entry
            runtime.steer_queue.put_nowait(entry)
            count += 1
    return count


def bootstrap_runtime_storage() -> dict[str, int]:
    """Repair transcripts and ensure every registered chat has a session."""
    repaired = repair_all_chat_files()
    created = 0
    state_changed = False
    for runtime in all_chat_runtimes():
        session_id = runtime.state.get("current_session_id")
        session_valid = bool(session_id and session_exists(str(session_id)))
        ended = bool(session_valid and session_is_ended(str(session_id)))
        thread = runtime.state.get("thread")
        if (not session_valid or ended) and isinstance(thread, dict):
            parent_id = str(thread.get("parent_session_id") or "")
            branch_id = str(thread.get("session_id") or "")
            parent_valid = bool(
                parent_id
                and session_exists(parent_id)
                and not session_is_ended(parent_id)
            )
            if str(session_id) == branch_id and parent_valid:
                runtime.state["thread"] = None
                runtime.state["current_session_id"] = parent_id
                runtime.state["interrupted_subagents"] = copy.deepcopy(
                    thread.get("parent_interrupted_subagents") or []
                )
                state_changed = True
                session_id = parent_id
                session_valid = True
                ended = False
            else:
                runtime.state["thread"] = None
                runtime.state["interrupted_subagents"] = []
                state_changed = True
        if (
            not session_id
            or not session_exists(str(session_id))
            or ended
        ):
            runtime.state["current_session_id"] = create_session(
                kind="main",
                chat_key=runtime.key,
            )
            created += 1
            state_changed = True
    if state_changed:
        _write_state()
    return {
        "created_sessions": created,
        "repaired_chats": repaired,
    }
