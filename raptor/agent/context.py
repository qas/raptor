"""Active context building and checkpoint compaction."""
import copy
from collections.abc import Awaitable, Callable
from typing import Any

from raptor.observability import log_event

from raptor.state.chat_store import (
    active_checkpoint,
    active_item_events,
    active_projection,
    append_checkpoint,
    append_meta,
    render_compaction_records,
)
from raptor.config import (
    COMPACT_KEEP_RECENT_TOKENS,
    COMPACTION_MAX_RECORD_CHARS,
    COMPACTION_OUTPUT_TOKENS,
    COMPACTION_GENERATION_TOKENS,
    COMPACTION_USER_ANCHOR_TOKENS,
)
from raptor.agent.engine import estimate_tokens, response_text
from raptor.model.response_errors import ContextLengthError, TransientResponsesError

COMPACTION_INSTRUCTIONS = """
Create a precise handoff checkpoint for another agent continuing this session.

Use these sections in order:
1. Active objective and current focus
2. User requirements, constraints, corrections, and preferences
3. Decisions and reasoning already settled
4. Files, identifiers, commands, environment facts, and resulting state
5. Work completed and how it was verified
6. Failures, warnings, and unresolved risks
7. Remaining work and the exact next action

Preserve exact names, paths, values, and user wording whenever they affect the
result. Treat preserved user-request records as authoritative. Distinguish
completed work from proposed work. Do not invent facts or claim verification
that did not happen. Drop chatter, duplicated logs, and obsolete intermediate
detail.

The complete lossless transcript is archived and can be searched later, so
do not try to reproduce every old message verbatim.

Output only the checkpoint.
""".strip()

COMPACTION_RETRY_INSTRUCTIONS = """
The previous compaction attempt completed without any visible checkpoint.
Do not continue internal analysis. Emit the checkpoint as the final answer now.
""".strip()

CHECKPOINT_CONTINUATION_INPUT = {
    "role": "user",
    "content": (
        "Continue the in-progress turn from the checkpoint. Treat anything "
        "the checkpoint records as completed or already communicated as "
        "done; do not repeat it. Resume only unresolved actions, and give "
        "the user-facing final response after the remaining work is complete."
    ),
}

EstimateCompactionRequest = Callable[
    [list[dict[str, Any]], str],
    int,
]
CreateCompactionResponse = Callable[
    [list[dict[str, Any]], str],
    Awaitable[dict[str, Any]],
]

_CHECKPOINT_TRUNCATION_MARKER = (
    "\n\n[checkpoint middle truncated]\n\n"
)


def context_input_budget() -> int:
    """Default for provider-neutral callers; production agents pass a target budget."""
    return 0


def compaction_generation_budget() -> int:
    """Default for provider-neutral callers; agents pass a target allowance."""
    return COMPACTION_GENERATION_TOKENS


def checkpoint_summary_item(summary: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            "Raptor conversation checkpoint "
            "(historical context, not a new user request). "
            "The complete transcript remains available through "
            "the chat_history tool.\n\n"
            + summary
        ),
    }


def _checkpoint_anchors(
    checkpoint: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not checkpoint:
        return []
    anchors = checkpoint.get("anchors")
    if not isinstance(anchors, list):
        return []
    return [
        copy.deepcopy(anchor)
        for anchor in anchors
        if isinstance(anchor, dict)
        and isinstance(anchor.get("item"), dict)
    ]


def _checkpoint_bounded_tail(
    checkpoint: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]] | None:
    """Return the visible copy of the oldest truncated tail message."""
    if not checkpoint:
        return None
    tail = checkpoint.get("tail")
    if not isinstance(tail, dict) or not isinstance(tail.get("item"), dict):
        return None
    seq = int(tail.get("seq") or 0)
    if seq <= 0:
        return None
    return seq, tail["item"]


def _anchor_input_items(
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _chronological_input_item(anchor["item"])
        for anchor in anchors
        if isinstance(anchor.get("item"), dict)
    ]


def _chronological_input_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep instruction authority out of chronological model input."""
    result = copy.deepcopy(item)
    if result.get("role") in {"developer", "system"}:
        result["role"] = "user"
    return result


def checkpoint_continuation_input() -> list[dict[str, Any]]:
    return [copy.deepcopy(CHECKPOINT_CONTINUATION_INPUT)]


def build_active_context(
    session_id: str,
) -> list[dict[str, Any]]:
    projection = active_projection(session_id)
    items = projection.items
    checkpoint = projection.checkpoint
    if not checkpoint:
        return [
            _chronological_input_item(event["item"])
            for event in items
            if isinstance(event.get("item"), dict)
        ]
    through_seq = int(checkpoint.get("through_seq") or 0)
    bounded = _checkpoint_bounded_tail(checkpoint)
    active = _anchor_input_items(_checkpoint_anchors(checkpoint))
    active.append(
        checkpoint_summary_item(str(checkpoint.get("summary") or ""))
    )
    for event in items:
        seq = int(event.get("seq") or 0)
        if seq <= through_seq:
            continue
        item = event.get("item")
        if bounded is not None and seq == bounded[0]:
            item = bounded[1]
        if isinstance(item, dict):
            active.append(_chronological_input_item(item))
    return active


def _item_role(item: dict[str, Any]) -> str | None:
    if item.get("role") == "user":
        return "user"
    return None


def _is_user_item_event(event: dict[str, Any]) -> bool:
    item = event.get("item")
    return isinstance(item, dict) and _item_role(item) == "user"


def _is_user_anchor_event(event: dict[str, Any]) -> bool:
    if not _is_user_item_event(event) or event.get("source") == "internal":
        return False
    item = event.get("item") or {}
    content = item.get("content")
    if not isinstance(content, str):
        return True
    return not (
        content.startswith("Raptor conversation checkpoint ")
        or content == CHECKPOINT_CONTINUATION_INPUT["content"]
        or content.startswith("Continue working toward the active persistent goal.")
    )


_ANCHOR_TRUNCATION_MARKER = (
    "\n\n[historical user request middle truncated]\n\n"
)


def _truncate_user_anchor(
    item: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any] | None:
    if max_tokens <= 0:
        return None
    candidate = copy.deepcopy(item)
    if estimate_tokens(candidate) <= max_tokens:
        return candidate
    content = candidate.get("content")
    if not isinstance(content, str):
        content = str(content)
    marker = _ANCHOR_TRUNCATION_MARKER

    def _candidate(keep_chars: int) -> dict[str, Any]:
        prefix_chars = (keep_chars + 1) // 2
        suffix_chars = keep_chars // 2
        prefix = content[:prefix_chars]
        suffix = content[-suffix_chars:] if suffix_chars else ""
        bounded = copy.deepcopy(candidate)
        bounded["content"] = prefix + marker + suffix
        return bounded

    low = 0
    high = len(content)
    best: dict[str, Any] | None = None
    while low <= high:
        keep_chars = (low + high) // 2
        bounded = _candidate(keep_chars)
        if estimate_tokens(bounded) <= max_tokens:
            best = bounded
            low = keep_chars + 1
        else:
            high = keep_chars - 1
    return best


def _select_user_anchors(
    items: list[dict[str, Any]],
    *,
    through_seq: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    if max_tokens <= 0:
        return []
    candidates = [
        event
        for event in items
        if int(event.get("seq") or 0) <= through_seq
        and _is_user_anchor_event(event)
    ]
    selected: list[dict[str, Any]] = []
    used = 0
    for event in reversed(candidates):
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        remaining = max_tokens - used
        bounded = _truncate_user_anchor(item, remaining)
        if bounded is None:
            break
        selected.append(
            {
                "seq": int(event.get("seq") or 0),
                "item": bounded,
            }
        )
        used += estimate_tokens(bounded)
        if bounded != item:
            break
    selected.reverse()
    return selected


_TOOL_ITEM_TYPES = {"function_call", "function_call_output"}


def _complete_tool_spans(
    events: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Spans that contain both a call and a result for every included id."""
    calls: dict[str, list[int]] = {}
    outputs: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") not in _TOOL_ITEM_TYPES:
            continue
        call_id = str(item.get("call_id") or "")
        if not call_id:
            continue
        bucket = calls if item.get("type") == "function_call" else outputs
        bucket.setdefault(call_id, []).append(index)
    spans: list[tuple[int, int]] = []
    for call_id, call_indexes in calls.items():
        result_indexes = outputs.get(call_id)
        if not result_indexes:
            continue
        indexes = [*call_indexes, *result_indexes]
        spans.append((min(indexes), max(indexes)))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            previous = merged[-1]
            merged[-1] = (previous[0], max(previous[1], end))
        else:
            merged.append((start, end))
    return merged


def _coherent_groups(
    events: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Partition history so a complete call and its result stay together."""
    groups: list[list[dict[str, Any]]] = []
    cursor = 0
    for start, end in _complete_tool_spans(events):
        while cursor < start:
            groups.append([events[cursor]])
            cursor += 1
        groups.append(events[start:end + 1])
        cursor = end + 1
    while cursor < len(events):
        groups.append([events[cursor]])
        cursor += 1
    return groups


def _group_is_retainable(group: list[dict[str, Any]]) -> bool:
    """A native tool group needs a call and a result for every call id."""
    calls: dict[str, int] = {}
    outputs: dict[str, int] = {}
    saw_tool = False
    for event in group:
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") not in _TOOL_ITEM_TYPES:
            continue
        saw_tool = True
        call_id = str(item.get("call_id") or "")
        if not call_id:
            return False
        bucket = calls if item.get("type") == "function_call" else outputs
        bucket[call_id] = bucket.get(call_id, 0) + 1
    if not saw_tool:
        return True
    identities = set(calls) | set(outputs)
    return all(
        calls.get(call_id, 0) >= 1 and outputs.get(call_id, 0) >= 1
        for call_id in identities
    )


def _group_cost(group: list[dict[str, Any]]) -> int:
    total = 0
    for event in group:
        item = event.get("item")
        if isinstance(item, dict):
            total += estimate_tokens(item)
    return total


def _truncation_marker(removed: str) -> str:
    count = max(1, estimate_tokens(removed) if removed else 1)
    return f"…{count} tokens truncated…"


def _fit_truncated_text(
    text: str,
    max_tokens: int,
    build: Callable[[str], dict[str, Any]],
) -> dict[str, Any] | None:
    """Keep both ends of one message when the remaining tail budget is short."""
    def _candidate(keep_chars: int) -> dict[str, Any]:
        if keep_chars >= len(text):
            return build(text)
        prefix_chars = (keep_chars + 1) // 2
        suffix_chars = keep_chars // 2
        prefix = text[:prefix_chars]
        suffix = text[-suffix_chars:] if suffix_chars else ""
        removed = text[len(prefix):len(text) - len(suffix)]
        return build(prefix + _truncation_marker(removed) + suffix)
    if estimate_tokens(_candidate(0)) > max_tokens:
        return None
    low = 0
    high = len(text)
    best = _candidate(0)
    while low <= high:
        keep_chars = (low + high) // 2
        bounded = _candidate(keep_chars)
        if estimate_tokens(bounded) <= max_tokens:
            best = bounded
            low = keep_chars + 1
        else:
            high = keep_chars - 1
    if estimate_tokens(best) > max_tokens or best == build(text):
        return None
    return best


def _is_text_part(part: Any) -> bool:
    if isinstance(part, str):
        return True
    return isinstance(part, dict) and isinstance(part.get("text"), str)


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    return str(part.get("text") or "")


def _part_with_text(part: Any, text: str) -> Any:
    if isinstance(part, str):
        return text
    cloned = copy.deepcopy(part)
    cloned["text"] = text
    return cloned


def _item_with_content(item: dict[str, Any], content: Any) -> dict[str, Any]:
    cloned = copy.deepcopy(item)
    cloned["content"] = content
    return cloned


def _truncate_content_parts(
    item: dict[str, Any],
    parts: list[Any],
    budget: int,
) -> dict[str, Any] | None:
    """Middle-truncate the overflowing text part and keep parts that fit."""
    if not any(_is_text_part(part) for part in parts):
        return None
    kept: list[Any] = []
    for index, part in enumerate(parts):
        if not _is_text_part(part):
            trial = _item_with_content(item, [*kept, copy.deepcopy(part)])
            if estimate_tokens(trial) <= budget:
                kept.append(copy.deepcopy(part))
            continue
        text = _part_text(part)
        full_part = _part_with_text(part, text)
        full = _item_with_content(item, [*kept, full_part])
        if estimate_tokens(full) <= budget:
            kept.append(full_part)
            continue
        snapshot = list(kept)
        def _build(truncated: str) -> dict[str, Any]:
            return _item_with_content(
                item,
                [*snapshot, _part_with_text(part, truncated)],
            )
        bounded = _fit_truncated_text(text, budget, _build)
        if bounded is None:
            break
        content = bounded.get("content")
        if not isinstance(content, list) or not content:
            break
        kept.append(content[-1])
        for later in parts[index + 1:]:
            if _is_text_part(later):
                continue
            trial = _item_with_content(item, [*kept, copy.deepcopy(later)])
            if estimate_tokens(trial) <= budget:
                kept.append(copy.deepcopy(later))
        break
    if not kept:
        return None
    result = _item_with_content(item, kept)
    if estimate_tokens(result) > budget or result == item:
        return None
    return result


def _truncate_textual_group(
    group: list[dict[str, Any]],
    budget: int,
) -> tuple[int, dict[str, Any]] | None:
    """Middle-truncate one textual message. Tool groups stay whole."""
    if budget <= 0 or len(group) != 1:
        return None
    event = group[0]
    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") in _TOOL_ITEM_TYPES:
        return None
    content = item.get("content")
    bounded: dict[str, Any] | None = None
    if isinstance(content, str):
        def _build_string(text: str) -> dict[str, Any]:
            return _item_with_content(item, text)
        bounded = _fit_truncated_text(content, budget, _build_string)
    elif isinstance(content, list):
        bounded = _truncate_content_parts(item, content, budget)
    if bounded is None:
        return None
    return int(event.get("seq") or 0), bounded


def _select_recent_tail(
    events: list[dict[str, Any]],
    budget: int,
) -> tuple[list[dict[str, Any]], tuple[int, dict[str, Any]] | None]:
    """Walk coherent groups newest-first, then return them chronologically."""
    if budget <= 0 or not events:
        return [], None
    remaining = budget
    chosen: list[list[dict[str, Any]]] = []
    bounded: tuple[int, dict[str, Any]] | None = None
    for group in reversed(_coherent_groups(events)):
        if remaining <= 0 or not _group_is_retainable(group):
            break
        cost = _group_cost(group)
        if cost <= remaining:
            chosen.append(group)
            remaining -= cost
            continue
        truncated = _truncate_textual_group(group, remaining)
        if truncated is not None:
            chosen.append(group)
            bounded = truncated
        break
    chosen.reverse()
    retained = [event for group in chosen for event in group]
    return retained, bounded


def _compaction_user_input(rendered: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": rendered,
        }
    ]


def _compaction_response_diagnostics(
    response: dict[str, Any],
) -> dict[str, Any]:
    """Return structural, bounded diagnostics without logging model content."""
    output = response.get("output")
    item_shapes: list[dict[str, Any]] = []
    if isinstance(output, list):
        for item in output[:16]:
            if not isinstance(item, dict):
                item_shapes.append({"type": type(item).__name__})
                continue
            content = item.get("content")
            content_types = (
                [
                    str(part.get("type") or "unknown")
                    for part in content[:16]
                    if isinstance(part, dict)
                ]
                if isinstance(content, list)
                else []
            )
            item_shapes.append(
                {
                    "type": item.get("type"),
                    "status": item.get("status"),
                    "content_types": content_types,
                }
            )

    usage = response.get("usage")
    usage_fields = {}
    if isinstance(usage, dict):
        usage_fields = {
            key: usage.get(key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if usage.get(key) is not None
        }
    incomplete = response.get("incomplete_details")
    incomplete_reason = (
        incomplete.get("reason")
        if isinstance(incomplete, dict)
        else None
    )
    return {
        "response_id": response.get("id"),
        "status": response.get("status"),
        "incomplete_reason": incomplete_reason,
        "output_items": item_shapes,
        "usage": usage_fields,
    }


def _truncate_checkpoint_summary(
    summary: str,
    max_tokens: int,
) -> tuple[str, bool]:
    """Bound a model-produced checkpoint while preserving both ends."""
    if estimate_tokens(summary) <= max_tokens:
        return summary, False
    marker = _CHECKPOINT_TRUNCATION_MARKER
    if estimate_tokens(marker) > max_tokens:
        return "", True

    def _candidate(keep_chars: int) -> str:
        prefix_chars = (keep_chars + 1) // 2
        suffix_chars = keep_chars // 2
        prefix = summary[:prefix_chars]
        suffix = summary[-suffix_chars:] if suffix_chars else ""
        return prefix + marker + suffix

    low = 0
    high = len(summary)
    best = marker
    while low <= high:
        keep_chars = (low + high) // 2
        candidate = _candidate(keep_chars)
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            low = keep_chars + 1
        else:
            high = keep_chars - 1
    return best, True


def _fit_checkpoint_to_active_budget(
    session_id: str,
    *,
    estimate_active_fn: Callable[[list[dict[str, Any]]], int],
    budget: int,
    include_continuation: bool,
) -> bool:
    """Shrink only the checkpoint text until the full request fits.

    This accounts for fixed instructions, tools, goals, and any retained
    native tail. It is a deterministic final bound after semantic model
    compaction, not a replacement for that compaction.
    """
    projection = active_projection(session_id)
    checkpoint = projection.checkpoint
    if not checkpoint:
        return False
    summary = str(checkpoint.get("summary") or "")
    anchors = _checkpoint_anchors(checkpoint)
    through_seq = int(checkpoint.get("through_seq") or 0)
    bounded = _checkpoint_bounded_tail(checkpoint)
    stored_tail = checkpoint.get("tail")
    tail_record = (
        copy.deepcopy(stored_tail)
        if isinstance(stored_tail, dict)
        else None
    )
    tail = []
    for event in projection.items:
        seq = int(event.get("seq") or 0)
        if seq <= through_seq:
            continue
        item = event.get("item")
        if bounded is not None and seq == bounded[0]:
            item = bounded[1]
        if isinstance(item, dict):
            tail.append(_chronological_input_item(item))

    def _work(
        candidate_summary: str,
        candidate_anchors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        work = [
            *_anchor_input_items(candidate_anchors),
            checkpoint_summary_item(candidate_summary),
            *tail,
        ]
        if include_continuation:
            work.extend(checkpoint_continuation_input())
        return work

    before = estimate_active_fn(_work(summary, anchors))
    if before < budget:
        return False

    marker = _CHECKPOINT_TRUNCATION_MARKER
    fitted_anchors = anchors
    while (
        fitted_anchors
        and estimate_active_fn(_work(marker, fitted_anchors)) >= budget
    ):
        fitted_anchors = fitted_anchors[1:]
    if estimate_active_fn(_work(marker, fitted_anchors)) >= budget:
        return False

    def _candidate(keep_chars: int) -> str:
        prefix_chars = (keep_chars + 1) // 2
        suffix_chars = keep_chars // 2
        prefix = summary[:prefix_chars]
        suffix = summary[-suffix_chars:] if suffix_chars else ""
        return prefix + marker + suffix

    low = 0
    high = len(summary)
    fitted = marker
    while low <= high:
        keep_chars = (low + high) // 2
        candidate = _candidate(keep_chars)
        if estimate_active_fn(_work(candidate, fitted_anchors)) < budget:
            fitted = candidate
            low = keep_chars + 1
        else:
            high = keep_chars - 1

    append_checkpoint(
        session_id,
        summary=fitted,
        through_seq=int(checkpoint.get("through_seq") or 0),
        input_from_seq=checkpoint.get("input_from_seq"),
        input_to_seq=checkpoint.get("input_to_seq"),
        reason="budget_fit",
        anchors=fitted_anchors,
        tail=tail_record,
    )
    append_meta(
        session_id,
        "checkpoint_budget_fit",
        {
            "before": before,
            "after": estimate_active_fn(_work(fitted, fitted_anchors)),
            "budget": budget,
            "anchors_before": len(anchors),
            "anchors_after": len(fitted_anchors),
        },
    )
    return True


def _fit_contiguous_records(
    records: list[dict[str, Any]],
    *,
    preserve_checkpoint: dict[str, Any] | None,
    estimate_fn: EstimateCompactionRequest,
    budget: int,
) -> list[dict[str, Any]]:
    """Select the largest chronological prefix of coherent groups."""
    selected: list[dict[str, Any]] = []
    start = 0
    if preserve_checkpoint is not None:
        selected.append(preserve_checkpoint)
        start = 1
    groups = _coherent_groups(records[start:])
    chosen: list[dict[str, Any]] = []
    for group in groups:
        candidate = [*records[:start], *chosen, *group]
        rendered = render_compaction_records(candidate)
        if estimate_fn(
            _compaction_user_input(rendered),
            COMPACTION_INSTRUCTIONS,
        ) <= budget:
            chosen.extend(group)
            continue
        break
    if chosen:
        return [*records[:start], *chosen]
    # Always make forward progress on a whole tool group. Individual record
    # rendering is bounded and the request path retries with a smaller view.
    if groups:
        return [*records[:start], *groups[0]]
    return selected


async def compact_session(
    session_id: str,
    *,
    estimate_compaction_request: EstimateCompactionRequest,
    create_compaction_response: CreateCompactionResponse,
    force: bool = False,
    reason: str = "threshold",
    input_budget: int | None = None,
    generation_budget: int | None = None,
) -> bool:
    projection = active_projection(session_id)
    items = projection.items
    previous = projection.checkpoint
    if not items and not previous:
        return False
    through_seq = int(previous.get("through_seq") or 0) if previous else 0
    candidates = [
        event
        for event in items
        if int(event.get("seq") or 0) > through_seq
    ]
    recompact_checkpoint = bool(
        previous
        and not candidates
        and reason == "overflow"
    )
    if not candidates and not recompact_checkpoint:
        return False
    keep_recent = 0 if force else COMPACT_KEEP_RECENT_TOKENS
    retained, bounded_tail = _select_recent_tail(candidates, keep_recent)
    retained_seqs = {
        int(event.get("seq") or 0) for event in retained
    }
    retired = [
        event
        for event in candidates
        if int(event.get("seq") or 0) not in retained_seqs
    ]
    if not retired and not recompact_checkpoint:
        return False
    records: list[dict[str, Any]] = []
    preserve: dict[str, Any] | None = None
    if previous:
        preserve = {
            "type": "checkpoint",
            "seq": previous.get("seq"),
            "summary": previous.get("summary"),
        }
        records.append(preserve)
    records.extend(retired)
    budget = (
        context_input_budget()
        if input_budget is None
        else max(0, input_budget)
    )
    generation_tokens = (
        compaction_generation_budget()
        if generation_budget is None
        else max(0, generation_budget)
    )
    compact_budget = (
        budget if budget else 12000
    )
    # Leave headroom for the full generation allowance (reasoning plus text),
    # not merely the smaller checkpoint text retained afterward.
    request_budget = max(
        1024,
        compact_budget - generation_tokens,
    )
    selected = _fit_contiguous_records(
        records,
        preserve_checkpoint=preserve,
        estimate_fn=estimate_compaction_request,
        budget=request_budget,
    )
    selected_native = [
        record for record in selected if record.get("type") == "item"
    ]
    if not selected or (retired and not selected_native):
        return False

    async def _summarize(
        selected: list[dict[str, Any]],
        *,
        max_record_chars: int | None = None,
    ) -> str:
        rendered = render_compaction_records(
            selected,
            max_record_chars=max_record_chars,
        )
        summary = ""
        for attempt in range(1, 3):
            instructions = COMPACTION_INSTRUCTIONS
            if attempt > 1:
                instructions += "\n\n" + COMPACTION_RETRY_INSTRUCTIONS
            response = await create_compaction_response(
                _compaction_user_input(rendered),
                instructions,
            )
            summary = response_text(response)
            if summary:
                break
            log_event(
                "context",
                "compaction_empty_response",
                {
                    "session_id": session_id,
                    "reason": reason,
                    "attempt": attempt,
                    **_compaction_response_diagnostics(response),
                    "selected_records": len(selected),
                    "rendered_chars": len(rendered),
                },
            )
        if not summary:
            raise TransientResponsesError(
                "compaction returned no summary after 2 attempts"
            )
        original_summary_tokens = estimate_tokens(summary)
        summary, truncated = _truncate_checkpoint_summary(
            summary,
            COMPACTION_OUTPUT_TOKENS,
        )
        if truncated:
            log_event(
                "context",
                "compaction_summary_truncated",
                {
                    "session_id": session_id,
                    "reason": reason,
                    "summary_tokens": original_summary_tokens,
                    "limit": COMPACTION_OUTPUT_TOKENS,
                },
            )
        if not summary:
            raise RuntimeError(
                "compaction summary token budget is too small"
            )
        return summary

    render_limit: int | None = None
    for context_attempt in range(4):
        try:
            summary = await _summarize(
                selected,
                max_record_chars=render_limit,
            )
            break
        except ContextLengthError:
            prefix_n = 1 if preserve and selected[0] is preserve else 0
            native = selected[prefix_n:]
            groups = _coherent_groups(native)
            if len(groups) > 1:
                keep_groups = max(1, len(groups) // 2)
                chosen: list[dict[str, Any]] = []
                for group in groups[:keep_groups]:
                    chosen.extend(group)
                selected = selected[:prefix_n] + chosen
                continue
            current_limit = (
                COMPACTION_MAX_RECORD_CHARS
                if render_limit is None
                else render_limit
            )
            if current_limit <= 1024:
                raise
            render_limit = max(1024, current_limit // 2)
    else:
        raise ContextLengthError("compaction input could not be fitted")
    input_seqs = [
        int(record.get("seq") or 0)
        for record in selected
        if record.get("type") in {"item", "checkpoint"}
        and record.get("seq") is not None
    ]
    item_seqs = [
        int(record.get("seq") or 0)
        for record in selected
        if record.get("type") == "item"
    ]
    through = (
        int(selected_native[-1].get("seq") or 0)
        if (selected_native := [
            record
            for record in selected
            if record.get("type") == "item"
        ])
        else through_seq
    )
    anchors = _select_user_anchors(
        [*_checkpoint_anchors(previous), *items],
        through_seq=through,
        max_tokens=COMPACTION_USER_ANCHOR_TOKENS,
    )
    tail = None
    if bounded_tail is not None and bounded_tail[0] > through:
        tail = {"seq": bounded_tail[0], "item": bounded_tail[1]}
    append_checkpoint(
        session_id,
        summary=summary,
        through_seq=through,
        input_from_seq=min(item_seqs) if item_seqs else None,
        input_to_seq=max(item_seqs) if item_seqs else through,
        reason=reason,
        anchors=anchors,
        tail=tail,
    )
    append_meta(
        session_id,
        "manual_compact" if reason == "manual" else "context_overflow"
        if reason == "overflow"
        else "threshold_compact",
        {
            "through_seq": through,
            "input_seqs": input_seqs,
            "anchors": len(anchors),
            "reason": reason,
        },
    )
    return True


async def ensure_context_under_budget(
    session_id: str,
    *,
    estimate_active_fn: Callable[[list[dict[str, Any]]], int],
    estimate_compaction_request: EstimateCompactionRequest,
    create_compaction_response: CreateCompactionResponse,
    reason: str = "threshold",
    force: bool = False,
    max_passes: int | None = None,
    include_continuation: bool = False,
    log_source: str = "context",
    input_budget: int | None = None,
    compaction_input_budget: int | None = None,
    generation_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Compact until rebuilt active context is under the input budget.

    Never returns work known to exceed the selected input budget when one is
    configured—raises ``RuntimeError`` instead. Agent callers pass the budget
    derived from their immutable provider/model target.
    """
    budget = (
        context_input_budget()
        if input_budget is None
        else max(0, input_budget)
    )
    compact_budget = (
        budget
        if compaction_input_budget is None
        else max(0, compaction_input_budget)
    )

    def _active_work() -> list[dict[str, Any]]:
        work = build_active_context(session_id)
        if include_continuation:
            work.extend(checkpoint_continuation_input())
        return work

    work = _active_work()
    estimate = estimate_active_fn(work)
    log_event(
        log_source,
        "checkpoint_before",
        {
            "active_estimate": estimate,
            "active_events": len(work),
            "reason": reason,
        },
    )
    if not budget:
        return work
    minimum_work = (
        checkpoint_continuation_input()
        if include_continuation
        else []
    )
    minimum_estimate = estimate_active_fn(minimum_work)
    if minimum_estimate >= budget:
        log_event(
            log_source,
            "fixed_context_over_budget",
            {
                "minimum_estimate": minimum_estimate,
                "budget": budget,
                "reason": reason,
            },
        )
        raise RuntimeError(
            "fixed instructions and tools exceed the input budget "
            f"(~{minimum_estimate:,} tokens; budget {budget:,})"
        )
    if estimate < budget and not force:
        return work
    pass_reason = reason
    pass_force = force
    # A fitted request may contain only one native record. By default allow
    # enough passes to cover every record once, plus checkpoint-only recovery,
    # while explicit callers can still impose a smaller operational limit.
    pass_limit = (
        max_passes
        if max_passes is not None
        else len(active_item_events(session_id)) + 2
    )
    for pass_n in range(pass_limit):
        if estimate < budget and not (force and pass_n == 0):
            break
        checkpoint_before = active_checkpoint(session_id)
        through_before = (
            int(checkpoint_before.get("through_seq") or 0)
            if checkpoint_before
            else 0
        )
        attempt_reason = "overflow" if pass_force else pass_reason
        ok = await compact_session(
            session_id,
            estimate_compaction_request=(
                estimate_compaction_request
            ),
            create_compaction_response=(
                create_compaction_response
            ),
            force=pass_force,
            reason=attempt_reason,
            input_budget=compact_budget,
            generation_budget=generation_budget,
        )
        if not ok:
            if not pass_force:
                log_event(
                    log_source,
                    "checkpoint_force_retry",
                    {
                        "active_estimate": estimate,
                        "active_events": len(work),
                        "reason": reason,
                    },
                )
                pass_force = True
                pass_reason = "overflow"
                continue
            if _fit_checkpoint_to_active_budget(
                session_id,
                estimate_active_fn=estimate_active_fn,
                budget=budget,
                include_continuation=include_continuation,
            ):
                work = _active_work()
                estimate = estimate_active_fn(work)
            break
        work = _active_work()
        estimate = estimate_active_fn(work)
        checkpoint = active_checkpoint(session_id)
        through_after = (
            int(checkpoint.get("through_seq") or 0)
            if checkpoint
            else 0
        )
        log_event(
            log_source,
            "checkpoint_after",
            {
                "checkpoint_through": (
                    checkpoint.get("through_seq")
                    if checkpoint
                    else None
                ),
                "active_estimate": estimate,
                "active_events": len(work),
                "reason": attempt_reason,
                "pass": pass_n + 1,
            },
        )
        if estimate < budget:
            return work
        remaining_native = any(
            int(event.get("seq") or 0) > through_after
            for event in active_item_events(session_id)
        )
        if not remaining_native and _fit_checkpoint_to_active_budget(
            session_id,
            estimate_active_fn=estimate_active_fn,
            budget=budget,
            include_continuation=include_continuation,
        ):
            work = _active_work()
            estimate = estimate_active_fn(work)
            if estimate < budget:
                return work
        if through_after <= through_before:
            if _fit_checkpoint_to_active_budget(
                session_id,
                estimate_active_fn=estimate_active_fn,
                budget=budget,
                include_continuation=include_continuation,
            ):
                work = _active_work()
                estimate = estimate_active_fn(work)
                if estimate < budget:
                    return work
            break
    if estimate >= budget:
        if _fit_checkpoint_to_active_budget(
            session_id,
            estimate_active_fn=estimate_active_fn,
            budget=budget,
            include_continuation=include_continuation,
        ):
            work = _active_work()
            estimate = estimate_active_fn(work)
    if estimate >= budget:
        log_event(
            log_source,
            "checkpoint_still_over_budget",
            {
                "active_estimate": estimate,
                "active_events": len(work),
                "budget": budget,
                "reason": reason,
            },
        )
        raise RuntimeError(
            "active context still exceeds input budget after "
            f"compaction (~{estimate:,} tokens; budget "
            f"{budget:,})"
        )
    return work


async def request_with_checkpoint_retry(
    work: list[dict[str, Any]],
    *,
    request_fn: Callable[
        [list[dict[str, Any]]],
        Awaitable[dict[str, Any]],
    ],
    compact_fn: Callable[
        [list[dict[str, Any]]],
        Awaitable[list[dict[str, Any]] | None],
    ],
    overflow_error: type[BaseException],
    on_overflow: Callable[[BaseException], None] | None = None,
) -> dict[str, Any]:
    try:
        return await request_fn(work)
    except overflow_error as exc:
        if on_overflow is not None:
            on_overflow(exc)
        compacted = await compact_fn(work)
        if compacted is None:
            raise
        work[:] = compacted
        return await request_fn(work)


def session_context_stats(session_id: str) -> dict[str, Any]:
    projection = active_projection(session_id)
    items = projection.items
    checkpoint = projection.checkpoint
    through = int(checkpoint.get("through_seq") or 0) if checkpoint else 0
    active_native = [
        event for event in items if int(event.get("seq") or 0) > through
    ]
    return {
        "archive_events": projection.archive_events,
        "checkpoint": bool(checkpoint),
        "checkpoint_through": through if checkpoint else None,
        "active_native_events": len(active_native),
    }
