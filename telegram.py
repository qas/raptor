"""Telegram chat-provider adapter and Markdown rendering."""
import asyncio
import re
from html import escape
from typing import Any

import httpx

from chat_provider import (
    ChatEvent,
    ConversationId,
    Controls,
    IncomingAction,
    IncomingMessage,
    PollResult,
    ProviderCapabilities,
)
from config import (
    TELEGRAM_MARKDOWN,
    TG_API,
    TG_BOT_TOKEN,
    TG_CHAT_ID,
    TG_USER_ID,
)
from observability import log_event, log_exception
from activity import ActivitySnapshot

_client: httpx.AsyncClient | None = None

_CHAT_REQUEST_INTERVAL = 1.1
_GLOBAL_REQUEST_INTERVAL = 1.0 / 28.0
_PACED_CHAT_METHODS = frozenset(
    {
        "deletemessage",
        "editmessagereplymarkup",
        "editmessagetext",
        "pinchatmessage",
        "sendchataction",
        "sendmessage",
        "sendmessagedraft",
        "unpinchatmessage",
    }
)
_RATE_LOOP: asyncio.AbstractEventLoop | None = None
_RATE_GUARD: asyncio.Lock | None = None
_CHAT_LOCKS: dict[int | str, asyncio.Lock] = {}
_CHAT_READY_AT: dict[int | str, float] = {}
_GLOBAL_READY_AT = 0.0
_FLOOD_BLOCKED_UNTIL = 0.0


def _telegram_destination(
    conversation_id: ConversationId,
) -> tuple[int, int | None]:
    value = str(conversation_id)
    chat_value, separator, thread_value = value.partition("/")
    try:
        chat_id = int(chat_value)
        thread_id = int(thread_value) if separator else None
    except ValueError as exc:
        raise ValueError(
            f"invalid Telegram conversation ID: {value}"
        ) from exc
    if not chat_id or (thread_id is not None and thread_id <= 1):
        raise ValueError(f"invalid Telegram conversation ID: {value}")
    return chat_id, thread_id


def _telegram_conversation_id(
    chat_id: int,
    thread_id: int | None,
) -> str:
    if not chat_id:
        raise ValueError("Telegram chat ID must not be zero")
    if thread_id is None or thread_id == 1:
        return str(chat_id)
    if thread_id <= 1:
        raise ValueError("Telegram topic ID must be positive")
    return f"{chat_id}/{thread_id}"


def _message_thread_id(message: dict[str, Any]) -> int | None:
    if not message.get("is_topic_message"):
        return None
    value = message.get("message_thread_id")
    return value if isinstance(value, int) and value > 1 else None


def _telegram_payload(
    conversation_id: ConversationId,
) -> dict[str, int]:
    chat_id, thread_id = _telegram_destination(conversation_id)
    payload = {"chat_id": chat_id}
    if thread_id is not None:
        payload["message_thread_id"] = thread_id
    return payload


def _activity_surface_id(topic_id: int, message_id: int) -> str:
    if topic_id <= 1 or message_id <= 0:
        raise ValueError("invalid Telegram activity surface")
    return f"{topic_id}/{message_id}"


def _parse_activity_surface_id(value: str) -> tuple[int, int]:
    topic_value, separator, message_value = str(value).partition("/")
    try:
        topic_id = int(topic_value)
        message_id = int(message_value) if separator else 0
    except ValueError as exc:
        raise ValueError("invalid Telegram activity surface") from exc
    if topic_id <= 1 or message_id <= 0:
        raise ValueError("invalid Telegram activity surface")
    return topic_id, message_id


def _activity_text(snapshot: ActivitySnapshot) -> str:
    lines = [
        f"Subagent: {snapshot.activity_id}",
        "",
        f"Task: {snapshot.title}",
        f"Status: {snapshot.status}",
    ]
    if snapshot.detail:
        lines.extend(("", snapshot.detail))
    if snapshot.reasoning_summary:
        lines.extend(("", "Reasoning", snapshot.reasoning_summary))
    if snapshot.reply:
        lines.extend(("", "Reply", snapshot.reply))
    elif snapshot.result:
        lines.extend(("", "Result", snapshot.result))
    lines.extend(("", "Read-only. Steer from the parent chat."))
    return "\n".join(lines)


async def _delete_forum_topic(chat_id: int, topic_id: int) -> None:
    try:
        await tg_call(
            "deleteForumTopic",
            {"chat_id": chat_id, "message_thread_id": topic_id},
        )
    except TelegramApiError as exc:
        description = exc.description.casefold()
        if exc.is_bad_request and "topic" in description and any(
            marker in description
            for marker in ("not found", "deleted", "does not exist")
        ):
            return
        raise


class TelegramApiError(RuntimeError):
    """Structured Bot API failure that preserves flood-control metadata."""

    def __init__(
        self,
        method: str,
        *,
        status_code: int,
        error_code: int | None,
        description: str,
        retry_after: float | None = None,
    ) -> None:
        self.method = method
        self.status_code = status_code
        self.error_code = error_code
        self.description = description
        self.retry_after = retry_after
        code = error_code if error_code is not None else status_code
        super().__init__(f"Telegram {method} error {code}: {description}")

    @property
    def is_bad_request(self) -> bool:
        return self.status_code == 400 or self.error_code == 400


def _ensure_rate_state() -> asyncio.AbstractEventLoop:
    global _RATE_LOOP, _RATE_GUARD
    global _GLOBAL_READY_AT, _FLOOD_BLOCKED_UNTIL
    loop = asyncio.get_running_loop()
    if _RATE_LOOP is not loop:
        _RATE_LOOP = loop
        _RATE_GUARD = asyncio.Lock()
        _CHAT_LOCKS.clear()
        _CHAT_READY_AT.clear()
        _GLOBAL_READY_AT = 0.0
        _FLOOD_BLOCKED_UNTIL = 0.0
    return loop


def _chat_lock(chat_id: int | str) -> asyncio.Lock:
    _ensure_rate_state()
    lock = _CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _CHAT_LOCKS[chat_id] = lock
    return lock


async def _reserve_telegram_request(
    method: str,
    chat_id: int | str | None,
) -> None:
    """Reserve one globally and per-chat paced Bot API request slot."""
    del method
    global _GLOBAL_READY_AT
    loop = _ensure_rate_state()
    assert _RATE_GUARD is not None
    while True:
        async with _RATE_GUARD:
            now = loop.time()
            ready_at = max(_GLOBAL_READY_AT, _FLOOD_BLOCKED_UNTIL)
            if chat_id is not None:
                ready_at = max(ready_at, _CHAT_READY_AT.get(chat_id, 0.0))
            if ready_at <= now:
                _GLOBAL_READY_AT = now + _GLOBAL_REQUEST_INTERVAL
                if chat_id is not None:
                    _CHAT_READY_AT[chat_id] = now + _CHAT_REQUEST_INTERVAL
                return
            delay = ready_at - now
        await asyncio.sleep(delay)


async def _defer_telegram_requests(
    chat_id: int | str | None,
    retry_after: float,
) -> None:
    global _FLOOD_BLOCKED_UNTIL
    loop = _ensure_rate_state()
    assert _RATE_GUARD is not None
    async with _RATE_GUARD:
        blocked_until = loop.time() + max(0.0, retry_after) + 0.1
        _FLOOD_BLOCKED_UNTIL = max(
            _FLOOD_BLOCKED_UNTIL,
            blocked_until,
        )
        if chat_id is not None:
            _CHAT_READY_AT[chat_id] = max(
                _CHAT_READY_AT.get(chat_id, 0.0),
                blocked_until,
            )


def _telegram_error(
    method: str,
    response: httpx.Response,
    data: object,
) -> TelegramApiError:
    body = data if isinstance(data, dict) else {}
    parameters = body.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    retry_value = parameters.get("retry_after", parameters.get("retryafter"))
    if retry_value is None:
        retry_value = response.headers.get("Retry-After")
    try:
        retry_after = float(retry_value) if retry_value is not None else None
    except (TypeError, ValueError):
        retry_after = None
    error_value = body.get("error_code", body.get("errorcode"))
    try:
        error_code = int(error_value) if error_value is not None else None
    except (TypeError, ValueError):
        error_code = None
    return TelegramApiError(
        method,
        status_code=response.status_code,
        error_code=error_code,
        description=str(body.get("description") or response.reason_phrase),
        retry_after=retry_after,
    )

# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

async def tg_call(
    method: str,
    payload: dict[str, Any]
    | None = None,
) -> Any:
    if _client is None:
        raise RuntimeError("Telegram provider is not initialized")
    request_payload = payload or {}
    normalized_method = method.lower()
    raw_chat_id = request_payload.get("chat_id")
    chat_id = (
        raw_chat_id
        if normalized_method in _PACED_CHAT_METHODS
        and isinstance(raw_chat_id, (int, str))
        else None
    )

    async def perform() -> Any:
        while True:
            await _reserve_telegram_request(method, chat_id)
            response = await _client.post(
                f"{TG_API}/{method}",
                json=request_payload,
            )
            try:
                data: object = response.json()
            except ValueError:
                data = None
            error = _telegram_error(method, response, data)
            if response.status_code == 429 or error.error_code == 429:
                retry_after = error.retry_after or 1.0
                log_event(
                    "telegram",
                    "rate_limited",
                    {
                        "method": method,
                        "chat_id": chat_id,
                        "retry_after": retry_after,
                    },
                )
                await _defer_telegram_requests(chat_id, retry_after)
                continue
            if not response.is_success:
                raise error
            if not isinstance(data, dict) or not data.get("ok"):
                raise error
            return data.get("result")

    if chat_id is None:
        return await perform()
    async with _chat_lock(chat_id):
        return await perform()


def split_message(
    text: str,
    limit: int = 3900,
) -> list[str]:
    text = (
        text
        or "(no text returned)"
    )

    output: list[str] = []

    while len(text) > limit:
        cut = text.rfind(
            "\n",
            0,
            limit,
        )

        if cut < limit // 2:
            cut = limit

        output.append(
            text[:cut]
        )

        text = text[
            cut:
        ].lstrip(
            "\n"
        )

    output.append(
        text
    )

    return output


# ---------------------------------------------------------------------------
# Markdown -> Telegram HTML
# ---------------------------------------------------------------------------

def inline_markdown_to_html(
    text: str,
) -> str:
    placeholders: list[str] = []

    def stash(
        html_text: str,
    ) -> str:
        token = (
            f"\x00{len(placeholders)}\x00"
        )

        placeholders.append(
            html_text
        )

        return token

    text = re.sub(
        r"`([^`\n]+)`",
        lambda match: stash(
            "<code>"
            + escape(
                match.group(1)
            )
            + "</code>"
        ),
        text,
    )

    text = escape(
        text
    )

    text = re.sub(
        r"\[([^\]]+)\]"
        r"\((https?://[^\s)]+)\)",
        lambda match: (
            '<a href="'
            + escape(
                match.group(2),
                quote=True,
            )
            + '">'
            + match.group(1)
            + "</a>"
        ),
        text,
    )

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text,
    )

    text = re.sub(
        r"__(.+?)__",
        r"<b>\1</b>",
        text,
    )

    text = re.sub(
        r"~~(.+?)~~",
        r"<s>\1</s>",
        text,
    )

    text = re.sub(
        r"(?<!\*)"
        r"\*([^*\n]+?)\*"
        r"(?!\*)",
        r"<i>\1</i>",
        text,
    )

    text = re.sub(
        r"(?<!_)"
        r"_([^_\n]+?)_"
        r"(?!_)",
        r"<i>\1</i>",
        text,
    )

    for index, html_text in enumerate(
        placeholders
    ):
        text = text.replace(
            escape(
                f"\x00{index}\x00"
            ),
            html_text,
        )

    return text


def markdown_to_telegram_html(
    markdown: str,
) -> str:
    output: list[str] = []

    lines = markdown.splitlines()

    index = 0

    while index < len(lines):
        line = lines[
            index
        ]

        fence = re.match(
            r"^\s*```([^`]*)$",
            line,
        )

        if fence:
            language = (
                fence.group(1)
                .strip()
            )

            code_lines: list[str] = []

            index += 1

            while (
                index < len(lines)
                and not re.match(
                    r"^\s*```\s*$",
                    lines[index],
                )
            ):
                code_lines.append(
                    lines[index]
                )

                index += 1

            code = escape(
                "\n".join(
                    code_lines
                )
            )

            if (
                language
                and re.fullmatch(
                    r"[A-Za-z0-9_+.#-]+",
                    language,
                )
            ):
                output.append(
                    '<pre><code class="language-'
                    + escape(
                        language,
                        quote=True,
                    )
                    + '">'
                    + code
                    + "</code></pre>"
                )

            else:
                output.append(
                    f"<pre>{code}</pre>"
                )

            if index < len(lines):
                index += 1

            continue

        heading = re.match(
            r"^\s{0,3}"
            r"(#{1,6})\s+(.+)$",
            line,
        )

        if heading:
            output.append(
                "<b>"
                + inline_markdown_to_html(
                    heading.group(2)
                )
                + "</b>"
            )

        elif re.match(
            r"^\s*[-*+]\s+",
            line,
        ):
            body = re.sub(
                r"^\s*[-*+]\s+",
                "",
                line,
            )

            output.append(
                "• "
                + inline_markdown_to_html(
                    body
                )
            )

        elif re.match(
            r"^\s*\d+[.)]\s+",
            line,
        ):
            item = re.match(
                r"^\s*(\d+)[.)]\s+(.+)$",
                line,
            )

            if item:
                output.append(
                    f"{item.group(1)}. "
                    + inline_markdown_to_html(
                        item.group(2)
                    )
                )

            else:
                output.append(
                    inline_markdown_to_html(
                        line
                    )
                )

        elif re.match(
            r"^\s*>\s?",
            line,
        ):
            body = re.sub(
                r"^\s*>\s?",
                "",
                line,
            )

            output.append(
                "<blockquote>"
                + inline_markdown_to_html(
                    body
                )
                + "</blockquote>"
            )

        else:
            output.append(
                inline_markdown_to_html(
                    line
                )
            )

        index += 1

    return "\n".join(
        output
    )


async def send_rich(
    method: str,
    payload: dict[str, Any],
    raw_text: str,
) -> Any:
    if TELEGRAM_MARKDOWN:
        rich_payload = dict(
            payload
        )

        rich_payload["text"] = (
            markdown_to_telegram_html(
                raw_text
            )
        )

        rich_payload[
            "parse_mode"
        ] = "HTML"

        try:
            return await tg_call(
                method,
                rich_payload,
            )

        except TelegramApiError as exc:
            description = exc.description.lower()
            if not exc.is_bad_request or not any(
                marker in description
                for marker in (
                    "can't parse entities",
                    "can't find end tag",
                    "unsupported start tag",
                )
            ):
                raise

    plain_payload = dict(
        payload
    )

    plain_payload[
        "text"
    ] = raw_text

    plain_payload.pop(
        "parse_mode",
        None,
    )

    return await tg_call(
        method,
        plain_payload,
    )


def _is_unchanged_message_error(exc: TelegramApiError) -> bool:
    return (
        exc.is_bad_request
        and "message is not modified" in exc.description.casefold()
    )


async def _edit_rich_message(
    payload: dict[str, Any],
    text: str,
) -> None:
    try:
        await send_rich("editMessageText", payload, text)
    except TelegramApiError as exc:
        if _is_unchanged_message_error(exc):
            return
        raise


async def send(
    conversation_id: ConversationId,
    text: str,
) -> None:
    chat_id, _thread_id = _telegram_destination(conversation_id)
    log_event(
        "telegram",
        "sent",
        {
            "chat_id": chat_id,
            "text_chars": len(text),
        },
    )
    for part in split_message(
        text
    ):
        await send_rich(
            "sendMessage",
            _telegram_payload(conversation_id),
            part,
        )


async def send_draft(
    conversation_id: ConversationId,
    draft_id: int,
    text: str,
) -> None:
    preview = (
        text[-3900:]
        if len(text) > 3900
        else text
    )

    await send_rich(
        "sendMessageDraft",
        {
            **_telegram_payload(conversation_id),
            "draft_id":
                draft_id,
        },
        preview,
    )


def _reply_markup(controls: Controls) -> dict[str, Any] | None:
    if not controls:
        return None
    return {
        "inline_keyboard": [
            [
                {
                    "text": button.label,
                    "callback_data": button.action,
                }
                for button in row
            ]
            for row in controls
        ],
    }


class TelegramProvider:
    name = "telegram"
    authorized_user_id = TG_USER_ID
    primary_conversation_id = _telegram_conversation_id(TG_CHAT_ID, None)
    capabilities = ProviderCapabilities(
        drafts=True,
        pins=True,
        controls=True,
        typing=True,
    )

    def __init__(self) -> None:
        self._chat_type = ""
        self._forum_enabled = False
        self._activity_topic_ids: set[int] = set()

    async def initialize(
        self,
        commands: tuple[tuple[str, str], ...],
    ) -> None:
        global _client
        if not TG_USER_ID or not TG_CHAT_ID or not TG_BOT_TOKEN:
            raise RuntimeError(
                "Telegram requires TG_BOT_TOKEN, TG_USER_ID, and TG_CHAT_ID"
            )
        if _client is None:
            _client = httpx.AsyncClient(
                timeout=httpx.Timeout(65.0, connect=10.0),
            )
        await tg_call("deleteWebhook", {"drop_pending_updates": False})
        chat = await tg_call("getChat", {"chat_id": TG_CHAT_ID})
        if not isinstance(chat, dict):
            raise RuntimeError("Telegram returned no configured chat")
        self._chat_type = str(chat.get("type") or "")
        if self._chat_type not in {"private", "group", "supergroup"}:
            raise RuntimeError(
                "TG_CHAT_ID must identify a private chat, group, or supergroup"
            )
        self._forum_enabled = bool(chat.get("is_forum"))
        self.capabilities = ProviderCapabilities(
            drafts=self._chat_type == "private",
            pins=True,
            controls=True,
            typing=True,
        )
        if self._forum_enabled:
            bot = await tg_call("getMe")
            bot_id = bot.get("id") if isinstance(bot, dict) else None
            if not isinstance(bot_id, int):
                raise RuntimeError("Telegram returned no bot identity")
            member = await tg_call(
                "getChatMember",
                {"chat_id": TG_CHAT_ID, "user_id": bot_id},
            )
            status = str(member.get("status") or "") if isinstance(
                member, dict
            ) else ""
            can_manage_topics = bool(
                isinstance(member, dict) and member.get("can_manage_topics")
            )
            if status != "creator" and not (
                status == "administrator" and can_manage_topics
            ):
                raise RuntimeError(
                    "Telegram forum mode requires Manage Topics permission"
                )
        await tg_call(
            "setMyCommands",
            {
                "commands": [
                    {"command": name, "description": description}
                    for name, description in commands
                ],
            },
        )

    async def close(self) -> None:
        global _client
        if _client is not None:
            await _client.aclose()
            _client = None

    def normalize_update(self, update: dict[str, Any]) -> ChatEvent | None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            message = callback.get("message") or {}
            chat = message.get("chat") or {}
            sender = callback.get("from") or {}
            sender_id = sender.get("id")
            if sender_id is None:
                return None
            conversation_id = chat.get("id")
            thread_id = _message_thread_id(message)
            interactive = (
                conversation_id == TG_CHAT_ID
                and thread_id not in self._activity_topic_ids
            )
            return IncomingAction(
                action_id=str(callback.get("id") or ""),
                conversation_id=(
                    _telegram_conversation_id(conversation_id, thread_id)
                    if isinstance(conversation_id, int)
                    else None
                ),
                sender_id=sender_id,
                message_id=message.get("message_id"),
                data=str(callback.get("data") or ""),
                interactive=interactive,
            )
        message = update.get("message")
        if not isinstance(message, dict) or "text" not in message:
            return None
        sender = message.get("from") or {}
        chat = message.get("chat") or {}
        sender_id = sender.get("id")
        conversation_id = chat.get("id")
        message_id = message.get("message_id")
        if sender_id is None or conversation_id is None or message_id is None:
            return None
        thread_id = _message_thread_id(message)
        return IncomingMessage(
            conversation_id=_telegram_conversation_id(
                conversation_id,
                thread_id,
            ),
            sender_id=sender_id,
            message_id=message_id,
            text=str(message.get("text") or ""),
            interactive=(
                conversation_id == TG_CHAT_ID
                and thread_id not in self._activity_topic_ids
            ),
        )

    async def poll(
        self,
        cursor: object | None,
        *,
        timeout: int,
    ) -> PollResult:
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"],
        }
        if isinstance(cursor, int):
            payload["offset"] = cursor
        updates = await tg_call("getUpdates", payload)
        events: list[ChatEvent] = []
        next_cursor = cursor
        for update in updates or []:
            if not isinstance(update, dict):
                continue
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                next_cursor = update_id + 1
            event = self.normalize_update(update)
            if event is not None:
                events.append(event)
        return PollResult(tuple(events), next_cursor)

    async def send_text(self, conversation_id, text: str) -> None:
        await send(conversation_id, text)

    async def send_draft(
        self,
        conversation_id,
        draft_id: int,
        text: str,
    ) -> None:
        if self._chat_type != "private":
            return
        await send_draft(conversation_id, draft_id, text)

    async def send_reasoning_summary(
        self,
        conversation_id,
        delta: str,
    ) -> None:
        del conversation_id, delta

    async def create_message(
        self,
        conversation_id,
        text: str,
        controls: Controls = (),
    ) -> int:
        payload: dict[str, Any] = _telegram_payload(conversation_id)
        markup = _reply_markup(controls)
        if markup is not None:
            payload["reply_markup"] = markup
        result = await send_rich("sendMessage", payload, text)
        if not isinstance(result, dict) or not isinstance(
            result.get("message_id"), int
        ):
            raise RuntimeError("Telegram returned no message_id")
        return int(result["message_id"])

    async def edit_message(
        self,
        conversation_id,
        message_id,
        text: str,
        controls: Controls = (),
    ) -> None:
        chat_id, _thread_id = _telegram_destination(conversation_id)
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }
        markup = _reply_markup(controls)
        payload["reply_markup"] = markup or {"inline_keyboard": []}
        await _edit_rich_message(payload, text)

    async def delete_message(self, conversation_id, message_id) -> None:
        chat_id, _thread_id = _telegram_destination(conversation_id)
        await tg_call(
            "deleteMessage",
            {"chat_id": chat_id, "message_id": message_id},
        )

    async def pin_message(self, conversation_id, message_id) -> None:
        chat_id, _thread_id = _telegram_destination(conversation_id)
        await tg_call(
            "pinChatMessage",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "disable_notification": True,
            },
        )

    async def unpin_message(self, conversation_id, message_id) -> None:
        chat_id, _thread_id = _telegram_destination(conversation_id)
        await tg_call(
            "unpinChatMessage",
            {"chat_id": chat_id, "message_id": message_id},
        )

    async def set_typing(self, conversation_id, active: bool) -> None:
        if not active:
            return
        payload: dict[str, Any] = _telegram_payload(conversation_id)
        payload["action"] = "typing"
        await tg_call(
            "sendChatAction",
            payload,
        )

    async def reject_busy_message(self, conversation_id) -> bool:
        del conversation_id
        return False

    async def acknowledge_queued_message(self, conversation_id) -> None:
        del conversation_id

    async def finish_event(self, event: ChatEvent) -> None:
        del event

    @staticmethod
    def encode_conversation_id(conversation_id: ConversationId) -> str:
        chat_id, thread_id = _telegram_destination(conversation_id)
        return _telegram_conversation_id(chat_id, thread_id)

    @staticmethod
    def decode_conversation_id(value: str) -> str:
        chat_id, thread_id = _telegram_destination(value)
        return _telegram_conversation_id(chat_id, thread_id)

    @staticmethod
    def prepare_event(event: ChatEvent) -> None:
        del event

    def capture_delivery_context(self, conversation_id):
        del conversation_id
        return None

    def activate_delivery_context(self, conversation_id, delivery_context):
        del conversation_id, delivery_context
        return None

    def restore_delivery_context(self, token) -> None:
        del token

    async def open_activity_surface(
        self,
        conversation_id: ConversationId,
        snapshot: ActivitySnapshot,
    ) -> str | None:
        if not self._forum_enabled:
            return None
        chat_id, _parent_thread_id = _telegram_destination(conversation_id)
        topic_name = f"Subagent: {snapshot.activity_id}"
        topic = await tg_call(
            "createForumTopic",
            {"chat_id": chat_id, "name": topic_name},
        )
        topic_id = (
            topic.get("message_thread_id") if isinstance(topic, dict) else None
        )
        if not isinstance(topic_id, int) or topic_id <= 1:
            raise RuntimeError("Telegram returned no forum topic ID")
        self._activity_topic_ids.add(topic_id)
        try:
            result = await send_rich(
                "sendMessage",
                {"chat_id": chat_id, "message_thread_id": topic_id},
                _activity_text(snapshot),
            )
            message_id = (
                result.get("message_id") if isinstance(result, dict) else None
            )
            if not isinstance(message_id, int):
                raise RuntimeError("Telegram returned no activity message ID")
            return _activity_surface_id(topic_id, message_id)
        except asyncio.CancelledError:
            self._activity_topic_ids.discard(topic_id)
            try:
                await _delete_forum_topic(chat_id, topic_id)
            except Exception as exc:
                log_exception(
                    "telegram",
                    "activity_topic_cleanup_error",
                    exc,
                    {"topic_id": topic_id},
                )
            raise
        except Exception:
            self._activity_topic_ids.discard(topic_id)
            try:
                await _delete_forum_topic(chat_id, topic_id)
            except Exception as exc:
                log_exception(
                    "telegram",
                    "activity_topic_cleanup_error",
                    exc,
                    {"topic_id": topic_id},
                )
            raise

    async def update_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> None:
        chat_id, _parent_thread_id = _telegram_destination(conversation_id)
        _topic_id, message_id = _parse_activity_surface_id(surface_id)
        await _edit_rich_message(
            {"chat_id": chat_id, "message_id": message_id},
            _activity_text(snapshot),
        )

    async def close_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> None:
        chat_id, _parent_thread_id = _telegram_destination(conversation_id)
        topic_id, message_id = _parse_activity_surface_id(surface_id)
        try:
            await _edit_rich_message(
                {"chat_id": chat_id, "message_id": message_id},
                _activity_text(snapshot),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                "telegram",
                "activity_final_update_error",
                exc,
                {"topic_id": topic_id},
            )
        await _delete_forum_topic(chat_id, topic_id)
        self._activity_topic_ids.discard(topic_id)

    def restore_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> None:
        chat_id, _parent_thread_id = _telegram_destination(conversation_id)
        if chat_id != TG_CHAT_ID:
            raise ValueError("activity surface belongs to another Telegram chat")
        topic_id, _message_id = _parse_activity_surface_id(surface_id)
        self._activity_topic_ids.add(topic_id)

    async def answer_action(
        self,
        action_id: str,
        text: str = "",
        *,
        alert: bool = False,
    ) -> None:
        payload: dict[str, Any] = {"callback_query_id": action_id}
        if text:
            payload["text"] = text
        if alert:
            payload["show_alert"] = True
        await tg_call("answerCallbackQuery", payload)


telegram_provider = TelegramProvider()
