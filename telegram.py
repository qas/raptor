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
from config import TELEGRAM_MARKDOWN, TG_API, TG_BOT_TOKEN, TG_USER_ID
from observability import log_event

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
            if exc.status_code != 400 or not any(
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


async def send(
    chat_id: int,
    text: str,
) -> None:
    log_event(
        "telegram",
        "sent",
        {
            "chat_id": chat_id,
            "text": text,
        },
    )
    for part in split_message(
        text
    ):
        await send_rich(
            "sendMessage",
            {
                "chat_id": chat_id,
            },
            part,
        )


async def send_draft(
    chat_id: int,
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
            "chat_id":
                chat_id,
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
    primary_conversation_id = TG_USER_ID
    capabilities = ProviderCapabilities(
        drafts=True,
        pins=True,
        controls=True,
        typing=True,
    )

    async def initialize(
        self,
        commands: tuple[tuple[str, str], ...],
    ) -> None:
        global _client
        if not TG_USER_ID or not TG_BOT_TOKEN:
            raise RuntimeError(
                "Telegram requires TG_BOT_TOKEN and TG_USER_ID"
            )
        if _client is None:
            _client = httpx.AsyncClient(
                timeout=httpx.Timeout(65.0, connect=10.0),
            )
        await tg_call("deleteWebhook", {"drop_pending_updates": False})
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
            return IncomingAction(
                action_id=str(callback.get("id") or ""),
                conversation_id=chat.get("id"),
                sender_id=sender_id,
                message_id=message.get("message_id"),
                data=str(callback.get("data") or ""),
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
        return IncomingMessage(
            conversation_id=conversation_id,
            sender_id=sender_id,
            message_id=message_id,
            text=str(message.get("text") or ""),
            private=chat.get("type") == "private",
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
        await send(int(conversation_id), text)

    async def send_draft(
        self,
        conversation_id,
        draft_id: int,
        text: str,
    ) -> None:
        await send_draft(int(conversation_id), draft_id, text)

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
        payload: dict[str, Any] = {"chat_id": conversation_id}
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
        payload: dict[str, Any] = {
            "chat_id": conversation_id,
            "message_id": message_id,
        }
        markup = _reply_markup(controls)
        payload["reply_markup"] = markup or {"inline_keyboard": []}
        await send_rich("editMessageText", payload, text)

    async def delete_message(self, conversation_id, message_id) -> None:
        await tg_call(
            "deleteMessage",
            {"chat_id": conversation_id, "message_id": message_id},
        )

    async def pin_message(self, conversation_id, message_id) -> None:
        await tg_call(
            "pinChatMessage",
            {
                "chat_id": conversation_id,
                "message_id": message_id,
                "disable_notification": True,
            },
        )

    async def unpin_message(self, conversation_id, message_id) -> None:
        await tg_call(
            "unpinChatMessage",
            {"chat_id": conversation_id, "message_id": message_id},
        )

    async def set_typing(self, conversation_id, active: bool) -> None:
        if not active:
            return
        await tg_call(
            "sendChatAction",
            {"chat_id": conversation_id, "action": "typing"},
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
        return str(int(conversation_id))

    @staticmethod
    def decode_conversation_id(value: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"invalid Telegram conversation ID: {value}") from exc

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
