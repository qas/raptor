"""Errors shared by Responses transport and context management."""


class ContextLengthError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int | None = None,
        context_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens


class IncompleteResponsesStreamError(RuntimeError):
    """The transport closed before the protocol's terminal event."""


class TransientResponsesError(RuntimeError):
    """A retryable backend failure exhausted its bounded attempts."""
