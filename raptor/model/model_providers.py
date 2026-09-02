"""Workspace-local outbound model-provider configuration."""

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from raptor.config_document import CONFIG_PATH, load_config_document


_PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
@dataclass(frozen=True)
class ModelTarget:
    provider_id: str
    model: str

    def to_dict(self) -> dict[str, str]:
        return {"provider_id": self.provider_id, "model": self.model}

    @classmethod
    def from_value(cls, value: Any) -> "ModelTarget":
        if not isinstance(value, dict):
            raise ValueError("model target must be an object")
        provider_id = str(value.get("provider_id") or "").strip()
        model = str(value.get("model") or "").strip()
        if not provider_id:
            raise ValueError("model target requires provider_id")
        if not model:
            raise ValueError("model target requires model")
        return cls(provider_id=provider_id, model=model)


@dataclass(frozen=True)
class ModelSettings:
    context_window: int | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = "auto"


@dataclass(frozen=True)
class ModelProvider:
    id: str
    base_url: str
    api_key_env: str | None = None
    default_model: str = ""
    request_max_retries: int = 3
    retry_base_seconds: float = 0.5
    context_window: int | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = "auto"
    models: dict[str, ModelSettings] | None = None

    def api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        value = os.getenv(self.api_key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"Model provider {self.id!r} requires {self.api_key_env}"
            )
        return value

    def headers(self) -> dict[str, str]:
        key = self.api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def settings_for(self, model: str) -> ModelSettings:
        override = (self.models or {}).get(model)
        if override is not None:
            return override
        return ModelSettings(
            context_window=self.context_window,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
        )


@dataclass(frozen=True)
class ModelConfiguration:
    providers: dict[str, ModelProvider]
    default_target: ModelTarget
    compaction_provider_id: str | None = None
    compaction_model: str | None = None

    def provider(self, provider_id: str) -> ModelProvider:
        try:
            return self.providers[provider_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model provider {provider_id!r}"
            ) from exc

    def validate_target(self, target: ModelTarget) -> ModelTarget:
        self.provider(target.provider_id)
        if not target.model.strip():
            raise ValueError("model target requires model")
        return target

    def select_target(
        self,
        *,
        parent: ModelTarget,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> ModelTarget:
        selected_provider_id = (provider_id or parent.provider_id).strip()
        provider = self.provider(selected_provider_id)
        selected_model = (model or "").strip()
        if not selected_model:
            if selected_provider_id == parent.provider_id:
                selected_model = parent.model
            else:
                selected_model = provider.default_model
        if not selected_model:
            raise ValueError(
                f"Model provider {selected_provider_id!r} has no default_model; "
                "specify model explicitly"
            )
        return self.validate_target(
            ModelTarget(selected_provider_id, selected_model)
        )

    def select_compaction_target(self, parent: ModelTarget) -> ModelTarget:
        """Resolve the optional compaction override against an agent target."""
        if self.compaction_provider_id is None and self.compaction_model is None:
            return self.validate_target(parent)
        provider_id = (
            self.compaction_provider_id or parent.provider_id
        ).strip()
        provider = self.provider(provider_id)
        model = (
            provider.default_model
            if self.compaction_model is None
            else self.compaction_model
        ).strip()
        if not model:
            raise ValueError(
                f"Model provider {provider_id!r} has no default_model; "
                "set compaction.model explicitly"
            )
        return self.validate_target(
            ModelTarget(provider_id=provider_id, model=model)
        )


def _optional_string(value: Any, name: str, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip() or None


def _positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _provider_from_table(provider_id: str, value: Any) -> ModelProvider:
    if not _PROVIDER_ID_RE.fullmatch(provider_id):
        raise ValueError(f"Invalid model provider ID {provider_id!r}")
    if not isinstance(value, dict):
        raise ValueError(f"model_providers.{provider_id} must be a table")
    allowed = {
        "base_url",
        "api_key_env",
        "default_model",
        "request_max_retries",
        "retry_base_seconds",
        "context_window",
        "reasoning_effort",
        "reasoning_summary",
        "models",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown model_providers.{provider_id} fields: {', '.join(unknown)}"
        )
    base_url = str(value.get("base_url") or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            f"model_providers.{provider_id}.base_url must be an HTTP(S) URL"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"model_providers.{provider_id}.base_url cannot contain query or fragment"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            f"model_providers.{provider_id}.base_url cannot contain credentials"
        )
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(
            f"model_providers.{provider_id}.base_url has an invalid port"
        ) from exc
    api_key_env = _optional_string(
        value.get("api_key_env"),
        f"model_providers.{provider_id}.api_key_env",
        None,
    )
    if api_key_env and not _ENV_NAME_RE.fullmatch(api_key_env):
        raise ValueError(
            f"model_providers.{provider_id}.api_key_env is invalid"
        )
    retries = value.get("request_max_retries", 3)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError(
            f"model_providers.{provider_id}.request_max_retries must be non-negative"
        )
    retry_base = value.get("retry_base_seconds", 0.5)
    if not isinstance(retry_base, (int, float)) or isinstance(retry_base, bool):
        raise ValueError(
            f"model_providers.{provider_id}.retry_base_seconds must be a number"
        )
    retry_base = float(retry_base)
    if not math.isfinite(retry_base) or retry_base < 0:
        raise ValueError(
            f"model_providers.{provider_id}.retry_base_seconds must be non-negative"
        )
    provider_reasoning_effort = _optional_string(
        value.get("reasoning_effort"),
        f"model_providers.{provider_id}.reasoning_effort",
        None,
    )
    provider_reasoning_summary = _optional_string(
        value.get("reasoning_summary"),
        f"model_providers.{provider_id}.reasoning_summary",
        "auto",
    )
    provider_context_window = _positive_int(
        value.get("context_window"),
        f"model_providers.{provider_id}.context_window",
    )
    model_tables = value.get("models", {})
    if not isinstance(model_tables, dict):
        raise ValueError(f"model_providers.{provider_id}.models must be a table")
    models: dict[str, ModelSettings] = {}
    for model_id, raw in model_tables.items():
        if not str(model_id).strip() or not isinstance(raw, dict):
            raise ValueError(
                f"model_providers.{provider_id}.models entries must be tables"
            )
        model_unknown = sorted(
            set(raw) - {"context_window", "reasoning_effort", "reasoning_summary"}
        )
        if model_unknown:
            raise ValueError(
                f"Unknown model settings for {provider_id}/{model_id}: "
                + ", ".join(model_unknown)
            )
        models[str(model_id)] = ModelSettings(
            context_window=_positive_int(
                raw.get("context_window"),
                f"model_providers.{provider_id}.models.{model_id}.context_window",
            ) or provider_context_window,
            reasoning_effort=_optional_string(
                raw.get("reasoning_effort"),
                f"model_providers.{provider_id}.models.{model_id}.reasoning_effort",
                provider_reasoning_effort,
            ),
            reasoning_summary=_optional_string(
                raw.get("reasoning_summary"),
                f"model_providers.{provider_id}.models.{model_id}.reasoning_summary",
                provider_reasoning_summary,
            ),
        )
    return ModelProvider(
        id=provider_id,
        base_url=base_url,
        api_key_env=api_key_env,
        default_model=_optional_string(
            value.get("default_model"),
            f"model_providers.{provider_id}.default_model",
            "",
        ) or "",
        request_max_retries=retries,
        retry_base_seconds=retry_base,
        context_window=provider_context_window,
        reasoning_effort=provider_reasoning_effort,
        reasoning_summary=provider_reasoning_summary,
        models=models,
    )


def load_model_configuration(path: Path = CONFIG_PATH) -> ModelConfiguration:
    data: dict[str, Any] = load_config_document(path)
    config_exists = path.exists()
    providers_raw = data.get("model_providers")
    if providers_raw is None:
        if config_exists and any(
            field in data for field in ("model_provider", "model")
        ):
            raise ValueError(
                f"Raptor config must define [model_providers.<id>]: {path}"
            )
        providers_raw = {
            "local": {
                "base_url": "http://127.0.0.1:8000/v1",
                "default_model": "",
            }
        }
    if not isinstance(providers_raw, dict) or not providers_raw:
        raise ValueError("model_providers must contain at least one provider")
    providers = {
        str(provider_id): _provider_from_table(str(provider_id), raw)
        for provider_id, raw in providers_raw.items()
    }
    default_provider_id = _optional_string(
        data.get("model_provider"),
        "model_provider",
        "local",
    ) or "local"
    if default_provider_id not in providers:
        raise ValueError(f"Unknown default model provider {default_provider_id!r}")
    provider = providers[default_provider_id]
    default_model = _optional_string(
        data.get("model"),
        "model",
        provider.default_model,
    ) or ""
    compaction_raw = data.get("compaction", {})
    if not isinstance(compaction_raw, dict):
        raise ValueError("compaction must be a table")
    compaction_provider_id = _optional_string(
        compaction_raw.get("model_provider"),
        "compaction.model_provider",
        None,
    )
    compaction_model = _optional_string(
        compaction_raw.get("model"),
        "compaction.model",
        None,
    )
    if compaction_provider_id is not None:
        if compaction_provider_id not in providers:
            raise ValueError(
                "Unknown compaction model provider "
                f"{compaction_provider_id!r}"
            )
        if compaction_model is None and not providers[
            compaction_provider_id
        ].default_model:
            raise ValueError(
                f"Model provider {compaction_provider_id!r} has no "
                "default_model; set compaction.model explicitly"
            )
    return ModelConfiguration(
        providers=providers,
        default_target=ModelTarget(default_provider_id, default_model),
        compaction_provider_id=compaction_provider_id,
        compaction_model=compaction_model,
    )


MODEL_CONFIGURATION = load_model_configuration()
