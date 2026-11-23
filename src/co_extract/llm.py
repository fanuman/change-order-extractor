"""The only module that talks to an LLM.

Works with any OpenAI-compatible API, configured by LLM_API_KEY, LLM_BASE_URL and LLM_MODEL.

Strategy:
1. Ask for structured output, where the API enforces our schema.
2. If the provider rejects that, or replies in a different shape, fall back to JSON mode:
   the schema goes into the instructions, we validate the reply ourselves and allow one repair.
3. Any failure is raised as ExtractionFailed with a reason a person can act on.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import (
    AuthenticationError,
    BadRequestError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAI,
    OpenAIError,
)
from pydantic import ValidationError

from co_extract.prompts import (
    JSON_MODE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    RawExtraction,
    repair_message,
    user_message,
)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
JSON_MODE_ATTEMPTS = 2  # the first reply, plus one repair after showing the model its errors

Message = dict[str, str]


class ConfigError(Exception):
    """Required settings are missing or inconsistent."""


class ExtractionFailed(Exception):
    """The LLM call failed or returned nothing usable."""


class Extractor(Protocol):
    def extract(self, document_text: str) -> RawExtraction: ...


# --- Configuration -------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()  # reads .env if present; variables already set in the environment win
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        base_url = os.environ.get("LLM_BASE_URL", "").strip()
        model = os.environ.get("LLM_MODEL", "").strip()

        missing = [name for name, value in (("LLM_API_KEY", api_key), ("LLM_BASE_URL", base_url)) if not value]
        if missing:
            raise ConfigError(f"missing {', '.join(missing)}: set it in .env (see .env.example) or the environment")
        if not model:
            if not _is_openai(base_url):
                raise ConfigError(
                    f"LLM_MODEL is required for {base_url}: "
                    f"the default model {DEFAULT_OPENAI_MODEL} only exists on OpenAI"
                )
            model = DEFAULT_OPENAI_MODEL
        return cls(api_key=api_key, base_url=base_url, model=model)


def _is_openai(base_url: str) -> bool:
    return urlparse(base_url).hostname == "api.openai.com"


# --- Extraction ----------------------------------------------------------------------------


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleExtractor:
    """Extracts a RawExtraction from document text with an OpenAI-compatible chat API."""

    def __init__(self, config: LLMConfig, client: Any = None) -> None:
        self.config = config
        self.usage = Usage()
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=2,  # the SDK retries rate limits and server errors with backoff
        )
        self._send_temperature = True  # switched off if the model only accepts its default

    def extract(self, document_text: str) -> RawExtraction:
        """Any API error becomes an ExtractionFailed that says what to check."""
        try:
            return self._extract_with_fallback(_message("user", user_message(document_text)))
        except OpenAIError as exc:
            raise _explain(exc, self.config) from exc

    def _extract_with_fallback(self, user: Message) -> RawExtraction:
        """Structured output first; JSON mode if the provider rejects it or replies in another shape."""
        try:
            return self._extract_structured(user)
        except (BadRequestError, ValidationError) as exc:
            structured_error = exc  # Python deletes `exc` when the except block ends

        try:
            return self._extract_json_mode(user)
        except (OpenAIError, ExtractionFailed) as json_error:
            raise ExtractionFailed(
                f"structured output failed ({_short(structured_error)}); JSON mode failed ({_short(json_error)})"
            ) from json_error

    def _extract_structured(self, user: Message) -> RawExtraction:
        completion = self._request(
            self._client.chat.completions.parse,
            messages=[_message("system", SYSTEM_PROMPT), user],
            response_format=RawExtraction,
        )
        reply = completion.choices[0].message
        if reply.refusal:
            raise ExtractionFailed(f"the model refused: {reply.refusal}")
        if reply.parsed is None:
            raise ExtractionFailed("the model returned no parsable output")
        parsed: RawExtraction = reply.parsed
        return parsed

    def _extract_json_mode(self, user: Message) -> RawExtraction:
        messages = [_message("system", JSON_MODE_SYSTEM_PROMPT), user]
        last_error: ValidationError | None = None
        for _ in range(JSON_MODE_ATTEMPTS):
            completion = self._request(
                self._client.chat.completions.create,
                messages=messages,
                response_format={"type": "json_object"},
            )
            reply: str = completion.choices[0].message.content or ""
            try:
                return RawExtraction.model_validate_json(reply)
            except ValidationError as exc:
                last_error = exc
                messages = [*messages, _message("assistant", reply), _message("user", repair_message(str(exc)))]
        errors = last_error.error_count() if last_error else 0
        raise ExtractionFailed(f"JSON mode reply still invalid after one repair attempt ({errors} errors)")

    def _request(self, send: Callable[..., Any], **kwargs: Any) -> Any:
        """Send one chat request and record its token usage.

        Asks for temperature 0 (repeatable output). Some models only accept their default
        temperature; if the provider says so, retry once without it and stop sending it.
        """
        kwargs["model"] = self.config.model
        if self._send_temperature:
            try:
                return self._record(send(temperature=0, **kwargs))
            except BadRequestError as exc:
                if "temperature" not in str(exc).lower():
                    raise
                self._send_temperature = False
        return self._record(send(**kwargs))

    def _record(self, completion: Any) -> Any:
        self.usage.calls += 1
        if completion.usage is not None:
            self.usage.prompt_tokens += completion.usage.prompt_tokens or 0
            self.usage.completion_tokens += completion.usage.completion_tokens or 0
        return completion


# --- Helpers -------------------------------------------------------------------------------


def _message(role: str, content: str) -> Message:
    return {"role": role, "content": content}


def _explain(exc: OpenAIError, config: LLMConfig) -> ExtractionFailed:
    """Turn an API error into a message that says what to check."""
    if isinstance(exc, NotFoundError):
        return ExtractionFailed(f"model {config.model!r} not found at {config.base_url}; check LLM_MODEL")
    if isinstance(exc, AuthenticationError):
        return ExtractionFailed(f"{config.base_url} rejected the API key; check LLM_API_KEY")
    if isinstance(exc, LengthFinishReasonError):
        return ExtractionFailed("the reply was cut off before the JSON was complete")
    return ExtractionFailed(f"LLM call failed: {_short(exc)}")


def _short(exc: Exception, limit: int = 200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."
