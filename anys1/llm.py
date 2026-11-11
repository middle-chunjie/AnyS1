"""OpenAI-compatible System-2 LLM wrapper for AnyS1 compilation and fallback."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI


@dataclass
class LLMCallRecord:
    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class System2Client:
    """Thin, deterministic wrapper around any OpenAI-compatible System-2 endpoint.

    Configuration is read from standard environment variables:
    - ``OPENAI_API_KEY`` (or ``ANYS1_S2_API_KEY``)
    - ``OPENAI_BASE_URL`` (or ``ANYS1_S2_BASE_URL``, optional)
    - ``ANYS1_S2_MODEL`` (default: ``deepseek-chat``)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model or os.environ.get("ANYS1_S2_MODEL", "deepseek-chat")
        self.api_key = (
            api_key
            or os.environ.get("ANYS1_S2_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.base_url = (
            base_url
            or os.environ.get("ANYS1_S2_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.ledger: list[LLMCallRecord] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        role: str = "s2",
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0

        usage = getattr(resp, "usage", None)
        self.ledger.append(
            LLMCallRecord(
                role=role,
                model=self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=round(dt, 4),
            )
        )
        return (resp.choices[0].message.content or "").strip()

    def decide_option(
        self,
        state: str,
        question: str,
        options: list[str],
        *,
        role: str = "s2_enum",
    ) -> tuple[int, str]:
        """Ask System-2 to choose one option from a fixed candidate set."""
        letters = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(f"{L}. {opt}" for L, opt in zip(letters, options))
        prompt = (
            f"### Current Execution State\n{state}\n\n"
            f"### Decision Question\n{question}\n\n"
            f"### Candidate Options\n{opt_lines}\n\n"
            "Respond with a JSON object containing `\"choice\"` (the option letter, e.g. \"A\") "
            "and `\"reason\"` (one concise sentence)."
        )
        raw = self.complete(prompt, role=role, temperature=0.0, max_tokens=256)
        idx = _parse_choice_index(raw, len(options))
        return idx, options[idx]

    @property
    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.ledger)


def _parse_choice_index(raw: str, n_options: int) -> int:
    text = raw.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(text[start : end + 1])
            ch = str(obj.get("choice", "")).strip().upper()
            if len(ch) == 1 and "A" <= ch < chr(ord("A") + n_options):
                return ord(ch) - ord("A")
    except Exception:
        pass
    for ch in text.upper():
        if "A" <= ch < chr(ord("A") + n_options):
            return ord(ch) - ord("A")
    return 0
