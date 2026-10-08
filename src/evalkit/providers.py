"""LLM providers for the judge. Add your own with `register_provider`."""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable[[str, str], str]] = {}


def register_provider(name: str, fn: Callable[[str, str], str]) -> None:
    """fn(model, prompt) -> raw text response."""
    _REGISTRY[name] = fn


def complete(provider: str, model: str, prompt: str) -> str:
    if provider in _REGISTRY:
        return _REGISTRY[provider](model, prompt)
    if provider == "anthropic":
        return _anthropic(model, prompt)
    if provider == "openai":
        return _openai(model, prompt)
    raise ValueError(f"Unknown judge provider '{provider}'. Built-ins: anthropic, openai.")


def _anthropic(model: str, prompt: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install 'evalkit[anthropic]'") from e
    msg = anthropic.Anthropic().messages.create(
        model=model, max_tokens=1024, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _openai(model: str, prompt: str) -> str:
    try:
        import openai
    except ImportError as e:
        raise RuntimeError("pip install 'evalkit[openai]'") from e
    client, msgs = openai.OpenAI(), [{"role": "user", "content": prompt}]
    try:
        r = client.chat.completions.create(model=model, temperature=0, messages=msgs)
    except openai.BadRequestError as e:  # some models only allow the default temperature
        if "temperature" not in str(e):
            raise
        r = client.chat.completions.create(model=model, messages=msgs)
    return r.choices[0].message.content or ""
