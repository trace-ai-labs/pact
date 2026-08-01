"""The only API layer: chat completions against an OpenAI-compatible endpoint.

Defaults to Baseten Model APIs (`BASETEN_API_KEY`). Any other OpenAI-compatible
endpoint works by setting PACT_BASE_URL and PACT_API_KEY_ENV; nothing else in
the package knows or cares which provider is behind them.

Smoke test:
  python -m pact.client --model kimi --prompt "say hi"
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)

try:  # optional; the key may already be in the environment
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
except ImportError:
    pass

BASE_URL = os.environ.get("PACT_BASE_URL", "https://inference.baseten.co/v1")
API_KEY_ENV = os.environ.get("PACT_API_KEY_ENV", "BASETEN_API_KEY")

# Transient failures worth retrying. Auth and validation errors fail fast.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError,
             InternalServerError)

# Short aliases for the models the paper evaluated that are served on Baseten
# Model APIs. Any full slug passes through untouched, so a model that is not
# listed here still runs: pass its slug as it appears on your endpoint.
#
# The paper also evaluated models that are NOT reachable on Model APIs (Claude
# Haiku 4.5, GPT-5.6 Luna, Grok 4.3, Gemini 3 Flash, and several served from
# dedicated deployments). To run one of those, point PACT_BASE_URL and
# PACT_API_KEY_ENV at a provider that serves it and pass its slug.
MODEL_REGISTRY: Dict[str, str] = {
    "deepseek":  "deepseek-ai/DeepSeek-V4-Pro",
    "glm":       "zai-org/GLM-5.2",
    "glm-5.2":   "zai-org/GLM-5.2",
    "glm-5.1":   "zai-org/GLM-5.1",
    "glm-4.7":   "zai-org/GLM-4.7",
    "kimi":      "moonshotai/Kimi-K2.6",
    "kimi-2.5":  "moonshotai/Kimi-K2.5",
    "kimi-code": "moonshotai/Kimi-K2.7-Code",
    "inkling":   "thinkingmachines/inkling",
    "gpt-oss":   "openai/gpt-oss-120b",
    "ultra":     "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
}

# The outcome extractor: it reads a reply and names the option that reply
# settled on. It is a reasoning model, so callers keep its max_tokens generous;
# a tight cap returns an empty completion with finish_reason="length".
DEFAULT_EXTRACT_MODEL = "openai/gpt-oss-120b"

# Models that spend hidden reasoning tokens before the visible answer and so
# need a larger output budget than the default. --max-tokens overrides.
OUTPUT_BUDGET: Dict[str, int] = {
    "deepseek-ai/DeepSeek-V4-Pro": 8192,
    "openai/gpt-oss-120b": 8192,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": 8192,
}
DEFAULT_MAX_TOKENS = 1024


def resolve_model(alias_or_id: str) -> str:
    return MODEL_REGISTRY.get(alias_or_id, alias_or_id)


def output_budget(model: str) -> int:
    return OUTPUT_BUDGET.get(model, DEFAULT_MAX_TOKENS)


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Set {API_KEY_ENV} (export it, or put it in a .env file). "
            f"Requests go to {BASE_URL}; override with PACT_BASE_URL and "
            f"PACT_API_KEY_ENV to use a different OpenAI-compatible provider.")
    return key


_client_lock = threading.Lock()
_clients: Dict[str, OpenAI] = {}


def _get_client(api_key: str) -> OpenAI:
    with _client_lock:
        client = _clients.get(api_key)
        if client is None:
            client = _clients[api_key] = OpenAI(base_url=BASE_URL, api_key=api_key)
        return client


@dataclass
class Request:
    """One chat completion. `id` must be unique and stable across runs."""
    id: str
    model: str
    messages: List[Dict[str, str]]        # includes the system message
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    id: str
    ok: bool
    content: str = ""
    error: str = ""
    model: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Smooth per-process request spacing, no bursts. rpm <= 0 disables."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next)
            self._next = scheduled + self.interval
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


def call_one(req: Request, api_key: str, max_retries: int = 5,
             timeout: float = 300.0) -> Result:
    """One request with exponential-backoff retries on transient failures.
    Fails fast on auth and validation errors. Never raises."""
    client = _get_client(api_key)
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=req.model,
                messages=req.messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                timeout=timeout,
            )
            choice = resp.choices[0]
            content = choice.message.content
            if content is None or not str(content).strip():
                last_err = f"empty completion (finish_reason={choice.finish_reason})"
                time.sleep(2.0 * (attempt + 1))
                continue
            return Result(req.id, True, str(content), "", req.model, req.meta)
        except RateLimitError as e:
            last_err = f"rate limited: {e}"
            time.sleep(min(90.0, 10.0 * (2 ** attempt)))
        except RETRYABLE as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 5.0 * (2 ** attempt)))
        except Exception as e:   # auth, bad request, unknown model: no retry
            return Result(req.id, False, "",
                          f"non-retryable {type(e).__name__}: {e}",
                          req.model, req.meta)
    return Result(req.id, False, "",
                  f"failed after {max_retries} attempts: {last_err}",
                  req.model, req.meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="kimi", type=resolve_model)
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    print(f"endpoint: {BASE_URL}  key: {API_KEY_ENV}  model: {args.model}")
    res = call_one(Request(id="smoke", model=args.model, max_tokens=2048,
                           messages=[{"role": "user", "content": args.prompt}]),
                   get_api_key())
    print(res.content if res.ok else f"FAILED: {res.error}")


if __name__ == "__main__":
    main()
