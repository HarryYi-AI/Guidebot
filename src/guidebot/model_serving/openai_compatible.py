"""Standard-library client for OpenAI-compatible servers such as vLLM."""

from __future__ import annotations

import json
from time import perf_counter
from urllib.request import Request, urlopen

from .models import ModelRequest, ModelResponse


class OpenAICompatibleBackend:
    """Call a `/v1/chat/completions` endpoint without adding an SDK dependency."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def infer(self, request: ModelRequest) -> ModelResponse:
        prompt = str(request.payload.get("prompt") or request.payload.get("text") or request.payload)
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        started = perf_counter()
        with urlopen(http_request, timeout=request.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latency_ms = (perf_counter() - started) * 1_000
        usage = payload.get("usage", {})
        text = payload["choices"][0]["message"]["content"]
        return ModelResponse(
            text=str(text),
            latency_ms=latency_ms,
            input_tokens=_optional_int(usage.get("prompt_tokens")),
            output_tokens=_optional_int(usage.get("completion_tokens")),
            metadata={"backend": self.name, "model": self.model},
        )


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
