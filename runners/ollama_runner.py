"""
Ollama inference provider adapter.

Provides:
- Configurable base URL (default http://127.0.0.1:11434)
- OpenAI-compatible chat completions endpoint with stream=false
- Exact model-tag membership and availability verification
- Full token usage capture (prompt, completion, total) without fabricating zeros
- Client-measured latency tracking
- Explicit generation controls (temperature, seed, max_tokens, stop)
- Structured error handling (ConnectionError, ModelNotFoundError, EmptyOutputError, ReasoningOnlyError, MalformedResponseError)
- Bounded retries and telemetry recording via InferenceProvider
"""

import json
import os
import socket
import time
from typing import Any
import urllib.error
import urllib.request

from router.config import ModelConfig
from runners.inference import GenerationRequest, GenerationResult, InferenceProvider


class OllamaError(Exception):
    """Base exception for Ollama provider errors."""
    pass


class ModelNotFoundError(OllamaError):
    """Raised when a candidate model tag is not installed in Ollama."""
    pass


class MalformedResponseError(OllamaError):
    """Raised when the response from Ollama is malformed or missing expected fields."""
    pass


class EmptyOutputError(OllamaError):
    """Raised when the model returns an empty or whitespace-only response."""
    pass


class ReasoningOnlyError(OllamaError):
    """Raised when the model outputs only thinking/reasoning content without a usable final answer."""
    pass


class BackendConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached."""
    pass


DEFAULT_CANDIDATE_TAG_MAP: dict[str, str] = {
    "qwen3-1.7b": "qwen3:1.7b",
    "qwen3-4b": "qwen3:4b",
    "qwen3-8b": "qwen3:8b",
}


def normalize_ollama_base_url(url: str | None = None) -> str:
    """Resolve and normalize the Ollama base URL."""
    base = url or os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    return base.rstrip("/")


def get_installed_ollama_models(api_base: str = "http://127.0.0.1:11434", timeout_seconds: float = 3.0) -> list[str]:
    """Query Ollama server and return a list of installed model names/tags."""
    norm_base = normalize_ollama_base_url(api_base)
    # Prefer /v1/models if available, fall back to /api/tags
    v1_url = f"{norm_base}/v1/models" if not norm_base.endswith("/v1") else f"{norm_base}/models"
    try:
        req = urllib.request.Request(v1_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
                return [m for m in models if m]
    except Exception:
        pass

    # Fallback to /api/tags
    tags_url = f"{norm_base.removesuffix('/v1')}/api/tags"
    try:
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") or m.get("model") for m in data.get("models", []) if isinstance(m, dict)]
                return [m for m in models if m]
    except Exception:
        pass

    return []


class OllamaProvider(InferenceProvider):
    """
    Adapter for local Ollama inference engine using its OpenAI-compatible HTTP API.
    Enforces deterministic controls, exact model tag membership, and rigorous telemetry.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        pricing: dict[str, float] | None = None,
        api_base: str | None = None,
        tag_mapping: dict[str, str] | None = None,
        endpoint_mode: str = "native",
    ) -> None:
        super().__init__(model_config, pricing=pricing)
        self.api_base = normalize_ollama_base_url(api_base)
        self.backend_name = "ollama"
        self.endpoint_mode = endpoint_mode

        # Determine exact model tag
        mapping = tag_mapping or DEFAULT_CANDIDATE_TAG_MAP
        self.model_tag = mapping.get(model_config.id, model_config.model_name)

        self._available: bool | None = None
        self._unavailable_reason: str | None = None
        self._unavailable_error_type: str | None = None
        self._installed_models_cache: list[str] | None = None

    @property
    def chat_completions_url(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/chat/completions"
        return f"{self.api_base}/v1/chat/completions"

    @property
    def native_chat_url(self) -> str:
        base = self.api_base.removesuffix("/v1")
        return f"{base}/api/chat"

    @property
    def models_url(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/models"
        return f"{self.api_base}/v1/models"

    def is_available(self, refresh: bool = False) -> bool:
        """
        Check if the Ollama server is reachable AND the configured model tag is installed.
        Returns False with a structured reason if either condition fails.
        """
        if self._available is not None and not refresh:
            return self._available

        try:
            req = urllib.request.Request(self.models_url, method="GET")
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status != 200:
                    self._available = False
                    self._unavailable_error_type = "BackendConnectionError"
                    self._unavailable_reason = f"Ollama endpoint {self.models_url} returned HTTP {resp.status}."
                    return False
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            self._available = False
            self._unavailable_error_type = "ConnectionError"
            self._unavailable_reason = f"Ollama server is unreachable at {self.api_base}: {e}"
            return False
        except Exception as e:
            self._available = False
            self._unavailable_error_type = "MalformedResponseError"
            self._unavailable_reason = f"Failed to parse models response from Ollama: {e}"
            return False

        # Extract model tags from /v1/models response
        model_entries = data.get("data", [])
        installed_tags: list[str] = []
        if isinstance(model_entries, list):
            for entry in model_entries:
                if isinstance(entry, dict) and "id" in entry:
                    installed_tags.append(entry["id"])

        self._installed_models_cache = installed_tags

        # Verify exact or normalized tag presence
        target = self.model_tag.lower()
        matched = any(
            t.lower() == target
            or t.lower().split(":")[0] == target.split(":")[0] and (":" in target and t.lower() == target)
            for t in installed_tags
        )

        if matched or self.model_tag in installed_tags:
            self._available = True
            self._unavailable_reason = None
            self._unavailable_error_type = None
            return True

        self._available = False
        self._unavailable_error_type = "ModelNotFoundError"
        self._unavailable_reason = (
            f"Candidate model tag '{self.model_tag}' is not installed in Ollama. "
            f"Installed tags: {installed_tags or 'none'}."
        )
        return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Execute request against Ollama.
        Uses native /api/chat by default (supporting think mode toggling, options.num_predict,
        and done_reason parsing) or /v1/chat/completions when endpoint_mode='openai'.
        """
        if not self.is_available():
            err_cls = ModelNotFoundError if self._unavailable_error_type == "ModelNotFoundError" else RuntimeError
            raise err_cls(self._unavailable_reason or f"Ollama backend unavailable for model '{self.model_tag}'.")

        use_native = getattr(self, "endpoint_mode", "native") == "native"

        if use_native:
            url = self.native_chat_url
            payload: dict[str, Any] = {
                "model": self.model_tag,
                "messages": [{"role": "user", "content": request.prompt}],
                "stream": False,
            }
            if request.think is not None:
                payload["think"] = request.think
            options: dict[str, Any] = {
                "num_predict": request.max_output_tokens,
                "temperature": request.temperature,
            }
            if request.seed is not None:
                options["seed"] = request.seed
            if request.stop:
                options["stop"] = request.stop
            payload["options"] = options
        else:
            url = self.chat_completions_url
            payload = {
                "model": self.model_tag,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_output_tokens,
                "temperature": request.temperature,
                "seed": request.seed,
                "stream": False,
            }
            if request.think is not None:
                payload["think"] = request.think
            if request.stop:
                payload["stop"] = request.stop

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=request.timeout_seconds) as resp:
                resp_bytes = resp.read()
                data = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            if e.code == 404:
                raise ModelNotFoundError(f"Ollama reported model '{self.model_tag}' not found (HTTP 404): {err_body}") from e
            raise RuntimeError(f"Ollama HTTP error {e.code}: {e.reason}. Body: {err_body}") from e
        except (socket.timeout, TimeoutError) as e:
            raise TimeoutError(f"Ollama inference request timed out after {request.timeout_seconds}s: {e}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout) or "timed out" in str(e.reason).lower():
                raise TimeoutError(f"Ollama inference request timed out: {e.reason}") from e
            raise ConnectionError(f"Connection to Ollama failed at {url}: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise MalformedResponseError(f"Ollama response could not be parsed as JSON: {e}") from e

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if not isinstance(data, dict):
            raise MalformedResponseError(f"Expected dict response from Ollama, got {type(data).__name__}")

        if "message" in data and isinstance(data["message"], dict):
            # Parse native Ollama /api/chat response
            msg = data["message"]
            gen_text = msg.get("content", "")
            reasoning_content = msg.get("thinking")
            done_reason = data.get("done_reason")
            finish_reason = "length" if done_reason == "length" else ("stop" if done_reason == "stop" else done_reason)
            is_truncated = (done_reason == "length")

            p_tokens = data.get("prompt_eval_count") if isinstance(data.get("prompt_eval_count"), int) else None
            o_tokens = data.get("eval_count") if isinstance(data.get("eval_count"), int) else None
            tot_tokens = (p_tokens + o_tokens) if (p_tokens is not None and o_tokens is not None) else None
            reasoning_tokens = len(reasoning_content.split()) if reasoning_content else (0 if request.think is False else None)

        elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            # Parse OpenAI-compatible /v1/chat/completions response
            first_choice = data["choices"][0]
            if not isinstance(first_choice, dict):
                raise MalformedResponseError(f"Choice entry is not a dict: {first_choice}")

            msg = first_choice.get("message")
            if not isinstance(msg, dict):
                raise MalformedResponseError(f"Choice missing 'message' object: {first_choice}")

            gen_text = msg.get("content", "")
            reasoning_content = msg.get("reasoning_content") or msg.get("reasoning")
            finish_reason = first_choice.get("finish_reason")
            is_truncated = (finish_reason == "length")

            usage = data.get("usage")
            p_tokens = None
            o_tokens = None
            tot_tokens = None
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int):
                    p_tokens = usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int):
                    o_tokens = usage["completion_tokens"]
                if isinstance(usage.get("total_tokens"), int):
                    tot_tokens = usage["total_tokens"]
                elif p_tokens is not None and o_tokens is not None:
                    tot_tokens = p_tokens + o_tokens
            reasoning_tokens = None
        else:
            raise MalformedResponseError(f"Ollama response missing both 'message' and 'choices': {data}")

        # Validate usable output: do not treat empty or reasoning-only responses as success
        if not isinstance(gen_text, str) or not gen_text.strip():
            if reasoning_content and isinstance(reasoning_content, str) and reasoning_content.strip():
                raise ReasoningOnlyError("Model returned reasoning_content but empty final content.")
            raise EmptyOutputError("Model returned empty or whitespace-only output.")

        # Check for thinking-only content wrapped in <think>...</think>
        stripped_text = gen_text.strip()
        if stripped_text.startswith("<think>") and "</think>" in stripped_text:
            post_think = stripped_text.split("</think>", 1)[-1].strip()
            if not post_think:
                raise ReasoningOnlyError("Model output contains only <think> reasoning without a usable answer.")

        tokens_per_second: float | None = None
        if o_tokens is not None and latency_ms > 0.0:
            tokens_per_second = round(o_tokens / (latency_ms / 1000.0), 2)

        return GenerationResult(
            generated_text=gen_text,
            finish_reason=finish_reason,
            prompt_tokens=p_tokens,
            output_tokens=o_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=tot_tokens,
            is_truncated=is_truncated,
            latency_ms=round(latency_ms, 2),
            tokens_per_second=tokens_per_second,
            engine_version=f"ollama-{data.get('model', self.model_tag)}",
            raw_response=data,
        )
