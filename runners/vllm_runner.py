import os
import time
from typing import Any
import urllib.request
import json

from router.config import ModelConfig
from runners.inference import GenerationRequest, GenerationResult, InferenceProvider


class VLLMProvider(InferenceProvider):
    """
    Adapter for local vLLM inference engine.
    Obtains model configurations, limits, and capabilities from configs/models.yaml.
    Supports in-process vLLM engine if installed and GPU-equipped, or vLLM HTTP OpenAI-compatible server.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        pricing: dict[str, float] | None = None,
        api_base: str | None = None,
    ) -> None:
        super().__init__(model_config, pricing=pricing)
        self.api_base = api_base or os.environ.get("VLLM_API_BASE")
        self._llm: Any = None
        self._vllm_available: bool | None = None

    def is_available(self) -> bool:
        """Check if local vLLM module or vLLM HTTP server is reachable."""
        if self._vllm_available is not None:
            return self._vllm_available

        # 1. Check HTTP endpoint if configured
        if self.api_base:
            try:
                req = urllib.request.Request(f"{self.api_base.rstrip('/')}/models", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        self._vllm_available = True
                        return True
            except Exception:
                pass

        # 2. Check in-process vLLM module
        try:
            import vllm  # type: ignore # noqa: F401
            import torch # type: ignore # noqa: F401
            if torch.cuda.is_available():
                self._vllm_available = True
                return True
        except ImportError:
            pass
        except Exception:
            pass

        self._vllm_available = False
        return False

    def _get_in_process_llm(self) -> Any:
        if self._llm is None:
            from vllm import LLM  # type: ignore
            self._llm = LLM(
                model=self.model_config.model_name,
                max_model_len=self.model_config.limits.context_window_tokens,
                trust_remote_code=True,
            )
        return self._llm

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute request using in-process vLLM or HTTP API."""
        if not self.is_available():
            raise RuntimeError(
                f"vLLM backend is not available for model '{self.model_config.id}'."
            )

        t_start = time.perf_counter()

        # Path A: HTTP API endpoint
        if self.api_base:
            return self._generate_http(request, t_start)

        # Path B: In-process vLLM
        return self._generate_in_process(request, t_start)

    def _generate_http(self, request: GenerationRequest, t_start: float) -> GenerationResult:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_config.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
        }
        if request.stop:
            payload["stop"] = request.stop

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=request.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"vLLM HTTP request failed: {e}") from e

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        gen_text = msg.get("content", "")
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage", {})

        p_tokens = usage.get("prompt_tokens")
        o_tokens = usage.get("completion_tokens")
        tot_tokens = usage.get("total_tokens")

        return GenerationResult(
            generated_text=gen_text,
            finish_reason=finish_reason,
            prompt_tokens=p_tokens,
            output_tokens=o_tokens,
            total_tokens=tot_tokens,
            latency_ms=round(latency_ms, 2),
            raw_response=data,
        )

    def _generate_in_process(self, request: GenerationRequest, t_start: float) -> GenerationResult:
        from vllm import SamplingParams  # type: ignore

        llm = self._get_in_process_llm()
        params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            seed=request.seed,
            stop=request.stop,
        )

        outputs = llm.generate([request.prompt], params)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        first_out = outputs[0]
        gen_text = first_out.outputs[0].text if first_out.outputs else ""
        finish_reason = first_out.outputs[0].finish_reason if first_out.outputs else None
        p_tokens = len(first_out.prompt_token_ids) if hasattr(first_out, "prompt_token_ids") else None
        o_tokens = len(first_out.outputs[0].token_ids) if first_out.outputs else None
        tot_tokens = (p_tokens + o_tokens) if (p_tokens and o_tokens) else None

        return GenerationResult(
            generated_text=gen_text,
            finish_reason=finish_reason,
            prompt_tokens=p_tokens,
            output_tokens=o_tokens,
            total_tokens=tot_tokens,
            latency_ms=round(latency_ms, 2),
            engine_version="vllm-local",
        )


class MockInferenceProvider(InferenceProvider):
    """
    Mock inference provider for testing and contract validation.
    Generates deterministic responses with realistic telemetry and error injection capabilities.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        pricing: dict[str, float] | None = None,
        should_fail_attempts: int = 0,
        fail_error_type: type[Exception] = RuntimeError,
        simulate_timeout: bool = False,
        fixed_latency_ms: float = 45.0,
        fixed_output: str | None = None,
    ) -> None:
        super().__init__(model_config, pricing=pricing)
        self.should_fail_attempts = should_fail_attempts
        self.fail_error_type = fail_error_type
        self.simulate_timeout = simulate_timeout
        self.fixed_latency_ms = fixed_latency_ms
        self.fixed_output = fixed_output
        self.attempts_seen = 0

    def is_available(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.attempts_seen += 1

        if self.simulate_timeout:
            raise TimeoutError("Simulated request timeout.")

        if self.attempts_seen <= self.should_fail_attempts:
            raise self.fail_error_type(f"Simulated failure on attempt {self.attempts_seen}")

        # Deterministic token count and text based on prompt
        words = request.prompt.split()
        p_tokens = max(1, len(words) * 2)

        if self.fixed_output is not None:
            text = self.fixed_output
        else:
            text = f"Mock response from {self.model_config.id} for prompt of length {len(request.prompt)}."

        o_tokens = max(1, len(text.split()) * 2)
        tot_tokens = p_tokens + o_tokens

        return GenerationResult(
            generated_text=text,
            finish_reason="stop",
            prompt_tokens=p_tokens,
            output_tokens=o_tokens,
            total_tokens=tot_tokens,
            latency_ms=self.fixed_latency_ms,
            time_to_first_token_ms=round(self.fixed_latency_ms * 0.25, 2),
            tokens_per_second=round(o_tokens / (self.fixed_latency_ms / 1000.0), 2),
            engine_version="mock-1.0",
        )
