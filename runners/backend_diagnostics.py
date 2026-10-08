"""
Backend diagnostics and real model preflight verification.

Reports:
- Host OS and Python runtime
- PyTorch and CUDA availability
- vLLM package availability
- OpenAI-compatible HTTP endpoint reachability
- Candidate model registry configuration and backend executability
- Single-model and multi-model preflight execution using production InferenceProvider
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
import urllib.request

from eval.scorer import OutcomeScorer
from eval.scoring_schema import ScoreOutcome
from router.benchmark_schema import BenchmarkPrompt
from router.config import load_model_registry
from runners.inference import load_pricing_config
from runners.outcome_schema import ModelOutcome
from runners.vllm_runner import VLLMProvider


def check_openai_endpoint(endpoint_url: str | None, timeout_seconds: float = 2.0) -> bool:
    """Check if an OpenAI-compatible /models endpoint responds with HTTP 200."""
    if not endpoint_url:
        return False
    try:
        url = f"{endpoint_url.rstrip('/')}/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_backend_diagnostics(
    models_config_path: str | Path = "configs/models.yaml",
    pricing_config_path: str | Path = "configs/pricing.yaml",
) -> dict[str, Any]:
    """
    Collect comprehensive diagnostic information about current execution host and backend.
    """
    models_path = Path(models_config_path)
    pricing_path = Path(pricing_config_path)

    # 1. Python & OS
    py_version = sys.version
    os_name = f"{platform.system()} {platform.release()} ({platform.platform()})"

    # 2. PyTorch & CUDA
    torch_installed = False
    torch_version: str | None = None
    cuda_available = False
    cuda_version: str | None = None
    gpu_names: list[str] = []

    try:
        import torch  # type: ignore
        torch_installed = True
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    except ImportError:
        pass
    except Exception as e:
        torch_version = f"Error during detection: {e}"

    # If torch is not installed or has no cuda, check nvidia-smi for host GPU presence
    if not gpu_names:
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_names = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        except Exception:
            pass

    # 3. vLLM package
    vllm_installed = False
    vllm_version: str | None = None
    try:
        import vllm  # type: ignore
        vllm_installed = True
        vllm_version = getattr(vllm, "__version__", "unknown")
    except ImportError:
        pass
    except Exception as e:
        vllm_version = f"Error during import: {e}"

    # 4. OpenAI-compatible endpoint
    configured_endpoint = os.environ.get("VLLM_API_BASE")
    endpoint_reachable = check_openai_endpoint(configured_endpoint)

    # 5. Model registry & candidate models
    registry_loaded = False
    models_info: list[dict[str, Any]] = []

    if models_path.exists():
        try:
            registry = load_model_registry(models_path)
            pricing_map = load_pricing_config(pricing_path)
            registry_loaded = True
            for m in registry.enabled_models():
                provider = VLLMProvider(
                    model_config=m,
                    pricing=pricing_map.get(m.id),
                    api_base=configured_endpoint,
                )
                is_executable = provider.is_available()
                models_info.append({
                    "id": m.id,
                    "display_name": m.display_name,
                    "backend": m.backend,
                    "model_name": m.model_name,
                    "provider": m.provider,
                    "is_executable": is_executable,
                })
        except Exception as e:
            models_info = [{"error": f"Failed to load registry: {e}"}]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": py_version,
        "platform_os": os_name,
        "torch_installed": torch_installed,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_names": gpu_names,
        "vllm_installed": vllm_installed,
        "vllm_version": vllm_version,
        "openai_endpoint_configured": configured_endpoint,
        "openai_endpoint_reachable": endpoint_reachable,
        "model_registry_loaded": registry_loaded,
        "candidate_models": models_info,
    }


def create_preflight_prompt(
    prompt_text: str = "Return exactly the word: TEST",
    expected_output: str = "TEST",
) -> BenchmarkPrompt:
    """Create a minimal deterministic prompt for preflight contract testing."""
    return BenchmarkPrompt(
        prompt_id="preflight-trivial-001",
        prompt=prompt_text,
        task_type="qa",
        domain="preflight",
        difficulty="easy",
        source_dataset="preflight",
        source_split="preflight",
        source_id="preflight-1",
        split="test",
        expected_output=expected_output,
        evaluation_type="exact_match",
    )


def run_single_model_preflight(
    model_id: str = "qwen3-1.7b",
    models_config_path: str | Path = "configs/models.yaml",
    pricing_config_path: str | Path = "configs/pricing.yaml",
    max_output_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 42,
    prompt_text: str = "Return exactly the word: TEST",
) -> ModelOutcome:
    """
    Execute ONE candidate model against ONE deterministic trivial prompt
    using the production InferenceProvider interface.
    """
    registry = load_model_registry(models_config_path)
    pricing_map = load_pricing_config(pricing_config_path)

    model_config = next((m for m in registry.models if m.id == model_id), None)
    if not model_config:
        raise ValueError(f"Model ID '{model_id}' not found in registry.")

    provider = VLLMProvider(
        model_config=model_config,
        pricing=pricing_map.get(model_config.id),
    )

    prompt = create_preflight_prompt(prompt_text=prompt_text)
    run_id = f"preflight-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    return provider.execute_prompt(
        prompt=prompt,
        run_id=run_id,
        max_retries=1,
        temperature=temperature,
        seed=seed,
    )


def run_three_model_preflight(
    models_config_path: str | Path = "configs/models.yaml",
    pricing_config_path: str | Path = "configs/pricing.yaml",
    prompt_text: str = "Return exactly the word: TEST",
) -> list[ModelOutcome]:
    """
    Execute 3 candidate models against the single deterministic trivial prompt.
    Only called if single-model preflight succeeds.
    """
    outcomes: list[ModelOutcome] = []
    for mid in ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"]:
        outcome = run_single_model_preflight(
            model_id=mid,
            models_config_path=models_config_path,
            pricing_config_path=pricing_config_path,
            prompt_text=prompt_text,
        )
        outcomes.append(outcome)
    return outcomes


def evaluate_preflight_outcome(outcome: ModelOutcome, prompt: BenchmarkPrompt | None = None) -> ScoreOutcome:
    """Pass preflight ModelOutcome through Phase 3 OutcomeScorer."""
    scorer = OutcomeScorer(target_quality=0.80)
    p = prompt or create_preflight_prompt()
    return scorer.score_outcome(outcome, prompt=p)


def print_diagnostics_report(diag: dict[str, Any]) -> None:
    """Print human-readable diagnostic report."""
    print("=" * 60)
    print("BACKEND INFRASTRUCTURE DIAGNOSTICS")
    print("=" * 60)
    print(f"Timestamp (UTC)         : {diag['timestamp_utc']}")
    print(f"Python Version          : {diag['python_version'].splitlines()[0]}")
    print(f"Platform / OS           : {diag['platform_os']}")
    print(f"PyTorch Installed       : {diag['torch_installed']} (version: {diag['torch_version']})")
    print(f"CUDA Available (torch)  : {diag['cuda_available']} (version: {diag['cuda_version']})")
    print(f"Detected GPU(s)         : {', '.join(diag['gpu_names']) if diag['gpu_names'] else 'None detected'}")
    print(f"vLLM Package Installed  : {diag['vllm_installed']} (version: {diag['vllm_version']})")
    print(f"OpenAI Endpoint Env     : {diag['openai_endpoint_configured'] or 'Not set'}")
    print(f"OpenAI Endpoint Status  : {'Reachable' if diag['openai_endpoint_reachable'] else 'Unreachable / Not configured'}")
    print(f"Model Registry Loaded   : {diag['model_registry_loaded']}")
    print("\nCandidate Models Status:")
    for m in diag.get("candidate_models", []):
        if "error" in m:
            print(f"  - ERROR: {m['error']}")
        else:
            exec_str = "EXECUTABLE" if m["is_executable"] else "NOT EXECUTABLE"
            print(f"  - {m['id']} ({m['display_name']}): backend={m['backend']}, target={m['model_name']} -> [{exec_str}]")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Model Backend Preflight & Diagnostics")
    parser.add_argument("--json", action="store_true", help="Output diagnostics in JSON format")
    parser.add_argument("--run-preflight", action="store_true", help="Run contract test on qwen3-1.7b")
    args = parser.parse_args()

    diag = get_backend_diagnostics()

    if args.json and not args.run_preflight:
        print(json.dumps(diag, indent=2))
        return

    print_diagnostics_report(diag)

    if args.run_preflight:
        print("\nRunning single-model preflight test against 'qwen3-1.7b'...")
        outcome = run_single_model_preflight()
        print(f"Outcome Status  : {outcome.status}")
        print(f"Error Type      : {outcome.error_type}")
        print(f"Error Message   : {outcome.error_message}")
        print(f"Latency (ms)    : {outcome.latency_ms}")
        print(f"Prompt Tokens   : {outcome.prompt_tokens}")
        print(f"Output Tokens   : {outcome.output_tokens}")
        print(f"Total Tokens    : {outcome.total_tokens}")
        print(f"Generated Text  : {repr(outcome.generated_text)}")

        score_res = evaluate_preflight_outcome(outcome)
        print(f"\nScorer Integration:")
        print(f"Scorer Status   : {score_res.scoring_status}")
        print(f"Quality Score   : {score_res.quality_score}")
        print(f"Scorer Error    : {score_res.scoring_error_type}: {score_res.scoring_error_message}")


if __name__ == "__main__":
    main()
