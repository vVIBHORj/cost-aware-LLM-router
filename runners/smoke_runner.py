from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any
import json
import argparse

from router.benchmark_loader import load_benchmark_jsonl
from router.benchmark_schema import BenchmarkPrompt
from router.config import load_model_registry
from runners.inference import load_pricing_config
from runners.ollama_runner import OllamaProvider
from runners.outcome_schema import ModelOutcome, RunManifest
from runners.storage import save_outcomes_parquet, save_run_manifest
from runners.vllm_runner import MockInferenceProvider, VLLMProvider
from eval.scorer import OutcomeScorer
from eval.storage import save_scores_parquet


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def select_smoke_prompts(
    processed_dir: Path = Path("data/processed"),
    prompts_per_split: int = 5,
    splits: list[str] | None = None,
) -> list[BenchmarkPrompt]:
    """Select exactly `prompts_per_split` prompts from each specified split."""
    target_splits = splits or ["train", "validation", "test", "stress"]
    selected: list[BenchmarkPrompt] = []

    for s in target_splits:
        p_path = processed_dir / f"benchmark_{s}.jsonl"
        items = load_benchmark_jsonl(p_path)
        if len(items) < prompts_per_split:
            raise ValueError(f"Insufficient prompts in {p_path}: {len(items)} < {prompts_per_split}")
        selected.extend(items[:prompts_per_split])

    return selected


def run_smoke_test(
    output_dir: Path = Path("data/outcomes/smoke"),
    run_id: str | None = None,
    use_mock: bool = False,
    backend: str | None = None,
    models_config_path: Path = Path("configs/models.yaml"),
    pricing_config_path: Path = Path("configs/pricing.yaml"),
    processed_dir: Path = Path("data/processed"),
    prompts_per_split: int = 5,
    splits: list[str] | None = None,
    score_outcomes: bool = False,
    timeout_seconds: float = 120.0,
    think: bool | None = None,
    max_output_tokens: int | None = None,
    model_ids: list[str] | None = None,
) -> tuple[list[ModelOutcome], RunManifest]:
    """
    Execute smoke matrix evaluations.
    Writes smoke_outcomes.parquet, smoke_manifest.json (and optional smoke_scores.parquet) under output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"smoke-run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # 1. Load registry and pricing
    registry = load_model_registry(models_config_path)
    pricing_map = load_pricing_config(pricing_config_path)
    models = registry.enabled_models()

    if model_ids:
        models = [m for m in models if m.id in model_ids]
        if not models:
            raise ValueError(f"No enabled models matched requested IDs: {model_ids}")
    elif len(models) != 3:
        raise ValueError(f"Expected exactly 3 enabled candidate models, found {len(models)}")

    # 2. Select smoke prompts
    prompts = select_smoke_prompts(processed_dir, prompts_per_split=prompts_per_split, splits=splits)
    target_splits = splits or ["train", "validation", "test", "stress"]

    # 3. Benchmark manifest checksum
    bench_manifest_path = processed_dir / "benchmark_manifest.json"
    bench_checksum = compute_file_sha256(bench_manifest_path) if bench_manifest_path.exists() else "unknown"
    models_checksum = compute_file_sha256(models_config_path)

    # 4. Instantiate providers
    providers = {}
    for m in models:
        m_pricing = pricing_map.get(m.id)
        if use_mock:
            providers[m.id] = MockInferenceProvider(model_config=m, pricing=m_pricing)
        elif (backend or m.backend).lower() == "ollama":
            providers[m.id] = OllamaProvider(model_config=m, pricing=m_pricing)
        else:
            providers[m.id] = VLLMProvider(model_config=m, pricing=m_pricing)

    # 5. Execute evaluations (grouped by model to minimize Ollama weights thrashing)
    outcomes: list[ModelOutcome] = []
    per_model_counts: dict[str, dict[str, int]] = {
        m.id: {"total": 0, "success": 0, "failed": 0, "retries": 0, "truncated": 0} for m in models
    }
    per_split_counts: dict[str, int] = {s: 0 for s in target_splits}
    for prompt in prompts:
        per_split_counts[prompt.split] += 1

    for m in models:
        provider = providers[m.id]
        for prompt in prompts:
            outcome = provider.execute_prompt(
                prompt,
                run_id=run_id,
                max_retries=3,
                temperature=0.0,
                seed=42,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                think=think,
            )
            outcomes.append(outcome)

            per_model_counts[m.id]["total"] += 1
            if outcome.status == "SUCCESS":
                per_model_counts[m.id]["success"] += 1
            elif outcome.status == "TRUNCATED":
                per_model_counts[m.id]["truncated"] += 1
                per_model_counts[m.id]["failed"] += 1
            else:
                per_model_counts[m.id]["failed"] += 1
            per_model_counts[m.id]["retries"] += outcome.retry_count


    expected_rows = len(prompts) * len(models)  # 20 * 3 = 60
    success_count = sum(1 for o in outcomes if o.status == "SUCCESS")
    failure_count = sum(1 for o in outcomes if o.status != "SUCCESS")
    total_retries = sum(o.retry_count for o in outcomes)

    # Telemetry availability
    token_accounting_avail = any(o.prompt_tokens is not None for o in outcomes)
    latency_avail = any(o.latency_ms > 0.0 for o in outcomes)

    manifest = RunManifest(
        run_id=run_id,
        benchmark_manifest_checksum=bench_checksum,
        models_config_version=f"sha256:{models_checksum}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        number_of_prompts=len(prompts),
        number_of_candidate_models=len(models),
        expected_outcome_rows=expected_rows,
        actual_outcome_rows=len(outcomes),
        success_count=success_count,
        failure_count=failure_count,
        retry_count=total_retries,
        per_model_counts=per_model_counts,
        per_split_counts=per_split_counts,
        token_accounting_available=token_accounting_avail,
        latency_available=latency_avail,
    )

    # 6. Save outcomes and manifest
    parquet_path = output_dir / "smoke_outcomes.parquet"
    manifest_path = output_dir / "smoke_manifest.json"

    save_outcomes_parquet(outcomes, parquet_path)
    save_run_manifest(manifest, manifest_path)

    # 7. Optional scoring
    if score_outcomes:
        prompt_map = {p.prompt_id: p for p in prompts}
        scorer = OutcomeScorer(benchmark_manifest_checksum=bench_checksum)
        scores = []
        for outcome in outcomes:
            p = prompt_map.get(outcome.prompt_id)
            score = scorer.score_outcome(outcome, prompt=p)
            scores.append(score)
        scores_path = output_dir / "smoke_scores.parquet"
        save_scores_parquet(scores, scores_path)

    return outcomes, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate Model Inference Smoke Test")
    parser.add_argument("--mock", action="store_true", help="Use mock provider for contract testing")
    parser.add_argument("--backend", type=str, default=None, help="Inference backend override (e.g. ollama, vllm)")
    parser.add_argument("--splits", type=str, default=None, help="Comma-separated list of splits (e.g. train)")
    parser.add_argument("--prompts-per-split", type=int, default=5, help="Number of prompts per split")
    parser.add_argument("--score", action="store_true", help="Score outcomes after inference")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-prompt timeout in seconds")
    parser.add_argument("--think", action="store_true", default=None, help="Explicitly enable reasoning / thinking mode")
    parser.add_argument("--no-think", action="store_false", dest="think", help="Explicitly disable reasoning / thinking mode")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Explicit maximum output tokens override")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of candidate model IDs")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID")
    parser.add_argument("--output-dir", type=str, default="data/outcomes/smoke", help="Output directory")
    args = parser.parse_args()

    split_list = [s.strip() for s in args.splits.split(",")] if args.splits else None
    model_list = [m.strip() for m in args.models.split(",")] if args.models else None

    outcomes, manifest = run_smoke_test(
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        use_mock=args.mock,
        backend=args.backend,
        prompts_per_split=args.prompts_per_split,
        splits=split_list,
        score_outcomes=args.score,
        timeout_seconds=args.timeout,
        think=args.think,
        max_output_tokens=args.max_output_tokens,
        model_ids=model_list,
    )

    print("Smoke test completed.")
    print(f"Run ID: {manifest.run_id}")
    print(f"Expected Outcomes: {manifest.expected_outcome_rows}")
    print(f"Actual Outcomes: {manifest.actual_outcome_rows}")
    print(f"Successes: {manifest.success_count}")
    print(f"Failures: {manifest.failure_count}")
    print(f"Per-Model Summary: {json.dumps(manifest.per_model_counts, indent=2)}")
    print(f"Token Accounting Available: {manifest.token_accounting_available}")
    print(f"Latency Telemetry Available: {manifest.latency_available}")


if __name__ == "__main__":
    main()

