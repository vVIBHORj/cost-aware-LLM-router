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
from runners.outcome_schema import ModelOutcome, RunManifest
from runners.storage import save_outcomes_parquet, save_run_manifest
from runners.vllm_runner import MockInferenceProvider, VLLMProvider


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def select_smoke_prompts(
    processed_dir: Path = Path("data/processed"),
    prompts_per_split: int = 5,
) -> list[BenchmarkPrompt]:
    """Select exactly `prompts_per_split` prompts from each of train, validation, test, and stress."""
    splits = ["train", "validation", "test", "stress"]
    selected: list[BenchmarkPrompt] = []

    for s in splits:
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
    models_config_path: Path = Path("configs/models.yaml"),
    pricing_config_path: Path = Path("configs/pricing.yaml"),
    processed_dir: Path = Path("data/processed"),
) -> tuple[list[ModelOutcome], RunManifest]:
    """
    Execute 20 prompts × 3 models = 60 evaluation smoke matrix.
    Writes smoke_outcomes.parquet and smoke_manifest.json under output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"smoke-run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # 1. Load registry and pricing
    registry = load_model_registry(models_config_path)
    pricing_map = load_pricing_config(pricing_config_path)
    models = registry.enabled_models()

    if len(models) != 3:
        raise ValueError(f"Expected exactly 3 enabled candidate models, found {len(models)}")

    # 2. Select 20 smoke prompts (5 per split)
    prompts = select_smoke_prompts(processed_dir, prompts_per_split=5)
    if len(prompts) != 20:
        raise ValueError(f"Expected 20 smoke prompts, got {len(prompts)}")

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
        else:
            providers[m.id] = VLLMProvider(model_config=m, pricing=m_pricing)

    # 5. Execute 20 × 3 = 60 evaluations
    outcomes: list[ModelOutcome] = []
    per_model_counts: dict[str, dict[str, int]] = {
        m.id: {"total": 0, "success": 0, "failed": 0, "retries": 0} for m in models
    }
    per_split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0, "stress": 0}

    for prompt in prompts:
        per_split_counts[prompt.split] += 1
        for m in models:
            provider = providers[m.id]
            outcome = provider.execute_prompt(
                prompt,
                run_id=run_id,
                max_retries=3,
                temperature=0.0,
                seed=42,
            )
            outcomes.append(outcome)

            per_model_counts[m.id]["total"] += 1
            if outcome.status == "SUCCESS":
                per_model_counts[m.id]["success"] += 1
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

    return outcomes, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate Model Inference Smoke Test")
    parser.add_argument("--mock", action="store_true", help="Use mock provider for contract testing")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID")
    parser.add_argument("--output-dir", type=str, default="data/outcomes/smoke", help="Output directory")
    args = parser.parse_args()

    outcomes, manifest = run_smoke_test(
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        use_mock=args.mock,
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
