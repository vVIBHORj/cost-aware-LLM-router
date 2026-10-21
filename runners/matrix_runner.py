"""
Matrix Runner for Cost-Aware LLM Router.
Provides durable, crash-safe, resumable execution of benchmark matrix evaluations.
Guarantees incremental persistence, deterministic deduplication with attempt-history preservation,
experiment configuration fingerprinting, failure preservation, and safe recovery.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from eval.scorer import OutcomeScorer
from eval.scoring_schema import ScoreOutcome
from eval.storage import load_scores_parquet, save_scores_parquet
from router.benchmark_loader import load_benchmark_jsonl
from router.benchmark_schema import BenchmarkPrompt
from router.config import ModelConfig, load_model_registry
from runners.inference import InferenceProvider, load_pricing_config
from runners.inference_policy import (
    DEFAULT_TASK_OUTPUT_BUDGETS,
    DEFAULT_THINKING_POLICY,
    PROMPT_POLICY_V0,
    PROMPT_POLICY_V1_CONCISE,
    PROMPT_POLICY_V2_ANSWER_ONLY,
    PROMPT_POLICY_V3_FEW_SHOT,
    PROMPT_POLICY_V4_STRUCTURED_OUTPUT,
    resolve_response_format,
)
from runners.ollama_runner import OllamaProvider
from runners.outcome_schema import ModelOutcome, RunManifest
from runners.storage import (
    load_outcomes_parquet,
    load_run_manifest,
    save_outcomes_parquet,
    save_run_manifest,
)
from runners.vllm_runner import MockInferenceProvider, VLLMProvider

logger = logging.getLogger(__name__)

SUPPORTED_PROMPT_POLICIES = {
    PROMPT_POLICY_V0,
    PROMPT_POLICY_V1_CONCISE,
    PROMPT_POLICY_V2_ANSWER_ONLY,
    PROMPT_POLICY_V3_FEW_SHOT,
    PROMPT_POLICY_V4_STRUCTURED_OUTPUT,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MatrixRunnerError(Exception):
    """Base exception for matrix runner errors."""
    pass


class ConfigurationMismatchError(MatrixRunnerError):
    """Raised when an existing run's manifest does not match the current run configuration."""
    pass


class CorruptedArtifactError(MatrixRunnerError):
    """Raised when an existing artifact file is unreadable and cannot be recovered from backup."""
    pass


# ---------------------------------------------------------------------------
# Attempt Identity & Planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedAttempt:
    """Canonical representation of an expected (prompt, candidate_model) evaluation."""
    prompt_id: str
    model_id: str
    prompt: BenchmarkPrompt
    model_config: ModelConfig

    @property
    def key(self) -> tuple[str, str]:
        return (self.prompt_id, self.model_id)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a local file."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_experiment_fingerprint(
    benchmark_split: str,
    split_file_sha256: str,
    benchmark_manifest_sha256: str,
    model_ids: list[str],
    model_tags: dict[str, str],
    models_config_sha256: str,
    prompt_policy_version: str,
    output_budget: int | None,
    think: bool | None,
    temperature: float,
    seed: int,
    backend: str,
    response_schema: dict[str, Any] | None,
    target_quality: float,
) -> tuple[str, dict[str, Any]]:
    """
    Compute a deterministic cryptographic fingerprint of the entire execution configuration.
    Includes benchmark split hash, candidate models & tags, prompt policy, budget,
    sampling parameters, backend, schema, and scoring configuration.
    Returns (sha256_hex_digest, canonical_config_dict).
    """
    canonical_dict: dict[str, Any] = {
        "backend": backend.lower(),
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "benchmark_split": benchmark_split,
        "benchmark_split_file_sha256": split_file_sha256,
        "candidate_models": sorted(model_ids),
        "candidate_model_tags": {m: model_tags.get(m, "") for m in sorted(model_ids)},
        "models_config_sha256": models_config_sha256,
        "outcome_schema_version": "1.0.0",
        "output_budget": output_budget,
        "prompt_policy_version": prompt_policy_version,
        "response_schema": response_schema,
        "scorer_name": "OutcomeScorer",
        "scorer_version": "1.0.0",
        "seed": seed,
        "target_quality": target_quality,
        "temperature": temperature,
        "think": think,
    }
    serialized = json.dumps(canonical_dict, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest, canonical_dict


def plan_matrix_attempts(
    prompts: list[BenchmarkPrompt],
    models: list[ModelConfig],
) -> list[PlannedAttempt]:
    """
    Generate deterministic list of planned attempts.
    Attempts are grouped by model first to minimize local backend weight swapping,
    preserving prompt order within each model partition.
    """
    planned: list[PlannedAttempt] = []
    for model in models:
        for prompt in prompts:
            planned.append(
                PlannedAttempt(
                    prompt_id=prompt.prompt_id,
                    model_id=model.id,
                    prompt=prompt,
                    model_config=model,
                )
            )
    return planned


# ---------------------------------------------------------------------------
# Deterministic Deduplication & History Preservation
# ---------------------------------------------------------------------------

def deduplicate_outcomes(
    outcomes: list[ModelOutcome],
) -> tuple[list[ModelOutcome], int]:
    """
    Deterministically deduplicate outcomes by canonical key (prompt_id, model_id).
    Resolution rule:
      1. Prefer SUCCESS over non-SUCCESS (FAILED, TRUNCATED, etc.).
      2. If both have identical status class, prefer the later request_timestamp.
    Crucial Guarantee:
      A successful retry or winner NEVER erases evidence that an earlier attempt failed.
      The loser's telemetry (timestamp, status, latency, error_message) is merged into
      the winner's attempt_history, and retry_count is incremented cumulatively.
    Returns (deduplicated_list, duplicate_count).
    """
    seen: dict[tuple[str, str], ModelOutcome] = {}
    duplicates = 0

    for outcome in outcomes:
        key = (outcome.prompt_id, outcome.model_id)
        if key in seen:
            duplicates += 1
            existing = seen[key]

            # Merge attempt history from both records
            merged_history = list(existing.attempt_history)
            existing_record = {
                "timestamp": existing.request_timestamp,
                "status": existing.status,
                "latency_ms": existing.latency_ms,
                "error_type": existing.error_type,
                "error_message": existing.error_message,
                "prompt_tokens": existing.prompt_tokens,
                "output_tokens": existing.output_tokens,
                "run_id": existing.run_id,
            }
            if not any(h.get("timestamp") == existing.request_timestamp for h in merged_history):
                merged_history.append(existing_record)

            for h in outcome.attempt_history:
                if h not in merged_history:
                    merged_history.append(h)

            outcome_record = {
                "timestamp": outcome.request_timestamp,
                "status": outcome.status,
                "latency_ms": outcome.latency_ms,
                "error_type": outcome.error_type,
                "error_message": outcome.error_message,
                "prompt_tokens": outcome.prompt_tokens,
                "output_tokens": outcome.output_tokens,
                "run_id": outcome.run_id,
            }
            if not any(h.get("timestamp") == outcome.request_timestamp for h in merged_history):
                merged_history.append(outcome_record)

            # Determine primary winner
            if existing.status != "SUCCESS" and outcome.status == "SUCCESS":
                winner = outcome
            elif existing.status == "SUCCESS" and outcome.status != "SUCCESS":
                winner = existing
            elif outcome.request_timestamp > existing.request_timestamp:
                winner = outcome
            else:
                winner = existing

            total_retries = existing.retry_count + outcome.retry_count + 1
            winner_with_history = winner.model_copy(
                update={
                    "attempt_history": merged_history,
                    "retry_count": total_retries,
                }
            )
            seen[key] = winner_with_history
        else:
            seen[key] = outcome

    return list(seen.values()), duplicates


def deduplicate_scores(
    scores: list[ScoreOutcome],
) -> tuple[list[ScoreOutcome], int]:
    """
    Deterministically deduplicate scores by canonical key (prompt_id, model_id).
    Resolution rule:
      1. Prefer SUCCESS over non-SUCCESS.
      2. If both have identical status, prefer later scoring_timestamp.
    Returns (deduplicated_list, duplicate_count).
    """
    seen: dict[tuple[str, str], ScoreOutcome] = {}
    duplicates = 0

    for score in scores:
        key = (score.prompt_id, score.model_id)
        if key in seen:
            duplicates += 1
            existing = seen[key]
            if existing.scoring_status != "SUCCESS" and score.scoring_status == "SUCCESS":
                seen[key] = score
            elif existing.scoring_status == "SUCCESS" and score.scoring_status != "SUCCESS":
                pass
            elif score.scoring_timestamp > existing.scoring_timestamp:
                seen[key] = score
        else:
            seen[key] = score

    return list(seen.values()), duplicates


# ---------------------------------------------------------------------------
# Crash-Safe Atomic Persistence & Recovery
# ---------------------------------------------------------------------------

def safe_load_outcomes(path: Path) -> list[ModelOutcome]:
    """
    Load ModelOutcome records from Parquet with automatic recovery from backup (.bak)
    or partial write (.tmp) if the primary file is corrupted.
    Raises CorruptedArtifactError if the file exists but cannot be read or recovered.
    """
    bak_path = path.with_suffix(path.suffix + ".bak")
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    if path.exists():
        try:
            return load_outcomes_parquet(path)
        except Exception as exc:
            logger.warning(
                f"Primary outcome artifact at {path} is corrupted: {exc}. "
                f"Attempting recovery from backup or temporary file."
            )
            if bak_path.exists():
                try:
                    recovered = load_outcomes_parquet(bak_path)
                    logger.info(f"Successfully recovered {len(recovered)} outcomes from {bak_path}.")
                    return recovered
                except Exception:
                    pass
            if tmp_path.exists():
                try:
                    recovered = load_outcomes_parquet(tmp_path)
                    logger.info(f"Successfully recovered {len(recovered)} outcomes from {tmp_path}.")
                    return recovered
                except Exception:
                    pass
            raise CorruptedArtifactError(
                f"Outcome artifact {path} is corrupted ({exc}) and cannot be recovered from "
                f"available backups. Manual inspection required before resuming."
            ) from exc

    elif bak_path.exists():
        try:
            return load_outcomes_parquet(bak_path)
        except Exception as exc:
            raise CorruptedArtifactError(f"Backup outcome artifact {bak_path} is corrupted: {exc}") from exc
    elif tmp_path.exists():
        try:
            return load_outcomes_parquet(tmp_path)
        except Exception:
            return []

    return []


def safe_load_scores(path: Path) -> list[ScoreOutcome]:
    """
    Load ScoreOutcome records from Parquet with automatic recovery from backup (.bak)
    or partial write (.tmp) if the primary file is corrupted.
    """
    bak_path = path.with_suffix(path.suffix + ".bak")
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    if path.exists():
        try:
            return load_scores_parquet(path)
        except Exception as exc:
            logger.warning(
                f"Primary score artifact at {path} is corrupted: {exc}. "
                f"Attempting recovery from backup or temporary file."
            )
            if bak_path.exists():
                try:
                    recovered = load_scores_parquet(bak_path)
                    logger.info(f"Successfully recovered {len(recovered)} scores from {bak_path}.")
                    return recovered
                except Exception:
                    pass
            if tmp_path.exists():
                try:
                    recovered = load_scores_parquet(tmp_path)
                    logger.info(f"Successfully recovered {len(recovered)} scores from {tmp_path}.")
                    return recovered
                except Exception:
                    pass
            raise CorruptedArtifactError(
                f"Score artifact {path} is corrupted ({exc}) and cannot be recovered from "
                f"available backups."
            ) from exc

    elif bak_path.exists():
        try:
            return load_scores_parquet(bak_path)
        except Exception as exc:
            raise CorruptedArtifactError(f"Backup score artifact {bak_path} is corrupted: {exc}") from exc
    elif tmp_path.exists():
        try:
            return load_scores_parquet(tmp_path)
        except Exception:
            return []

    return []


def atomic_save_outcomes(outcomes: list[ModelOutcome], target_path: Path) -> None:
    """
    Atomically persist ModelOutcome records to Parquet.
    Ensures deterministic deduplication prior to writing to satisfy uniqueness constraints.
    Writes to .tmp, copies prior version to .bak, then atomically renames .tmp to target.
    """
    deduped_outcomes, _ = deduplicate_outcomes(outcomes)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    bak_path = target_path.with_suffix(target_path.suffix + ".bak")

    save_outcomes_parquet(deduped_outcomes, tmp_path)

    if target_path.exists():
        try:
            shutil.copy2(target_path, bak_path)
        except Exception as exc:
            logger.debug(f"Could not create backup file {bak_path}: {exc}")

    os.replace(tmp_path, target_path)


def atomic_save_scores(scores: list[ScoreOutcome], target_path: Path) -> None:
    """
    Atomically persist ScoreOutcome records to Parquet.
    Deduplicates prior to writing.
    """
    deduped_scores, _ = deduplicate_scores(scores)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    bak_path = target_path.with_suffix(target_path.suffix + ".bak")

    save_scores_parquet(deduped_scores, tmp_path)

    if target_path.exists():
        try:
            shutil.copy2(target_path, bak_path)
        except Exception as exc:
            logger.debug(f"Could not create backup file {bak_path}: {exc}")

    os.replace(tmp_path, target_path)


def atomic_save_manifest(manifest: RunManifest, target_path: Path) -> None:
    """
    Atomically persist RunManifest JSON file with backup.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    bak_path = target_path.with_suffix(target_path.suffix + ".bak")

    save_run_manifest(manifest, tmp_path)

    if target_path.exists():
        try:
            shutil.copy2(target_path, bak_path)
        except Exception as exc:
            logger.debug(f"Could not create backup file {bak_path}: {exc}")

    os.replace(tmp_path, target_path)


def check_smoke_artifact_collision(output_dir: Path, force: bool = False) -> None:
    """
    Guard against mixing smoke test artifacts and matrix run artifacts.
    Prevents accidental corruption of smoke benchmarks.
    """
    if force:
        return
    smoke_indicators = ["smoke_manifest.json", "smoke_outcomes.parquet", "smoke_scores.parquet"]
    found = [f for f in smoke_indicators if (output_dir / f).exists()]
    if found:
        raise ConfigurationMismatchError(
            f"Output directory '{output_dir}' contains smoke test artifacts ({', '.join(found)}). "
            f"Refusing to execute matrix run in a smoke directory to avoid artifact pollution. "
            f"Specify a dedicated matrix directory (e.g. data/outcomes/matrix_train) or use --force."
        )


# ---------------------------------------------------------------------------
# Configuration & Resume Verification
# ---------------------------------------------------------------------------

def validate_resume_compatibility(
    existing_manifest: RunManifest,
    expected_fingerprint: str,
    expected_config: dict[str, Any],
    run_id: str | None = None,
) -> None:
    """
    Verify that an existing RunManifest in output_dir is strictly compatible with the
    current execution parameters using cryptographic experiment fingerprinting.
    Raises ConfigurationMismatchError if any setting conflicts.
    """
    settings = existing_manifest.effective_inference_settings or {}
    prev_fingerprint = settings.get("experiment_fingerprint")

    # If fingerprint matches exactly, configuration is 100% verified identical
    if prev_fingerprint and prev_fingerprint == expected_fingerprint:
        if run_id and existing_manifest.run_id != run_id:
            raise ConfigurationMismatchError(
                f"run_id mismatch: existing run_id='{existing_manifest.run_id}', requested='{run_id}'"
            )
        return

    # Otherwise, perform fine-grained parameter diff to generate actionable error
    mismatches: list[str] = []

    # 1. Benchmark split and file checksums
    prev_split = settings.get("benchmark_split") or settings.get("split")
    if prev_split and prev_split != expected_config["benchmark_split"]:
        mismatches.append(f"benchmark_split mismatch: existing='{prev_split}', current='{expected_config['benchmark_split']}'")

    prev_split_sha = settings.get("benchmark_split_file_sha256")
    if prev_split_sha and prev_split_sha != expected_config["benchmark_split_file_sha256"]:
        mismatches.append(f"benchmark_split_file_sha256 mismatch: existing='{prev_split_sha}', expected='{expected_config['benchmark_split_file_sha256']}'")

    if existing_manifest.benchmark_manifest_checksum != expected_config["benchmark_manifest_sha256"]:
        mismatches.append(f"benchmark_manifest_checksum mismatch: existing='{existing_manifest.benchmark_manifest_checksum}', expected='{expected_config['benchmark_manifest_sha256']}'")

    # 2. Candidate models
    prev_models = sorted(existing_manifest.per_model_counts.keys())
    cur_models = expected_config["candidate_models"]
    if prev_models != cur_models:
        mismatches.append(f"candidate_models mismatch: existing={prev_models}, current={cur_models}")

    # 3. Policy and budget
    if existing_manifest.prompt_policy_version != expected_config["prompt_policy_version"]:
        mismatches.append(f"prompt_policy_version mismatch: existing='{existing_manifest.prompt_policy_version}', current='{expected_config['prompt_policy_version']}'")

    if existing_manifest.output_budget != expected_config["output_budget"]:
        mismatches.append(f"output_budget mismatch: existing={existing_manifest.output_budget}, current={expected_config['output_budget']}")

    # 4. Sampling & execution parameters
    prev_backend = settings.get("backend")
    if prev_backend and prev_backend.lower() != expected_config["backend"].lower():
        mismatches.append(f"backend mismatch: existing='{prev_backend}', current='{expected_config['backend']}'")

    prev_think = settings.get("think")
    if prev_think is not None and prev_think != expected_config["think"]:
        mismatches.append(f"think mismatch: existing={prev_think}, current={expected_config['think']}")

    prev_temp = settings.get("temperature")
    if prev_temp is not None and prev_temp != expected_config["temperature"]:
        mismatches.append(f"temperature mismatch: existing={prev_temp}, current={expected_config['temperature']}")

    prev_seed = existing_manifest.seed
    if prev_seed != expected_config["seed"]:
        mismatches.append(f"seed mismatch: existing={prev_seed}, current={expected_config['seed']}")

    if run_id and existing_manifest.run_id != run_id:
        mismatches.append(f"run_id mismatch: existing='{existing_manifest.run_id}', requested='{run_id}'")

    if not mismatches and prev_fingerprint != expected_fingerprint:
        mismatches.append(f"experiment_fingerprint mismatch: existing='{prev_fingerprint}', expected='{expected_fingerprint}'")

    if mismatches:
        err_msg = (
            "Cannot resume matrix run due to configuration mismatch with existing manifest:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + "\nUse a different --output-dir, change parameters to match, or specify --force to overwrite."
        )
        raise ConfigurationMismatchError(err_msg)


# ---------------------------------------------------------------------------
# Manifest Builder
# ---------------------------------------------------------------------------

def build_run_manifest(
    run_id: str,
    benchmark_manifest_checksum: str,
    models_checksum: str,
    seed: int,
    prompts: list[BenchmarkPrompt],
    models: list[ModelConfig],
    outcomes: list[ModelOutcome],
    prompt_policy_version: str,
    output_budget: int | None,
    backend: str,
    split: str,
    flush_interval: int,
    temperature: float,
    timeout_seconds: float,
    think: bool | None,
    experiment_fingerprint: str | None = None,
    experiment_config: dict[str, Any] | None = None,
) -> RunManifest:
    """Build a complete, strongly-validated RunManifest from current state."""
    expected_rows = len(prompts) * len(models)
    success_count = sum(1 for o in outcomes if o.status == "SUCCESS")
    failure_count = sum(1 for o in outcomes if o.status != "SUCCESS")
    total_retries = sum(o.retry_count for o in outcomes)

    per_model_counts: dict[str, dict[str, int]] = {
        m.id: {"total": 0, "success": 0, "failed": 0, "retries": 0, "truncated": 0} for m in models
    }
    for o in outcomes:
        if o.model_id in per_model_counts:
            per_model_counts[o.model_id]["total"] += 1
            if o.status == "SUCCESS":
                per_model_counts[o.model_id]["success"] += 1
            elif o.status == "TRUNCATED":
                per_model_counts[o.model_id]["truncated"] += 1
                per_model_counts[o.model_id]["failed"] += 1
            else:
                per_model_counts[o.model_id]["failed"] += 1
            per_model_counts[o.model_id]["retries"] += o.retry_count

    per_split_counts: dict[str, int] = {split: len(prompts)}

    token_accounting_avail = any(o.prompt_tokens is not None for o in outcomes)
    latency_avail = any(o.latency_ms > 0.0 for o in outcomes)

    schema_used = resolve_response_format(policy_version=prompt_policy_version)
    if isinstance(schema_used, str):
        schema_dict = {"format": schema_used}
    else:
        schema_dict = schema_used

    effective_settings: dict[str, Any] = {
        "temperature": temperature,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "think": think,
        "max_output_tokens": output_budget,
        "backend": backend,
        "benchmark_split": split,
        "flush_interval": flush_interval,
    }
    if experiment_fingerprint:
        effective_settings["experiment_fingerprint"] = experiment_fingerprint
    if experiment_config:
        effective_settings["experiment_config"] = experiment_config

    return RunManifest(
        run_id=run_id,
        benchmark_manifest_checksum=benchmark_manifest_checksum,
        models_config_version=f"sha256:{models_checksum}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=seed,
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
        prompt_policy_version=prompt_policy_version,
        response_schema=schema_dict,
        output_budget=output_budget,
        effective_inference_settings=effective_settings,
    )


# ---------------------------------------------------------------------------
# Progress & Telemetry Reporter
# ---------------------------------------------------------------------------

class MatrixProgressReporter:
    """
    Reports sequential execution progress, empirical latency tracking,
    and remaining time estimates without speculative extrapolations.
    """

    def __init__(
        self,
        total_planned: int,
        previously_completed: int,
    ) -> None:
        self.total_planned = total_planned
        self.previously_completed = previously_completed
        self.session_completed = 0
        self.session_start_time = time.monotonic()
        self.recent_latencies: list[float] = []

    def record_attempt(self, outcome: ModelOutcome) -> str:
        self.session_completed += 1
        current_total = self.previously_completed + self.session_completed
        pct = (current_total / self.total_planned * 100.0) if self.total_planned > 0 else 100.0

        if outcome.latency_ms > 0:
            self.recent_latencies.append(outcome.latency_ms / 1000.0)
            if len(self.recent_latencies) > 50:
                self.recent_latencies.pop(0)

        elapsed_sec = time.monotonic() - self.session_start_time
        remaining_attempts = max(0, self.total_planned - current_total)

        if self.recent_latencies:
            avg_sec = sum(self.recent_latencies) / len(self.recent_latencies)
            eta_sec = remaining_attempts * avg_sec
            eta_str = self._format_seconds(eta_sec)
            rate_str = f"{avg_sec:.2f}s/att"
        elif self.session_completed > 0:
            avg_sec = elapsed_sec / self.session_completed
            eta_sec = remaining_attempts * avg_sec
            eta_str = self._format_seconds(eta_sec)
            rate_str = f"{avg_sec:.2f}s/att"
        else:
            eta_str = "calculating..."
            rate_str = "n/a"

        elapsed_str = self._format_seconds(elapsed_sec)
        msg = (
            f"[{current_total}/{self.total_planned}] ({pct:5.1f}%) | "
            f"Model: {outcome.model_id:<11} | "
            f"Status: {outcome.status:<9} | "
            f"Latency: {outcome.latency_ms/1000.0:5.2f}s | "
            f"Rate: {rate_str} | "
            f"Elapsed: {elapsed_str} | "
            f"Est. Remaining: ~{eta_str}"
        )
        return msg

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = minutes / 60.0
        return f"{hours:.2f}h"


# ---------------------------------------------------------------------------
# Core Matrix Execution Engine
# ---------------------------------------------------------------------------

def run_matrix(
    split: str = "train",
    processed_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/outcomes/matrix_train"),
    run_id: str | None = None,
    models_config_path: Path = Path("configs/models.yaml"),
    pricing_config_path: Path = Path("configs/pricing.yaml"),
    model_ids: list[str] | None = None,
    backend: str | None = None,
    use_mock: bool = False,
    prompt_policy_version: str = "v0",
    max_output_tokens: int | None = None,
    think: bool | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    timeout_seconds: float = 120.0,
    max_retries: int = 3,
    score_outcomes: bool = True,
    flush_interval: int = 1,
    dry_run: bool = False,
    force: bool = False,
    limit: int | None = None,
    retry_failed: bool = False,
    providers_override: dict[str, InferenceProvider] | None = None,
    prompts_file: Path | str | None = None,
) -> tuple[list[ModelOutcome], list[ScoreOutcome], RunManifest]:
    """
    Execute durable, crash-safe, resumable matrix evaluation across candidate models and benchmark split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Policy & smoke collision validation
    if prompt_policy_version not in SUPPORTED_PROMPT_POLICIES:
        raise ValueError(
            f"Unknown prompt policy version: '{prompt_policy_version}'. "
            f"Supported versions: {sorted(SUPPORTED_PROMPT_POLICIES)}"
        )

    check_smoke_artifact_collision(output_dir=output_dir, force=force)

    # 1. Load benchmark split or custom prompts file
    if prompts_file:
        split_file = Path(prompts_file)
    else:
        split_file = processed_dir / f"benchmark_{split}.jsonl"
    if not split_file.exists():
        raise FileNotFoundError(f"Benchmark split file not found: {split_file}")
    prompts = load_benchmark_jsonl(split_file)
    if not prompts:
        raise ValueError(f"No benchmark prompts loaded from {split_file}")

    # 2. Load model registry & pricing
    registry = load_model_registry(models_config_path)
    pricing_map = load_pricing_config(pricing_config_path)

    all_registered_ids = {m.id for m in registry.models}
    if model_ids:
        unknown_ids = set(model_ids) - all_registered_ids
        if unknown_ids:
            raise ValueError(f"Unknown candidate model IDs requested: {sorted(unknown_ids)}")
        models = [m for m in registry.models if m.id in model_ids and m.routing.enabled]
        if not models:
            raise ValueError(f"None of requested model IDs {model_ids} are enabled in {models_config_path}")
    else:
        models = registry.enabled_models()

    # 3. Checksums & experiment fingerprint
    split_file_sha = compute_file_sha256(split_file)
    bench_manifest_file = processed_dir / "benchmark_manifest.json"
    bench_checksum = compute_file_sha256(bench_manifest_file) if bench_manifest_file.exists() else "unknown"
    models_checksum = compute_file_sha256(models_config_path)

    effective_backend = "mock" if use_mock else (backend or "ollama")
    schema_used = resolve_response_format(policy_version=prompt_policy_version)
    schema_dict = {"format": schema_used} if isinstance(schema_used, str) else schema_used
    model_tags = {m.id: m.model_name for m in models}

    experiment_fingerprint, experiment_config = compute_experiment_fingerprint(
        benchmark_split=split,
        split_file_sha256=split_file_sha,
        benchmark_manifest_sha256=bench_checksum,
        model_ids=[m.id for m in models],
        model_tags=model_tags,
        models_config_sha256=models_checksum,
        prompt_policy_version=prompt_policy_version,
        output_budget=max_output_tokens,
        think=think,
        temperature=temperature,
        seed=seed,
        backend=effective_backend,
        response_schema=schema_dict,
        target_quality=0.80,
    )

    # 4. Plan attempts
    planned_attempts = plan_matrix_attempts(prompts, models)
    if limit is not None and limit > 0:
        planned_attempts = planned_attempts[:limit]
    total_planned = len(planned_attempts)

    # 5. Check existing output state for resume or completion
    outcomes_path = output_dir / "matrix_outcomes.parquet"
    scores_path = output_dir / "matrix_scores.parquet"
    manifest_path = output_dir / "matrix_manifest.json"

    existing_manifest: RunManifest | None = None
    existing_outcomes: list[ModelOutcome] = []
    existing_scores: list[ScoreOutcome] = []

    if manifest_path.exists():
        try:
            existing_manifest = load_run_manifest(manifest_path)
        except Exception as exc:
            logger.warning(f"Could not load existing manifest: {exc}")

    if existing_manifest and not force:
        validate_resume_compatibility(
            existing_manifest=existing_manifest,
            expected_fingerprint=experiment_fingerprint,
            expected_config=experiment_config,
            run_id=run_id,
        )

    # Load existing outcomes & scores
    existing_outcomes = safe_load_outcomes(outcomes_path)
    existing_outcomes, dup_outcomes = deduplicate_outcomes(existing_outcomes)
    if dup_outcomes > 0:
        logger.warning(f"Deduplicated {dup_outcomes} duplicate outcome rows loaded from {outcomes_path}.")

    if score_outcomes:
        existing_scores = safe_load_scores(scores_path)
        existing_scores, dup_scores = deduplicate_scores(existing_scores)
        if dup_scores > 0:
            logger.warning(f"Deduplicated {dup_scores} duplicate score rows loaded from {scores_path}.")

        # Reconcile missing scores if previous run crashed after saving outcomes but before scores
        if len(existing_outcomes) > len(existing_scores):
            scored_keys = {(s.prompt_id, s.model_id) for s in existing_scores}
            prompt_map = {p.prompt_id: p for p in prompts}
            scorer_reconcile = OutcomeScorer(benchmark_manifest_checksum=bench_checksum)
            unscored = [o for o in existing_outcomes if (o.prompt_id, o.model_id) not in scored_keys]
            if unscored:
                logger.info(f"Reconciling {len(unscored)} unscored outcomes from previous interrupted flush.")
                for o in unscored:
                    sc = scorer_reconcile.score_outcome(o, prompt=prompt_map.get(o.prompt_id))
                    existing_scores.append(sc)
                atomic_save_scores(existing_scores, scores_path)

    # Establish canonical completed keys
    if retry_failed:
        # If retry_failed is True, only SUCCESS and TRUNCATED count as completed
        completed_keys: set[tuple[str, str]] = {
            (o.prompt_id, o.model_id) for o in existing_outcomes if o.status in ["SUCCESS", "TRUNCATED"]
        }
    else:
        # Default: all persisted outcomes count as completed
        completed_keys = {
            (o.prompt_id, o.model_id) for o in existing_outcomes
        }

    # Determine pending attempts
    pending_attempts = [
        att for att in planned_attempts if att.key not in completed_keys
    ]

    # Resolve active run_id
    if existing_manifest:
        active_run_id = existing_manifest.run_id
    elif run_id:
        active_run_id = run_id
    else:
        active_run_id = f"matrix-{split}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Handle dry-run mode
    if dry_run:
        policy_descriptions = {
            PROMPT_POLICY_V0: "Original benchmark prompt unmodified (unconstrained generation)",
            PROMPT_POLICY_V1_CONCISE: "Concise derivation ending with 'Final Answer: <letter>'",
            PROMPT_POLICY_V2_ANSWER_ONLY: "Strict answer-only prompt (negative constraint: no reasoning/explanation)",
            PROMPT_POLICY_V3_FEW_SHOT: "1-shot synthetic demonstration prompt",
            PROMPT_POLICY_V4_STRUCTURED_OUTPUT: "Native Ollama structured JSON schema enum constraint",
        }
        policy_desc = policy_descriptions.get(prompt_policy_version, "Custom prompt policy")
        budget_desc = f"{max_output_tokens} tokens (explicit override)" if max_output_tokens else f"dynamic per task (multiple_choice: {DEFAULT_TASK_OUTPUT_BUDGETS['multiple_choice']}, exact_match: {DEFAULT_TASK_OUTPUT_BUDGETS['exact_match']}, classification: {DEFAULT_TASK_OUTPUT_BUDGETS['classification']})"
        think_desc = f"{think} (explicit override)" if think is not None else f"dynamic default (multiple_choice: {DEFAULT_THINKING_POLICY['multiple_choice']}, exact_match: {DEFAULT_THINKING_POLICY['exact_match']})"

        print("=== Matrix Runner Dry Run ===")
        print(f"Benchmark Split: {split} ({split_file})")
        print(f"Split File SHA-256: {split_file_sha}")
        print(f"Prompts Loaded: {len(prompts)}")
        print(f"Candidate Models: {[m.id for m in models]}")
        print(f"Candidate Model Tags: {model_tags}")
        print(f"Backend: {effective_backend}")
        print(f"Total Planned Attempts: {total_planned}")
        print(f"Previously Completed Attempts: {len(completed_keys)}")
        print(f"Pending Attempts to Execute: {len(pending_attempts)}")
        print(f"Prompt Policy Version: {prompt_policy_version} ({policy_desc})")
        print(f"Output Budget: {budget_desc}")
        print(f"Thinking Mode: {think_desc}")
        print(f"Response Schema: {schema_dict}")
        print(f"Sampling Parameters: temperature={temperature}, seed={seed}, timeout={timeout_seconds}s")
        print(f"Experiment Fingerprint: {experiment_fingerprint}")
        print(f"Target Output Directory: {output_dir}")
        print(f"Dry run complete. No inference executed.")

        # Return draft manifest and existing outcomes
        manifest = build_run_manifest(
            run_id=active_run_id,
            benchmark_manifest_checksum=bench_checksum,
            models_checksum=models_checksum,
            seed=seed,
            prompts=prompts,
            models=models,
            outcomes=existing_outcomes,
            prompt_policy_version=prompt_policy_version,
            output_budget=max_output_tokens,
            backend=effective_backend,
            split=split,
            flush_interval=flush_interval,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            think=think,
            experiment_fingerprint=experiment_fingerprint,
            experiment_config=experiment_config,
        )
        return existing_outcomes, existing_scores, manifest

    # Check if already 100% completed
    if len(pending_attempts) == 0 and len(completed_keys) >= total_planned:
        logger.info(
            f"Matrix run '{active_run_id}' is already 100% complete "
            f"({len(completed_keys)}/{total_planned} attempts). Nothing to run."
        )
        print(
            f"Run '{active_run_id}' in '{output_dir}' is already complete "
            f"({len(completed_keys)}/{total_planned} attempts). Exiting cleanly."
        )
        manifest = build_run_manifest(
            run_id=active_run_id,
            benchmark_manifest_checksum=bench_checksum,
            models_checksum=models_checksum,
            seed=seed,
            prompts=prompts,
            models=models,
            outcomes=existing_outcomes,
            prompt_policy_version=prompt_policy_version,
            output_budget=max_output_tokens,
            backend=effective_backend,
            split=split,
            flush_interval=flush_interval,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            think=think,
            experiment_fingerprint=experiment_fingerprint,
            experiment_config=experiment_config,
        )
        return existing_outcomes, existing_scores, manifest

    # 6. Initialize providers
    providers: dict[str, InferenceProvider] = {}
    if providers_override:
        providers = providers_override
    else:
        for model in models:
            m_pricing = pricing_map.get(model.id)
            if use_mock:
                providers[model.id] = MockInferenceProvider(model_config=model, pricing=m_pricing)
            elif effective_backend.lower() == "ollama":
                providers[model.id] = OllamaProvider(model_config=model, pricing=m_pricing)
            else:
                providers[model.id] = VLLMProvider(model_config=model, pricing=m_pricing)

    # Scorer setup
    scorer = OutcomeScorer(benchmark_manifest_checksum=bench_checksum) if score_outcomes else None

    # Track in-memory collections
    outcomes: list[ModelOutcome] = list(existing_outcomes)
    scores: list[ScoreOutcome] = list(existing_scores)

    progress_reporter = MatrixProgressReporter(
        total_planned=total_planned,
        previously_completed=len(completed_keys),
    )

    if len(completed_keys) > 0:
        print(
            f"Resuming matrix run '{active_run_id}': "
            f"{len(completed_keys)} already completed, {len(pending_attempts)} remaining."
        )
    else:
        print(f"Starting fresh matrix run '{active_run_id}' ({total_planned} planned attempts)...")

    # Helper function for atomic checkpoint flush
    def flush_checkpoint() -> None:
        nonlocal outcomes, scores
        deduped_outcomes, _ = deduplicate_outcomes(outcomes)
        outcomes = deduped_outcomes
        atomic_save_outcomes(deduped_outcomes, outcomes_path)

        if score_outcomes:
            deduped_scores, _ = deduplicate_scores(scores)
            scores = deduped_scores
            atomic_save_scores(deduped_scores, scores_path)

        manifest = build_run_manifest(
            run_id=active_run_id,
            benchmark_manifest_checksum=bench_checksum,
            models_checksum=models_checksum,
            seed=seed,
            prompts=prompts,
            models=models,
            outcomes=deduped_outcomes,
            prompt_policy_version=prompt_policy_version,
            output_budget=max_output_tokens,
            backend=effective_backend,
            split=split,
            flush_interval=flush_interval,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            think=think,
            experiment_fingerprint=experiment_fingerprint,
            experiment_config=experiment_config,
        )
        atomic_save_manifest(manifest, manifest_path)

    # 7. Execute pending attempts sequentially
    pending_count_since_flush = 0
    try:
        for attempt in pending_attempts:
            provider = providers[attempt.model_id]
            outcome = provider.execute_prompt(
                attempt.prompt,
                run_id=active_run_id,
                max_retries=max_retries,
                temperature=temperature,
                seed=seed,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                think=think,
                prompt_policy_version=prompt_policy_version,
            )
            outcomes.append(outcome)

            if scorer:
                score = scorer.score_outcome(outcome, prompt=attempt.prompt)
                scores.append(score)

            # Telemetry reporting
            log_line = progress_reporter.record_attempt(outcome)
            print(log_line)

            pending_count_since_flush += 1
            if pending_count_since_flush >= flush_interval:
                flush_checkpoint()
                pending_count_since_flush = 0

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Received interrupt signal. Saving completed attempts to disk...")
        flush_checkpoint()
        print(f"[INTERRUPT] Successfully saved {len(outcomes)} outcomes to {output_dir}. Safe to resume.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] Unhandled exception during execution: {exc}. Saving completed attempts...")
        flush_checkpoint()
        raise

    # Final flush to ensure clean persistence
    flush_checkpoint()

    final_manifest = build_run_manifest(
        run_id=active_run_id,
        benchmark_manifest_checksum=bench_checksum,
        models_checksum=models_checksum,
        seed=seed,
        prompts=prompts,
        models=models,
        outcomes=outcomes,
        prompt_policy_version=prompt_policy_version,
        output_budget=max_output_tokens,
        backend=effective_backend,
        split=split,
        flush_interval=flush_interval,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        think=think,
        experiment_fingerprint=experiment_fingerprint,
        experiment_config=experiment_config,
    )

    print("\n=== Matrix Run Completed ===")
    print(f"Run ID: {final_manifest.run_id}")
    print(f"Expected Outcomes: {final_manifest.expected_outcome_rows}")
    print(f"Actual Outcomes: {final_manifest.actual_outcome_rows}")
    print(f"Success Count: {final_manifest.success_count}")
    print(f"Failure Count: {final_manifest.failure_count}")
    print(f"Experiment Fingerprint: {experiment_fingerprint}")
    print(f"Artifacts persisted to: {output_dir}")

    return outcomes, scores, final_manifest


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-Aware LLM Router: Matrix Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", type=str, default="train", help="Benchmark split to evaluate (e.g. train, validation)")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed benchmark directory")
    parser.add_argument("--output-dir", type=str, default="data/outcomes/matrix_train", help="Target output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run identifier")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model IDs (e.g. qwen3-1.7b,qwen3-4b)")
    parser.add_argument("--backend", type=str, default=None, help="Backend override (e.g. ollama, vllm, mock)")
    parser.add_argument("--mock", action="store_true", help="Execute with mock inference provider for testing")
    parser.add_argument("--prompt-policy", type=str, default="v0", help="Prompt policy version (e.g. v0, v1_concise, v4_structured_output)")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Maximum generated token budget override")
    parser.add_argument("--think", action="store_true", default=None, help="Explicitly enable reasoning/thinking tokens")
    parser.add_argument("--no-think", action="store_false", dest="think", help="Explicitly disable reasoning/thinking tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-prompt timeout in seconds")
    parser.add_argument("--flush-interval", type=int, default=1, help="Number of attempts between atomic disk flushes")
    parser.add_argument("--score", action="store_true", default=True, help="Score outcomes after inference")
    parser.add_argument("--no-score", action="store_false", dest="score", help="Skip scoring outcomes")
    parser.add_argument("--dry-run", action="store_true", help="Report planned attempts and exit without invoking inference")
    parser.add_argument("--force", action="store_true", help="Force execution even if existing manifest settings conflict")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of planned attempts (for testing/partial runs)")
    parser.add_argument("--retry-failed", action="store_true", help="Retry existing attempts with FAILED/TIMEOUT/BACKEND_UNAVAILABLE status")
    parser.add_argument("--prompts-file", type=str, default=None, help="Custom benchmark prompts JSONL file override")

    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")] if args.models else None

    try:
        run_matrix(
            split=args.split,
            processed_dir=Path(args.processed_dir),
            output_dir=Path(args.output_dir),
            run_id=args.run_id,
            model_ids=model_list,
            backend=args.backend,
            use_mock=args.mock,
            prompt_policy_version=args.prompt_policy,
            max_output_tokens=args.max_output_tokens,
            think=args.think,
            temperature=args.temperature,
            seed=args.seed,
            timeout_seconds=args.timeout,
            score_outcomes=args.score,
            flush_interval=args.flush_interval,
            dry_run=args.dry_run,
            force=args.force,
            limit=args.limit,
            retry_failed=args.retry_failed,
            prompts_file=Path(args.prompts_file) if args.prompts_file else None,
        )
    except ConfigurationMismatchError as exc:
        print(f"\n[CONFIGURATION ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except CorruptedArtifactError as exc:
        print(f"\n[CORRUPTED ARTIFACT] {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[INPUT ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Matrix execution interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[EXECUTION ERROR] Fatal matrix execution failure: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
