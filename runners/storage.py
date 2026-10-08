import json
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq

from runners.outcome_schema import ModelOutcome, RunManifest


def save_outcomes_parquet(outcomes: list[ModelOutcome], path: str | Path) -> None:
    """
    Save a list of ModelOutcome objects to a Parquet file.
    Enforces uniqueness of (prompt_id, model_id, run_id) across rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Uniqueness check
    seen_keys: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []

    for item in outcomes:
        key = (item.prompt_id, item.model_id, item.run_id)
        if key in seen_keys:
            raise ValueError(
                f"Duplicate canonical outcome key detected: prompt_id='{item.prompt_id}', "
                f"model_id='{item.model_id}', run_id='{item.run_id}'."
            )
        seen_keys.add(key)

        row = item.model_dump()
        # Serialize attempt_history list as JSON string for clean flat parquet schema
        row["attempt_history"] = json.dumps(row.get("attempt_history", []))
        rows.append(row)

    if not rows:
        # Create empty table matching schema fields
        field_names = list(ModelOutcome.model_fields.keys())
        empty_dict = {f: [] for f in field_names}
        table = pa.Table.from_pydict(empty_dict)
    else:
        table = pa.Table.from_pylist(rows)

    pq.write_table(table, str(path), compression="snappy")


def load_outcomes_parquet(path: str | Path) -> list[ModelOutcome]:
    """Load a list of ModelOutcome objects from a Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Outcome Parquet file not found: {path}")

    table = pq.read_table(str(path))
    raw_rows = table.to_pylist()

    outcomes: list[ModelOutcome] = []
    for r in raw_rows:
        # Deserialize attempt_history if string
        if isinstance(r.get("attempt_history"), str):
            try:
                r["attempt_history"] = json.loads(r["attempt_history"])
            except Exception:
                r["attempt_history"] = []
        outcomes.append(ModelOutcome.model_validate(r))

    return outcomes


def save_run_manifest(manifest: RunManifest, path: str | Path) -> None:
    """Save a RunManifest to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load a RunManifest from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return RunManifest.model_validate(data)
