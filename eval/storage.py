import json
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq

from eval.scoring_schema import ScoreOutcome


def save_scores_parquet(scores: list[ScoreOutcome], path: str | Path) -> None:
    """
    Save ScoreOutcome records to Parquet format.
    Enforces uniqueness of canonical_key across scored rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys: set[str] = set()
    rows: list[dict[str, Any]] = []

    for item in scores:
        if item.canonical_key in seen_keys:
            raise ValueError(
                f"Duplicate canonical score key detected: '{item.canonical_key}'."
            )
        seen_keys.add(item.canonical_key)

        row = item.model_dump()
        row["details"] = json.dumps(row.get("details", {}))
        rows.append(row)

    if not rows:
        field_names = list(ScoreOutcome.model_fields.keys())
        empty_dict = {f: [] for f in field_names}
        table = pa.Table.from_pydict(empty_dict)
    else:
        table = pa.Table.from_pylist(rows)

    pq.write_table(table, str(path), compression="snappy")


def load_scores_parquet(path: str | Path) -> list[ScoreOutcome]:
    """Load ScoreOutcome records from Parquet format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Score Parquet file not found: {path}")

    table = pq.read_table(str(path))
    raw_rows = table.to_pylist()

    scores: list[ScoreOutcome] = []
    for r in raw_rows:
        if isinstance(r.get("details"), str):
            try:
                r["details"] = json.loads(r["details"])
            except Exception:
                r["details"] = {}
        scores.append(ScoreOutcome.model_validate(r))

    return scores
