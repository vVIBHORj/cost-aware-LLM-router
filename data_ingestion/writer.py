import json
from pathlib import Path
from collections.abc import Iterable

from router.benchmark_schema import BenchmarkPrompt


def write_benchmark_jsonl(
    items: Iterable[BenchmarkPrompt],
    path: str | Path,
) -> Path:
    """
    Write validated BenchmarkPrompt objects to JSONL.

    Each line contains one benchmark prompt.
    Parent directories are created automatically.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with path.open("w", encoding="utf-8") as file:
        for item in items:
            if not isinstance(item, BenchmarkPrompt):
                raise TypeError(
                    "write_benchmark_jsonl() expects BenchmarkPrompt objects."
                )

            payload = item.model_dump(mode="json")

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    return path