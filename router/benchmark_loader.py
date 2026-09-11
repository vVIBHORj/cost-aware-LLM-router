from pathlib import Path
import json

from pydantic import ValidationError

from router.benchmark_schema import BenchmarkPrompt


def load_benchmark_jsonl(
    path: str | Path,
) -> list[BenchmarkPrompt]:
    path = Path(path)

    prompts: list[BenchmarkPrompt] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            try:
                item = BenchmarkPrompt.model_validate(data)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid benchmark record on line {line_number}: {exc}"
                ) from exc

            if item.prompt_id in seen_ids:
                raise ValueError(
                    f"Duplicate prompt_id: {item.prompt_id}"
                )

            seen_ids.add(item.prompt_id)
            prompts.append(item)

    return prompts