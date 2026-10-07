from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


def format_cnn_dailymail_prompt(article: str) -> str:
    """Format an article into a model-facing summarization prompt."""
    if not isinstance(article, str) or not article.strip():
        raise ValueError("CNN/DailyMail record is missing a valid 'article'.")
    return f"Summarize the following article:\n\n{article.strip()}"


class CNNDailyMailAdapter(DatasetAdapter):
    """Adapter for CNN/DailyMail summarization records."""

    source_dataset = "cnn_dailymail"

    def __init__(self, source_config: str = "3.0.0") -> None:
        if not isinstance(source_config, str) or not source_config.strip():
            raise ValueError("source_config must be a non-empty string.")
        self.source_config = source_config.strip()

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        if not isinstance(record, dict):
            raise ValueError("CNN/DailyMail record must be a dictionary.")

        article = record.get("article")
        highlights = record.get("highlights")
        native_id = record.get("id")

        if not isinstance(article, str) or not article.strip():
            raise ValueError("CNN/DailyMail record is missing a valid 'article'.")

        if not isinstance(highlights, str) or not highlights.strip():
            raise ValueError("CNN/DailyMail record is missing valid 'highlights' summary.")

        resolved_source_id = (
            str(native_id).strip()
            if native_id is not None and str(native_id).strip()
            else str(source_id).strip()
        )

        prompt_text = format_cnn_dailymail_prompt(article)

        prompt_id = (
            f"cnn_dailymail-{self.source_config}-"
            f"{source_split}-{resolved_source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text,
            task_type="summarization",
            domain="news",
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=resolved_source_id,
            split=benchmark_split,
            expected_output=highlights.strip(),
            evaluation_type="reference_based",
        )
