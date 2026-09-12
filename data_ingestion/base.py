from abc import ABC, abstractmethod
from typing import Any

from router.benchmark_schema import BenchmarkPrompt


class DatasetAdapter(ABC):
    """Convert records from one source dataset into BenchmarkPrompt records."""

    @abstractmethod
    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        """Convert one source record into the canonical benchmark schema."""
        raise NotImplementedError