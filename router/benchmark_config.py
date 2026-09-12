from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class CoreCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    count: int = Field(gt=0)
    sources: list[str] = Field(min_length=1)


class CoreSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: int = Field(gt=0)
    validation: int = Field(gt=0)
    test: int = Field(gt=0)


class CoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(gt=0)
    categories: list[CoreCategory] = Field(min_length=1)

    difficulty_levels: list[
        Literal["easy", "medium", "hard"]
    ] = Field(min_length=1)

    difficulty_strategy: Literal["stratified"]

    split: CoreSplit

    @model_validator(mode="after")
    def validate_totals(self):
        category_total = sum(
            category.count for category in self.categories
        )

        if category_total != self.total:
            raise ValueError(
                f"Core category counts sum to {category_total}, "
                f"expected {self.total}"
            )

        split_total = (
            self.split.train
            + self.split.validation
            + self.split.test
        )

        if split_total != self.total:
            raise ValueError(
                f"Core split counts sum to {split_total}, "
                f"expected {self.total}"
            )

        category_names = [category.name for category in self.categories]

        if len(category_names) != len(set(category_names)):
            raise ValueError("Core category names must be unique.")

        return self


class StressType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    count: int = Field(gt=0)


class StressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(gt=0)
    held_out: bool

    types: list[StressType] = Field(min_length=1)

    split: dict[str, int]

    @model_validator(mode="after")
    def validate_totals(self):
        type_total = sum(
            stress_type.count for stress_type in self.types
        )

        if type_total != self.total:
            raise ValueError(
                f"Stress type counts sum to {type_total}, "
                f"expected {self.total}"
            )

        if self.split.get("stress") != self.total:
            raise ValueError(
                "Stress split must contain exactly the configured "
                "stress total."
            )

        if not self.held_out:
            raise ValueError(
                "Stress benchmark must be held out."
            )

        type_names = [stress_type.name for stress_type in self.types]

        if len(type_names) != len(set(type_names)):
            raise ValueError("Stress type names must be unique.")

        return self


class SplittingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: Literal["prompt"]
    leakage_guard: bool

    allowed_splits: list[
        Literal["train", "validation", "test", "stress"]
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_splitting(self):
        required = {"train", "validation", "test", "stress"}

        if set(self.allowed_splits) != required:
            raise ValueError(
                "Allowed splits must be train, validation, test, and stress."
            )

        if not self.leakage_guard:
            raise ValueError(
                "Leakage guard must be enabled."
            )

        return self


class SourceDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str | None = None
    config: str | None = None
    covers: list[str] = Field(min_length=1)


class ReferenceDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    purpose: Literal[
        "methodology_reference",
        "final_evaluation_reference",
    ]


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["stratified"]
    stratify_by: list[str] = Field(min_length=1)
    dedup: bool
    max_prompt_tokens: int | None = Field(default=None, gt=0)
    shuffle: bool


class ProvenanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_fields: list[str] = Field(min_length=1)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_types: list[
        Literal[
            "exact_match",
            "multiple_choice",
            "code_execution",
            "ifeval",
            "reference_based",
            "llm_judge",
        ]
    ] = Field(min_length=1)


class BenchmarkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(gt=0)
    random_seed: int = Field(ge=0)

    total_prompts: int = Field(gt=0)
    core_prompts: int = Field(gt=0)
    stress_prompts: int = Field(gt=0)

    candidate_models: list[CandidateModel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_totals(self):
        if self.core_prompts + self.stress_prompts != self.total_prompts:
            raise ValueError(
                "Core and stress totals must equal total_prompts."
            )

        model_names = [model.name for model in self.candidate_models]

        if len(model_names) != len(set(model_names)):
            raise ValueError(
                "Candidate model names must be unique."
            )

        return self


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: BenchmarkMetadata
    core: CoreConfig
    stress: StressConfig
    splitting: SplittingConfig

    source_datasets: list[SourceDataset] = Field(min_length=1)

    reference_only: list[ReferenceDataset] = Field(
        default_factory=list
    )

    sampling: SamplingConfig
    provenance: ProvenanceConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def validate_benchmark(self):
        if self.core.total != self.benchmark.core_prompts:
            raise ValueError(
                "Core total does not match benchmark.core_prompts."
            )

        if self.stress.total != self.benchmark.stress_prompts:
            raise ValueError(
                "Stress total does not match benchmark.stress_prompts."
            )

        source_names = {
            source.name for source in self.source_datasets
        }

        referenced_sources = set()

        for category in self.core.categories:
            referenced_sources.update(category.sources)

        missing_sources = referenced_sources - source_names

        if missing_sources:
            raise ValueError(
                f"Categories reference undefined datasets: "
                f"{sorted(missing_sources)}"
            )

        return self


def load_benchmark_config(
    path: str | Path = "configs/benchmark.yaml",
) -> BenchmarkConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Benchmark configuration must contain a YAML mapping."
        )

    return BenchmarkConfig.model_validate(data)