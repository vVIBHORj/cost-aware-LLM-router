from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supports_chat: bool
    supports_reasoning: bool
    supports_code: bool
    supports_math: bool


class ModelLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_1m_tokens_usd: float = Field(ge=0)
    output_per_1m_tokens_usd: float = Field(ge=0)


class ModelRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    priority: int = Field(gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    provider: Literal["local", "hosted"]
    backend: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    role: Literal["cheap", "middle", "strong"]

    capabilities: ModelCapabilities
    limits: ModelLimits
    pricing: ModelPricing
    routing: ModelRouting


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelConfig]

    def enabled_models(self) -> list[ModelConfig]:
        return [model for model in self.models if model.routing.enabled]


def load_model_registry(
    path: str | Path = "configs/models.yaml",
) -> ModelRegistry:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Model registry must contain a YAML mapping.")

    registry = ModelRegistry.model_validate(data)

    model_ids = [model.id for model in registry.models]

    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Model IDs must be unique.")

    return registry