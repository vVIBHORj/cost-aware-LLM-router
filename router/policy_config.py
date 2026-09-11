from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class QualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: float = Field(gt=0, le=1)
    probability_threshold: float = Field(gt=0, le=1)


class LatencyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    max_latency_ms: int = Field(gt=0)


class ObjectivePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cost"]
    latency_penalty: float = Field(ge=0)


class FallbackPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    min_confidence: float = Field(gt=0, le=1)
    fallback_model: str = Field(min_length=1)


class RoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality: QualityPolicy
    latency: LatencyPolicy
    objective: ObjectivePolicy
    fallback: FallbackPolicy


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing: RoutingPolicy


def load_policy_config(
    path: str | Path = "configs/policies.yaml",
) -> PolicyConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Policy configuration must contain a YAML mapping.")

    policy = PolicyConfig.model_validate(data)

    return policy