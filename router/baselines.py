from abc import ABC, abstractmethod
import hashlib
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from router.config import ModelConfig, ModelRegistry, load_model_registry


class RoutingRequest(BaseModel):
    """Incoming prompt and request metadata available at routing time."""
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, description="Prompt text to be routed.")
    task_type: str | None = Field(default=None, description="Task category if provided by user/system.")
    domain: str | None = Field(default=None, description="Problem domain if provided.")
    difficulty: Literal["easy", "medium", "hard"] | None = Field(
        default=None, description="Estimated or declared difficulty."
    )


class BaselinePolicy(ABC):
    """Abstract interface for a routing policy."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or load_model_registry()
        self.enabled_models = self.registry.enabled_models()
        if not self.enabled_models:
            raise ValueError("No enabled models found in registry.")

    @abstractmethod
    def route(self, request: RoutingRequest) -> str:
        """Return the selected candidate model ID for this request."""
        pass


class AlwaysSmallPolicy(BaselinePolicy):
    """Always routes to the smallest/cheapest model (qwen3-1.7b)."""

    def route(self, request: RoutingRequest) -> str:
        cheap_models = [m for m in self.enabled_models if m.role == "cheap"]
        return cheap_models[0].id if cheap_models else self.enabled_models[0].id


class AlwaysMiddlePolicy(BaselinePolicy):
    """Always routes to the middle model (qwen3-4b)."""

    def route(self, request: RoutingRequest) -> str:
        middle_models = [m for m in self.enabled_models if m.role == "middle"]
        return middle_models[0].id if middle_models else self.enabled_models[0].id


class AlwaysStrongPolicy(BaselinePolicy):
    """Always routes to the strongest/largest model (qwen3-8b)."""

    def route(self, request: RoutingRequest) -> str:
        strong_models = [m for m in self.enabled_models if m.role == "strong"]
        return strong_models[0].id if strong_models else self.enabled_models[-1].id


class RandomPolicy(BaselinePolicy):
    """
    Uniform random routing baseline.
    Deterministically selects among enabled models based on cryptographic hash of (prompt + seed).
    """

    def __init__(self, registry: ModelRegistry | None = None, seed: int = 42) -> None:
        super().__init__(registry)
        self.seed = seed

    def route(self, request: RoutingRequest) -> str:
        h = int(hashlib.sha256(f"{request.prompt}:{self.seed}".encode()).hexdigest(), 16)
        idx = h % len(self.enabled_models)
        return self.enabled_models[idx].id


class CostWeightedRandomPolicy(BaselinePolicy):
    """
    Cost-weighted random routing baseline.
    Biases selection toward cheaper models.

    Monetary vs Operational Cost Handling:
    - If models have positive monetary pricing in configs/pricing.yaml, weights are proportional
      to inverse monetary cost: w_i = (1 / cost_i) / sum(1 / cost_j).
    - If local models have zero monetary cost (0.0 USD), monetary cost weighting is degenerate (0/0).
      Instead of inventing artificial monetary prices, this policy uses the operational parameter size
      proxy (1.7B, 4.0B, 8.0B) so that smaller models have proportionally higher selection probability:
      w_i proportional to (1 / param_count).
    - Prevents division-by-zero under all circumstances.
    """

    # Operational parameter scale proxy in billions
    PARAM_MAP: dict[str, float] = {
        "qwen3-1.7b": 1.7,
        "qwen3-4b": 4.0,
        "qwen3-8b": 8.0,
    }

    def __init__(self, registry: ModelRegistry | None = None, seed: int = 42) -> None:
        super().__init__(registry)
        self.seed = seed

        # Calculate selection cumulative probability distribution
        total_monetary_price = sum(
            m.pricing.input_per_1m_tokens_usd + m.pricing.output_per_1m_tokens_usd for m in self.enabled_models
        )

        raw_weights: list[float] = []
        if total_monetary_price > 0.0:
            # Positive monetary pricing available
            for m in self.enabled_models:
                unit_price = (m.pricing.input_per_1m_tokens_usd + m.pricing.output_per_1m_tokens_usd) / 2.0
                raw_weights.append(1.0 / max(unit_price, 0.0001))
        else:
            # Local zero-monetary cost: use operational compute / parameter proxy
            for m in self.enabled_models:
                params = self.PARAM_MAP.get(m.id, 4.0)
                raw_weights.append(1.0 / params)

        weight_sum = sum(raw_weights)
        self.probabilities = [w / weight_sum for w in raw_weights]

        # Cumulative thresholds in [0, 10000] for deterministic integer routing
        self.thresholds: list[int] = []
        cum = 0.0
        for p in self.probabilities:
            cum += p
            self.thresholds.append(int(cum * 10000))
        self.thresholds[-1] = 10000

    def route(self, request: RoutingRequest) -> str:
        h = int(hashlib.sha256(f"cost_weight:{request.prompt}:{self.seed}".encode()).hexdigest(), 16) % 10000
        for idx, th in enumerate(self.thresholds):
            if h < th:
                return self.enabled_models[idx].id
        return self.enabled_models[-1].id


class HeuristicPolicy(BaselinePolicy):
    """
    Transparent, deterministic rule-based heuristic router.
    Defined entirely a priori from request metadata without ML models, embeddings,
    or exposure to benchmark outcomes.

    Rules:
    1. High-difficulty tasks (difficulty='hard', task_type in {'reasoning', 'coding'}, or word length > 800)
       are routed to the strongest model (qwen3-8b).
    2. Medium-difficulty tasks (difficulty='medium' or task_type in {'math', 'instruction_following'})
       are routed to the balanced middle model (qwen3-4b).
    3. Low-difficulty / structural tasks (difficulty='easy' with task_type in {'qa', 'classification', 'summarization', 'other'})
       are routed to the cheapest model (qwen3-1.7b).
    4. Default fallback: qwen3-4b.
    """

    def route(self, request: RoutingRequest) -> str:
        prompt_words = len(request.prompt.split())

        # Rule 1: High complexity / coding / reasoning / long prompt
        if (
            request.difficulty == "hard"
            or request.task_type in ("reasoning", "coding")
            or prompt_words > 800
        ):
            return self._get_model_by_role("strong", fallback_id="qwen3-8b")

        # Rule 2: Medium complexity / multi-step math / instruction constraints
        if (
            request.difficulty == "medium"
            or request.task_type in ("math", "instruction_following")
        ):
            return self._get_model_by_role("middle", fallback_id="qwen3-4b")

        # Rule 3: Easy / low-complexity extraction or QA
        if (
            request.difficulty == "easy"
            and request.task_type in ("qa", "classification", "summarization", "other")
        ):
            return self._get_model_by_role("cheap", fallback_id="qwen3-1.7b")

        # Default fallback
        return self._get_model_by_role("middle", fallback_id="qwen3-4b")

    def _get_model_by_role(self, role: str, fallback_id: str) -> str:
        for m in self.enabled_models:
            if m.role == role:
                return m.id
        return fallback_id


def get_baseline_policy(
    policy_name: str,
    registry: ModelRegistry | None = None,
    seed: int = 42,
) -> BaselinePolicy:
    """Factory for instantiating named baseline routing policies."""
    policies = {
        "always-small": AlwaysSmallPolicy,
        "always-middle": AlwaysMiddlePolicy,
        "always-strong": AlwaysStrongPolicy,
        "random": lambda reg: RandomPolicy(reg, seed=seed),
        "cost-weighted-random": lambda reg: CostWeightedRandomPolicy(reg, seed=seed),
        "heuristic": HeuristicPolicy,
    }

    if policy_name not in policies:
        raise ValueError(
            f"Unknown baseline policy '{policy_name}'. Valid policies: {list(policies.keys())}"
        )

    creator = policies[policy_name]
    return creator(registry)
