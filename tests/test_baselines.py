import pytest

from router.baselines import (
    AlwaysMiddlePolicy,
    AlwaysSmallPolicy,
    AlwaysStrongPolicy,
    CostWeightedRandomPolicy,
    HeuristicPolicy,
    RandomPolicy,
    RoutingRequest,
    get_baseline_policy,
)
from router.config import load_model_registry


@pytest.fixture
def registry():
    return load_model_registry("configs/models.yaml")


# 1. Static policies
def test_always_small_always_selects_cheapest(registry):
    policy = AlwaysSmallPolicy(registry)
    req = RoutingRequest(prompt="What is gravity?", task_type="qa", difficulty="easy")
    assert policy.route(req) == "qwen3-1.7b"


def test_always_middle_always_selects_middle(registry):
    policy = AlwaysMiddlePolicy(registry)
    req = RoutingRequest(prompt="What is gravity?", task_type="qa", difficulty="medium")
    assert policy.route(req) == "qwen3-4b"


def test_always_strong_always_selects_strongest(registry):
    policy = AlwaysStrongPolicy(registry)
    req = RoutingRequest(prompt="What is gravity?", task_type="qa", difficulty="hard")
    assert policy.route(req) == "qwen3-8b"


# 2. Random policy determinism
def test_random_policy_deterministic_with_fixed_seed(registry):
    policy1 = RandomPolicy(registry, seed=42)
    policy2 = RandomPolicy(registry, seed=42)

    prompts = [f"Prompt query number {i}" for i in range(20)]
    routes1 = [policy1.route(RoutingRequest(prompt=p)) for p in prompts]
    routes2 = [policy2.route(RoutingRequest(prompt=p)) for p in prompts]

    assert routes1 == routes2
    # Verify all enabled models are selected across diverse prompts
    assert set(routes1) == {"qwen3-1.7b", "qwen3-4b", "qwen3-8b"}


# 3. Cost-weighted random policy
def test_cost_weighted_zero_price_division_by_zero_prevention(registry):
    # Local models have 0.0 USD price in models.yaml and pricing.yaml
    policy = CostWeightedRandomPolicy(registry, seed=42)

    # Probabilities should sum to 1.0 and be inversely proportional to parameter size
    assert sum(policy.probabilities) == pytest.approx(1.0, rel=1e-5)
    # qwen3-1.7b (1.7B) should have highest probability, qwen3-8b (8B) lowest
    assert policy.probabilities[0] > policy.probabilities[1] > policy.probabilities[2]

    # Determinism check
    req = RoutingRequest(prompt="Calculate the prime factors of 997.")
    route1 = policy.route(req)
    route2 = policy.route(req)
    assert route1 == route2


# 4. Heuristic policy rules
def test_heuristic_policy_rules(registry):
    heuristic = HeuristicPolicy(registry)

    # Rule 1: High difficulty / reasoning / coding -> qwen3-8b
    assert heuristic.route(RoutingRequest(prompt="Write a quicksort in Python", task_type="coding", difficulty="medium")) == "qwen3-8b"
    assert heuristic.route(RoutingRequest(prompt="Complex proof", task_type="reasoning", difficulty="easy")) == "qwen3-8b"
    assert heuristic.route(RoutingRequest(prompt="Hard question", task_type="qa", difficulty="hard")) == "qwen3-8b"
    assert heuristic.route(RoutingRequest(prompt="Long context " * 900, task_type="qa", difficulty="easy")) == "qwen3-8b"

    # Rule 2: Medium difficulty / math / IFEval -> qwen3-4b
    assert heuristic.route(RoutingRequest(prompt="Solve 3x + 4 = 19", task_type="math", difficulty="easy")) == "qwen3-4b"
    assert heuristic.route(RoutingRequest(prompt="Follow rules", task_type="instruction_following", difficulty="easy")) == "qwen3-4b"
    assert heuristic.route(RoutingRequest(prompt="Standard question", task_type="qa", difficulty="medium")) == "qwen3-4b"

    # Rule 3: Easy classification / QA / summarization -> qwen3-1.7b
    assert heuristic.route(RoutingRequest(prompt="Capital of France?", task_type="qa", difficulty="easy")) == "qwen3-1.7b"
    assert heuristic.route(RoutingRequest(prompt="Summarize article", task_type="summarization", difficulty="easy")) == "qwen3-1.7b"
    assert heuristic.route(RoutingRequest(prompt="Classify sentiment", task_type="classification", difficulty="easy")) == "qwen3-1.7b"


def test_heuristic_policy_determinism(registry):
    heuristic = HeuristicPolicy(registry)
    req = RoutingRequest(prompt="Analyze this passage", task_type="qa", difficulty="easy")
    assert heuristic.route(req) == heuristic.route(req)


# 5. Factory and error handling
def test_baseline_factory(registry):
    p_small = get_baseline_policy("always-small", registry)
    assert isinstance(p_small, AlwaysSmallPolicy)

    p_heuristic = get_baseline_policy("heuristic", registry)
    assert isinstance(p_heuristic, HeuristicPolicy)

    with pytest.raises(ValueError, match="Unknown baseline policy"):
        get_baseline_policy("non-existent-policy", registry)
