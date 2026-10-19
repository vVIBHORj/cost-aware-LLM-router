import json
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from router.benchmark_schema import BenchmarkPrompt
from router.config import ModelCapabilities, ModelConfig, ModelLimits, ModelPricing, ModelRouting
from runners.inference import GenerationRequest
from runners.ollama_runner import (
    DEFAULT_CANDIDATE_TAG_MAP,
    EmptyOutputError,
    MalformedResponseError,
    ModelNotFoundError,
    OllamaProvider,
    ReasoningOnlyError,
    normalize_ollama_base_url,
)
from runners.outcome_schema import ModelOutcome


@pytest.fixture
def dummy_model_config() -> ModelConfig:
    return ModelConfig(
        id="qwen3-1.7b",
        display_name="Qwen3 1.7B",
        provider="local",
        backend="ollama",
        model_name="Qwen/Qwen3-1.7B",
        role="cheap",
        capabilities=ModelCapabilities(
            supports_chat=True,
            supports_reasoning=True,
            supports_code=True,
            supports_math=True,
        ),
        limits=ModelLimits(
            context_window_tokens=32768,
            max_output_tokens=4096,
        ),
        pricing=ModelPricing(
            input_per_1m_tokens_usd=0.0,
            output_per_1m_tokens_usd=0.0,
        ),
        routing=ModelRouting(
            enabled=True,
            priority=1,
        ),
    )


@pytest.fixture
def dummy_prompt() -> BenchmarkPrompt:
    return BenchmarkPrompt(
        prompt_id="prompt-mock-001",
        prompt="What is the capital of France?",
        task_type="qa",
        domain="geography",
        difficulty="easy",
        source_dataset="test_qa",
        source_split="test",
        source_id="1",
        split="test",
        expected_output="Paris",
        evaluation_type="exact_match",
    )


def make_mock_http_response(status: int, data: dict):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


# 1. Base URL normalization
def test_normalize_ollama_base_url():
    assert normalize_ollama_base_url("http://localhost:11434/") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434/v1"
    assert normalize_ollama_base_url() == "http://127.0.0.1:11434"


# 2. Tag mapping from internal IDs
def test_ollama_tag_mapping(dummy_model_config):
    provider = OllamaProvider(dummy_model_config)
    # qwen3-1.7b maps to qwen3:1.7b by default
    assert provider.model_tag == "qwen3:1.7b"

    # Custom mapping
    custom_provider = OllamaProvider(dummy_model_config, tag_mapping={"qwen3-1.7b": "custom-qwen:tag"})
    assert custom_provider.model_tag == "custom-qwen:tag"


# 3. Availability: model installed vs missing vs connection failure
@patch("urllib.request.urlopen")
def test_ollama_availability_model_present(mock_urlopen, dummy_model_config):
    models_data = {
        "data": [
            {"id": "qwen3:1.7b"},
            {"id": "llama3.2:latest"},
        ]
    }
    mock_urlopen.return_value = make_mock_http_response(200, models_data)

    provider = OllamaProvider(dummy_model_config)
    assert provider.is_available() is True
    assert provider._unavailable_reason is None


@patch("urllib.request.urlopen")
def test_ollama_availability_model_missing(mock_urlopen, dummy_model_config, dummy_prompt):
    models_data = {
        "data": [
            {"id": "other-model:latest"},
        ]
    }
    mock_urlopen.return_value = make_mock_http_response(200, models_data)

    provider = OllamaProvider(dummy_model_config)
    assert provider.is_available() is False
    assert provider._unavailable_error_type == "ModelNotFoundError"
    assert "not installed in Ollama" in provider._unavailable_reason

    # execute_prompt should return structured BACKEND_UNAVAILABLE with ModelNotFoundError
    outcome = provider.execute_prompt(dummy_prompt, run_id="run-missing-model")
    assert outcome.status == "BACKEND_UNAVAILABLE"
    assert outcome.error_type == "ModelNotFoundError"
    assert outcome.prompt_tokens is None
    assert outcome.output_tokens is None


@patch("urllib.request.urlopen")
def test_ollama_availability_connection_error(mock_urlopen, dummy_model_config, dummy_prompt):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    provider = OllamaProvider(dummy_model_config)
    assert provider.is_available() is False
    assert provider._unavailable_error_type == "ConnectionError"

    outcome = provider.execute_prompt(dummy_prompt, run_id="run-conn-err")
    assert outcome.status == "BACKEND_UNAVAILABLE"
    assert outcome.error_type == "ConnectionError"


# 4. Request parameters: stream=false, temperature, seed, model tag
@patch("urllib.request.urlopen")
def test_ollama_request_payload_and_parameters(mock_urlopen, dummy_model_config):
    models_data = {"data": [{"id": "qwen3:1.7b"}]}
    chat_resp_data = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }

    mock_urlopen.side_effect = [
        make_mock_http_response(200, models_data),
        make_mock_http_response(200, chat_resp_data),
    ]

    provider = OllamaProvider(dummy_model_config, endpoint_mode="openai")
    gen_req = GenerationRequest(
        prompt="What is the capital of France?",
        max_output_tokens=64,
        temperature=0.0,
        seed=42,
        stop=["\n"],
    )

    result = provider.generate(gen_req)
    assert result.generated_text == "Paris"
    assert result.prompt_tokens == 10
    assert result.output_tokens == 2
    assert result.total_tokens == 12

    # Verify the second urlopen call was the chat completion request
    chat_call = mock_urlopen.call_args_list[1]
    req_obj = chat_call[0][0]
    assert req_obj.full_url.endswith("/v1/chat/completions")
    payload = json.loads(req_obj.data.decode("utf-8"))

    assert payload["model"] == "qwen3:1.7b"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 42
    assert payload["max_tokens"] == 64
    assert payload["stop"] == ["\n"]
    assert payload["messages"] == [{"role": "user", "content": "What is the capital of France?"}]


# 4b. Native Ollama /api/chat payload: options.num_predict, think flag, and done_reason
@patch("urllib.request.urlopen")
def test_ollama_native_request_payload_and_parameters(mock_urlopen, dummy_model_config):
    models_data = {"data": [{"id": "qwen3:1.7b"}]}
    native_chat_resp = {
        "model": "qwen3:1.7b",
        "message": {"role": "assistant", "content": "Paris"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 15,
        "eval_count": 3,
    }

    mock_urlopen.side_effect = [
        make_mock_http_response(200, models_data),
        make_mock_http_response(200, native_chat_resp),
    ]

    provider = OllamaProvider(dummy_model_config, endpoint_mode="native")
    gen_req = GenerationRequest(
        prompt="What is the capital of France?",
        max_output_tokens=128,
        temperature=0.0,
        seed=42,
        think=False,
    )

    result = provider.generate(gen_req)
    assert result.generated_text == "Paris"
    assert result.prompt_tokens == 15
    assert result.output_tokens == 3
    assert result.total_tokens == 18
    assert result.finish_reason == "stop"
    assert result.is_truncated is False

    # Verify call went to /api/chat with think: false and options.num_predict
    chat_call = mock_urlopen.call_args_list[1]
    req_obj = chat_call[0][0]
    assert req_obj.full_url.endswith("/api/chat")
    payload = json.loads(req_obj.data.decode("utf-8"))

    assert payload["model"] == "qwen3:1.7b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_predict"] == 128
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 42
    assert payload["messages"] == [{"role": "user", "content": "What is the capital of France?"}]



# 5. Token usage: present vs absent (do not fabricate zero)
@patch("urllib.request.urlopen")
def test_ollama_missing_usage_metadata(mock_urlopen, dummy_model_config):
    models_data = {"data": [{"id": "qwen3:1.7b"}]}
    chat_resp_no_usage = {
        "id": "chatcmpl-456",
        "choices": [{"message": {"role": "assistant", "content": "Answer without usage tokens."}, "finish_reason": "stop"}],
        # usage is missing
    }

    mock_urlopen.side_effect = [
        make_mock_http_response(200, models_data),
        make_mock_http_response(200, chat_resp_no_usage),
    ]

    provider = OllamaProvider(dummy_model_config)
    result = provider.generate(GenerationRequest(prompt="Hello"))
    assert result.generated_text == "Answer without usage tokens."
    assert result.prompt_tokens is None
    assert result.output_tokens is None
    assert result.total_tokens is None


# 6. Error handling: empty output and reasoning-only output
@patch("urllib.request.urlopen")
def test_ollama_empty_output_error(mock_urlopen, dummy_model_config, dummy_prompt):
    models_data = {"data": [{"id": "qwen3:1.7b"}]}
    empty_resp = {
        "id": "chatcmpl-empty",
        "choices": [{"message": {"role": "assistant", "content": "   "}, "finish_reason": "stop"}],
    }
    mock_urlopen.side_effect = [
        make_mock_http_response(200, models_data),
        make_mock_http_response(200, empty_resp),
        make_mock_http_response(200, empty_resp),
    ]

    provider = OllamaProvider(dummy_model_config)
    outcome = provider.execute_prompt(dummy_prompt, run_id="run-empty", max_retries=1)
    assert outcome.status == "INVALID_OUTPUT"
    assert outcome.error_type == "EmptyOutputError"


@patch("urllib.request.urlopen")
def test_ollama_reasoning_only_error(mock_urlopen, dummy_model_config, dummy_prompt):
    models_data = {"data": [{"id": "qwen3:1.7b"}]}
    reasoning_only_resp = {
        "id": "chatcmpl-think",
        "choices": [{"message": {"role": "assistant", "content": "<think>\nThinking about it...\n</think>"}, "finish_reason": "stop"}],
    }
    mock_urlopen.side_effect = [
        make_mock_http_response(200, models_data),
        make_mock_http_response(200, reasoning_only_resp),
        make_mock_http_response(200, reasoning_only_resp),
    ]

    provider = OllamaProvider(dummy_model_config)
    outcome = provider.execute_prompt(dummy_prompt, run_id="run-reasoning", max_retries=1)
    assert outcome.status == "INVALID_OUTPUT"
    assert outcome.error_type == "ReasoningOnlyError"


# 7. Malformed JSON and missing choices
@patch("urllib.request.urlopen")
def test_ollama_malformed_response(mock_urlopen, dummy_model_config, dummy_prompt):
    models_data = {"data": [{"id": "qwen3:1.7b"}]}
    malformed_resp = {"error": "Internal error", "no_choices": True}

    mock_urlopen.side_effect = [
        make_mock_http_response(200, models_data),
        make_mock_http_response(200, malformed_resp),
        make_mock_http_response(200, malformed_resp),
    ]

    provider = OllamaProvider(dummy_model_config)
    outcome = provider.execute_prompt(dummy_prompt, run_id="run-malformed", max_retries=1)
    assert outcome.status == "RETRY_EXHAUSTED"
    assert outcome.error_type == "MalformedResponseError"


# 8. Timeout handling and bounded retries
@patch("urllib.request.urlopen")
def test_ollama_timeout_and_retries(mock_urlopen, dummy_model_config, dummy_prompt):
    models_data = {"data": [{"id": "qwen3:1.7b"}]}
    success_resp = {
        "choices": [{"message": {"role": "assistant", "content": "Success after timeout"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    # First call: models list -> 200
    # Second call: timeout
    # Third call: success
    mock_urlopen.side_effect = [
        make_mock_http_response(200, models_data),
        TimeoutError("Request timed out"),
        make_mock_http_response(200, success_resp),
    ]

    provider = OllamaProvider(dummy_model_config)
    outcome = provider.execute_prompt(dummy_prompt, run_id="run-retry", max_retries=2)
    assert outcome.status == "SUCCESS"
    assert outcome.retry_count == 1
    assert len(outcome.attempt_history) == 2
    assert outcome.attempt_history[0]["status"] == "TIMEOUT"
    assert outcome.attempt_history[1]["status"] == "SUCCESS"
    assert outcome.generated_text == "Success after timeout"
