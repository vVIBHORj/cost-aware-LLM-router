from typing import Any


class CodeExecutionScorer:
    """
    Interface and safe stub for evaluating code synthesis (HumanEval / MBPP).
    DO NOT execute model-generated code directly on the host system.
    Returns explicit SANDBOX_UNAVAILABLE status unless an isolated containerized sandbox is attached.
    """

    def __init__(self, sandbox_available: bool = False) -> None:
        self.sandbox_available = sandbox_available

    def is_available(self) -> bool:
        return self.sandbox_available

    def score(
        self,
        prediction: str,
        reference: str,
        timeout_seconds: float = 5.0,
    ) -> dict[str, Any]:
        """
        Evaluate code against unit tests in isolated sandbox.
        Safely returns SANDBOX_UNAVAILABLE when host execution is prohibited.
        """
        if not self.sandbox_available:
            return {
                "quality_score": None,
                "pass_at_1": None,
                "scoring_status": "SANDBOX_UNAVAILABLE",
                "scoring_error_type": "SandboxUnavailableError",
                "scoring_error_message": (
                    "Direct host code execution is prohibited to prevent unsafe code execution. "
                    "A containerized execution sandbox (Docker/gVisor) is required for code execution evaluation."
                ),
                "details": {
                    "prediction_length": len(prediction),
                    "test_spec_present": bool(reference and reference.strip()),
                },
            }

        # If sandbox attached in future phase:
        return {
            "quality_score": None,
            "pass_at_1": None,
            "scoring_status": "NOT_IMPLEMENTED",
            "scoring_error_type": "SandboxNotConfiguredError",
            "scoring_error_message": "External sandbox runner is not configured.",
            "details": {},
        }
