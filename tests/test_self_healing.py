import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nexora.backend.app.models import (
    Run,
    RunStatus,
    RoutingMode,
    ValidationStatus,
    FinalStatus,
)
from nexora.backend.app.core.router import (
    InferenceRequest,
    InferenceResponse,
    ProviderAdapter,
)
from nexora.backend.app.core.validator import ValidatorCritic
from nexora.backend.app.worker.orchestrator import RunOrchestrator


class MockProviderAdapter(ProviderAdapter):
    """Mock provider that returns sequential responses to simulate bug -> fix."""
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_requests = []

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.recorded_requests.append(request)
        resp = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return resp

    def provider_for(self, model_id: str) -> str:
        return "mock"


def _create_mock_orchestrator(run, mock_adapter, workspace_path=None):
    orch = RunOrchestrator.__new__(RunOrchestrator)
    orch.run = run
    orch.events = []
    orch.workspace_path = workspace_path
    orch.requested_model_key = "test-model"
    orch.provider_adapter = mock_adapter
    orch.context_optimizer = None
    orch.validator = None
    orch.current_diff = None

    mock_db = MagicMock()
    # Mock cached run result lookup to return None (no cache hit)
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_scalars.all.return_value = []
    mock_exec_result = MagicMock()
    mock_exec_result.scalars.return_value = mock_scalars
    mock_exec_result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=mock_exec_result)
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.get = AsyncMock(return_value=None)
    orch.db = mock_db

    mock_router = MagicMock()
    mock_decision = MagicMock()
    mock_decision.model.model_id = "test-model"
    mock_decision.model.display_name = "Test Model"
    mock_decision.model.context_capacity = 32000
    mock_decision.model.tier = "super"
    mock_decision.model.max_output_tokens = 2048
    mock_router.route.return_value = mock_decision
    orch.model_router = mock_router

    async def fake_add_event(event_type: str, stage: str, message: str):
        orch.events.append({"type": event_type, "stage": stage, "message": message})

    orch._add_event = fake_add_event
    return orch


@pytest.mark.asyncio
async def test_self_healing_recovers_from_syntax_error():
    """Verify that a syntax error in generated code triggers auto-correction and succeeds."""
    broken_code = "```python\ndef calculate(x, y):\n    return x + \n```"
    fixed_code = "```python\ndef calculate(x, y):\n    return x + y\n```"

    mock_adapter = MockProviderAdapter([
        InferenceResponse(
            content=broken_code,
            input_tokens=100,
            output_tokens=30,
            model_id="test-model",
        ),
        InferenceResponse(
            content=fixed_code,
            input_tokens=150,
            output_tokens=35,
            model_id="test-model",
        ),
    ])

    run = Run(
        task_text="Create a calculate function in python",
        workspace_id="default",
        budget_profile_id="dev",
        routing_mode=RoutingMode.AUTO,
    )
    run.id = "run-test-syntax-healing"
    run.steps_completed = 0
    run.input_tokens_used = 0
    run.output_tokens_used = 0
    run.snapshot_id = "snapshot-1"
    run.task_hash = "abc12345"

    orch = _create_mock_orchestrator(run, mock_adapter)

    # Execute
    await orch._execute_run()

    # Assertions
    assert mock_adapter.call_count == 2
    assert run.result_summary["self_healed"] is True
    assert run.result_summary["attempts"] == 2
    assert "Self-Healing Verified" in run.result_summary["content"]
    assert "def calculate(x, y):\n    return x + y" in run.result_summary["content"]
    assert orch.run.status == RunStatus.COMPLETED

    # Verify the self-healing correction prompt was delivered to the model
    healing_prompt = mock_adapter.recorded_requests[1].messages[-1]["content"]
    assert "SyntaxError" in healing_prompt
    assert any("Auto-correcting" in e["message"] for e in orch.events)


@pytest.mark.asyncio
async def test_self_healing_recovers_from_failing_tests():
    """Verify that failing targeted tests cause auto-correction with the test failure output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create a workspace test file
        test_file = tmp_path / "test_calculator.py"
        test_file.write_text(
            "from calculator import add\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        # Attempt 1 has bug (subtracts instead of adds)
        buggy_code = "```python\ndef add(a, b):\n    return a - b\n```"
        # Attempt 2 is fixed
        correct_code = "```python\ndef add(a, b):\n    return a + b\n```"

        mock_adapter = MockProviderAdapter([
            InferenceResponse(
                content=buggy_code,
                input_tokens=100,
                output_tokens=30,
                model_id="test-model",
            ),
            InferenceResponse(
                content=correct_code,
                input_tokens=200,
                output_tokens=30,
                model_id="test-model",
            ),
        ])

        run = Run(
            task_text="Create calculator.py with add function",
            workspace_id="default",
            budget_profile_id="dev",
            routing_mode=RoutingMode.AUTO,
        )
        run.id = "run-test-targeted-healing"
        run.steps_completed = 0
        run.input_tokens_used = 0
        run.output_tokens_used = 0
        run.snapshot_id = "snapshot-1"
        run.task_hash = "calc12345"

        orch = _create_mock_orchestrator(run, mock_adapter, workspace_path=tmp_path)
        orch.validator = ValidatorCritic(workspace_path=tmp_path)

        # Mock validator._run_targeted_tests to simulate pytest failing on attempt 1 and passing on attempt 2
        validation_results = [
            MagicMock(status=ValidationStatus.FAIL, details={"output": "assert -1 == 5\nFAILED test_calculator.py::test_add"}, message="Targeted tests failed"),
            MagicMock(status=ValidationStatus.PASS, details={"output": "1 passed in 0.05s"}, message="Targeted tests passed"),
        ]
        test_call_idx = 0

        def mock_run_tests(cmd):
            nonlocal test_call_idx
            res = validation_results[min(test_call_idx, len(validation_results) - 1)]
            test_call_idx += 1
            return res

        orch.validator._run_targeted_tests = mock_run_tests

        # Execute
        await orch._execute_run()

        # Assertions
        assert mock_adapter.call_count == 2
        assert run.result_summary["self_healed"] is True
        assert run.result_summary["attempts"] == 2
        assert "Self-Healing Verified" in run.result_summary["content"]
        assert orch.run.status == RunStatus.COMPLETED

        # Check prompt contained test failure message
        healing_prompt = mock_adapter.recorded_requests[1].messages[-1]["content"]
        assert "Targeted tests failed" in healing_prompt or "assert -1 == 5" in healing_prompt


@pytest.mark.asyncio
async def test_clean_task_bypasses_healing():
    """Verify that correct code on first attempt passes immediately without retries."""
    valid_code = "```python\ndef greet(name):\n    return f'Hello, {name}'\n```"

    mock_adapter = MockProviderAdapter([
        InferenceResponse(
            content=valid_code,
            input_tokens=80,
            output_tokens=25,
            model_id="test-model",
        )
    ])

    run = Run(
        task_text="Write greet function in python",
        workspace_id="default",
        budget_profile_id="dev",
        routing_mode=RoutingMode.AUTO,
    )
    run.id = "run-test-clean"
    run.steps_completed = 0
    run.input_tokens_used = 0
    run.output_tokens_used = 0
    run.snapshot_id = "snapshot-1"
    run.task_hash = "greet12345"

    orch = _create_mock_orchestrator(run, mock_adapter)

    # Execute
    await orch._execute_run()

    # Assertions
    assert mock_adapter.call_count == 1
    assert run.result_summary["self_healed"] is False
    assert run.result_summary["attempts"] == 1
    assert "Self-Healing Verified" not in run.result_summary["content"]
    assert orch.run.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_html_web_task_bypasses_python_syntax_and_healing():
    """Verify that HTML / Web / 3D Animation code does NOT crash Python syntax check or trigger false healing loops."""
    html_code = """```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>3D Animation</title>
</head>
<body>
  <div id="container"></div>
  <script>
    console.log("3D scene loaded");
  </script>
</body>
</html>
```"""

    mock_adapter = MockProviderAdapter([
        InferenceResponse(
            content=html_code,
            input_tokens=100,
            output_tokens=80,
            model_id="test-model",
        )
    ])

    run = Run(
        task_text="i want to create 3d animation website so give me code one by one",
        workspace_id="default",
        budget_profile_id="dev",
        routing_mode=RoutingMode.AUTO,
    )
    run.id = "run-test-html-web"
    run.steps_completed = 0
    run.input_tokens_used = 0
    run.output_tokens_used = 0
    run.snapshot_id = "snapshot-1"
    run.task_hash = "web3d12345"

    orch = _create_mock_orchestrator(run, mock_adapter)

    # Execute
    await orch._execute_run()

    # The run must complete on first attempt, NOT trigger self-healing, and NOT fail syntax check
    assert mock_adapter.call_count == 1
    assert run.result_summary["self_healed"] is False
    assert run.result_summary["attempts"] == 1
    assert orch.run.status == RunStatus.COMPLETED
    assert "3D Animation" in run.result_summary["content"]

