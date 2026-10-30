import pytest
import tempfile
from pathlib import Path
from nexora.backend.app.core.context import (
    skeletonize_python,
    ContextOptimizer,
    truncate_head_tail,
    align_prefix_boundary,
)
from nexora.backend.app.core.tools import ToolRouter, ToolValidationError
from nexora.backend.app.core.router import InferenceResponse, get_provider_adapter
from nexora.backend.app.models import ModelRegistry
from nexora.backend.app.worker.orchestrator import (
    calculate_dynamic_output_budget,
    extract_clean_code,
)


def test_skeletonize_python_preserves_signatures_and_docstrings():
    sample_code = '''
import os
from typing import List, Optional

class DataStore:
    """Store data in memory."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.items = []

    def get_items(self) -> List[str]:
        """Return all items."""
        return [str(x) for x in self.items]

def process_data(store: DataStore, flag: bool = True) -> Optional[int]:
    """Process data function."""
    if not flag:
        return None
    return len(store.items)
'''
    skeleton = skeletonize_python(sample_code)
    
    # Signatures, imports, and docstrings must be present
    assert "import os" in skeleton
    assert "class DataStore:" in skeleton
    assert '"""Store data in memory."""' in skeleton
    assert "def __init__(self, capacity: int=100):" in skeleton or "def __init__(self, capacity: int = 100):" in skeleton
    assert "def get_items(self) -> List[str]:" in skeleton
    assert '"""Return all items."""' in skeleton
    assert "def process_data(store: DataStore, flag: bool=True) -> Optional[int]:" in skeleton or "def process_data(store: DataStore, flag: bool = True) -> Optional[int]:" in skeleton
    assert '"""Process data function."""' in skeleton
    
    # Internal bodies must be replaced
    assert "self.capacity = capacity" not in skeleton
    assert "return [str(x) for x in self.items]" not in skeleton
    assert "return len(store.items)" not in skeleton


def test_skeletonize_fallback_on_invalid_syntax():
    invalid_code = "def invalid_func(::: broken syntax"
    # Should safely return source code without crashing
    assert skeletonize_python(invalid_code) == invalid_code


def test_context_optimizer_safeguard_and_caching():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target_file = root / "calculator.py"
        target_file.write_text("""
class Calculator:
    \"\"\"Calculator class.\"\"\"
    def add(self, a: int, b: int) -> int:
        return a + b
""", encoding="utf-8")

        ref_file = root / "utils.py"
        ref_file.write_text("""
import math

def helper_function(val: float) -> float:
    \"\"\"Helper calculation.\"\"\"
    intermediate = math.sqrt(val) * 2.5
    return intermediate + 42.0
""", encoding="utf-8")

        optimizer = ContextOptimizer(str(root))
        model = ModelRegistry(
            model_id="test-model",
            display_name="Test",
            context_capacity=8192,
            max_output_tokens=1024,
        )

        # Query targeting calculator specifically
        pkg1 = optimizer.build_context(
            task_query="Fix addition bug in calculator.py",
            model=model,
        )

        included_map = {f["path"]: f for f in pkg1.included_files}
        
        # Calculator was targeted -> must have full content
        if "calculator.py" in included_map:
            assert included_map["calculator.py"]["content_mode"] == "full"
            assert "return a + b" in included_map["calculator.py"]["content"]

        # Run 2 on same query -> should hit cache
        pkg2 = optimizer.build_context(
            task_query="Fix addition bug in calculator.py",
            model=model,
        )
        assert pkg2.cache_hits >= 1


def test_skeletonize_preserves_decorators_and_async():
    sample = '''
from dataclasses import dataclass

@dataclass
class ServiceConfig:
    timeout: int = 30
    retry_count: int = 3

    @property
    def is_valid(self) -> bool:
        """Validate config values."""
        return self.timeout > 0 and self.retry_count >= 0

    @classmethod
    async def fetch_async(cls, endpoint: str) -> "ServiceConfig":
        """Asynchronously load config."""
        res = await some_http_client(endpoint)
        return cls(timeout=res.timeout)
'''
    skeleton = skeletonize_python(sample)

    # Decorators must be preserved
    assert "@dataclass" in skeleton
    assert "@property" in skeleton
    assert "@classmethod" in skeleton

    # Async keyword and signature preserved
    assert "async def fetch_async(cls, endpoint: str) -> 'ServiceConfig':" in skeleton or "async def fetch_async(cls, endpoint: str) -> \"ServiceConfig\":" in skeleton
    assert '"""Asynchronously load config."""' in skeleton

    # Class attributes preserved
    assert "timeout: int = 30" in skeleton
    assert "retry_count: int = 3" in skeleton

    # Body implementation replaced with pass
    assert "return self.timeout > 0" not in skeleton
    assert "res = await some_http_client" not in skeleton


def test_dynamic_token_budget_calculation():
    # Simple DSA or short utility query (< 12 words without fix/add/test/implement) -> 512
    assert calculate_dynamic_output_budget("Two sum in Python", ceiling=2000) == 512
    assert calculate_dynamic_output_budget("Binary search algorithm", ceiling=2000) == 512

    # Standard feature / fix / test / implement -> 1024
    assert calculate_dynamic_output_budget("Fix the user authentication token validation logic", ceiling=2000) == 1024
    assert calculate_dynamic_output_budget("Add unit tests for payment webhook processing", ceiling=2000) == 1024
    assert calculate_dynamic_output_budget("Implement a custom LRU cache class", ceiling=2000) == 1024

    # Complex refactor / architecture -> 1536
    assert calculate_dynamic_output_budget("Refactor the multi-tenant architecture and system data model", ceiling=2000) == 1536

    # Web / Animation / 3D tasks -> 3072 (clamped to ceiling if lower)
    assert calculate_dynamic_output_budget("Create 3d animation website with threejs", ceiling=4096) == 3072
    assert calculate_dynamic_output_budget("Build an interactive landing page with animations", ceiling=4096) == 3072

    # Clamping against ceiling
    assert calculate_dynamic_output_budget("Refactor database layer", ceiling=500) == 500
    assert calculate_dynamic_output_budget("Two sum in Python", ceiling=256) == 256


def test_adaptive_context_budget_scaling():
    from nexora.backend.app.core.context import calculate_adaptive_budget
    from types import SimpleNamespace

    # 1. Nano Tier (e.g. 16k capacity)
    nano_model = SimpleNamespace(
        model_id="nvidia/nemotron-nano",
        display_name="Nemotron Nano",
        tier="nano",
        context_capacity=16384,
        max_output_tokens=2048,
    )
    nano_budget = calculate_adaptive_budget(nano_model)
    assert nano_budget.tier == "nano"
    assert nano_budget.context_capacity == 16384
    assert nano_budget.max_output_tokens == 2048
    assert nano_budget.history_budget_tokens >= 2000
    assert nano_budget.repo_context_budget_tokens > 5000

    # 2. Super Tier (e.g. 64k capacity)
    super_model = SimpleNamespace(
        model_id="nvidia/nemotron-super",
        display_name="Nemotron Super",
        tier="super",
        context_capacity=65536,
        max_output_tokens=4096,
    )
    super_budget = calculate_adaptive_budget(super_model)
    assert super_budget.tier == "super"
    assert super_budget.history_budget_tokens > nano_budget.history_budget_tokens
    assert super_budget.repo_context_budget_tokens > nano_budget.repo_context_budget_tokens

    # 3. Ultra Tier (e.g. 131k Nemotron Ultra)
    ultra_model = SimpleNamespace(
        model_id="nvidia/nemotron-ultra",
        display_name="Nemotron Ultra",
        tier="ultra",
        context_capacity=131072,
        max_output_tokens=4096,
    )
    ultra_budget = calculate_adaptive_budget(ultra_model)
    assert ultra_budget.tier == "ultra"
    assert ultra_budget.history_budget_tokens >= 25000
    assert ultra_budget.repo_context_budget_tokens >= 50000

    # 4. Respecting user BudgetProfile constraint
    profile = SimpleNamespace(
        name="tight-profile",
        max_input_tokens=8000,
        max_output_tokens=1024,
    )
    profiled_budget = calculate_adaptive_budget(ultra_model, budget_profile=profile)
    assert profiled_budget.max_output_tokens == 1024
    assert profiled_budget.usable_input_tokens <= 8000


def test_extract_clean_code_preserves_docstrings_with_backticks():
    # Code with a markdown example inside its docstring
    raw_code = '''def markdown_parser(text: str) -> str:
    """
    Parse markdown snippet.

    Example:
    ```
    parsed = parser("# Header")
    ```

    Returns the formatted text.
    """
    return text.strip()'''
    # When text does not start with an outer markdown fence, extract_clean_code preserves it verbatim
    extracted = extract_clean_code(raw_code)
    assert extracted == raw_code
    assert "```\n    parsed = parser" in extracted
    assert 'Returns the formatted text.' in extracted


def test_extract_clean_code_strips_outer_fence_and_trailing_explanation():
    model_output = '''```python
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

### Explanation
This function performs addition of two integers.
'''
    cleaned = extract_clean_code(model_output)
    assert "def add(a: int, b: int) -> int:" in cleaned
    assert "return a + b" in cleaned
    assert "```" not in cleaned
    assert "### Explanation" not in cleaned
    assert "This function performs addition" not in cleaned


def test_truncate_head_tail_short_and_none_text():
    assert truncate_head_tail(None) == ""
    assert truncate_head_tail("") == ""
    short = "short single line output"
    assert truncate_head_tail(short, max_chars=100) == short
    # Non-string coercion
    assert truncate_head_tail(12345, max_chars=100) == "12345"


def test_truncate_head_tail_preserves_head_and_tail_tracebacks():
    lines = [f"line_{i:03d} = 'some data filler text for padding'" for i in range(100)]
    lines[0] = "HEADER_START = 'initialization imports and config'"
    lines[-1] = "FAILED: assert result == expected  # exact traceback line"
    full_text = "\n".join(lines)

    truncated = truncate_head_tail(full_text, max_chars=400)

    # 1. Total length is safely bounded
    assert len(truncated) <= len(full_text)
    # 2. Top header is preserved
    assert "HEADER_START" in truncated
    # 3. Bottom failure assertion / traceback is preserved
    assert "FAILED: assert result == expected" in truncated
    # 4. Informative omission marker is included
    assert "Omitted" in truncated
    assert "head & tail preserved" in truncated


def test_truncate_head_tail_boundary_snapping():
    text = (
        "def first_function():\n"
        "    return 1\n\n"
        "def middle_function():\n"
        "    x = 100\n"
        "    y = 200\n"
        "    return x + y\n\n"
        "def final_function():\n"
        "    return 999\n"
    )
    # Truncate to modest limit
    truncated = truncate_head_tail(text, max_chars=100, head_ratio=0.45, tail_ratio=0.45)
    # Head and tail functions must be intact
    assert "def first_function():" in truncated
    assert "def final_function():" in truncated
    assert "return 999" in truncated


def test_align_prefix_boundary():
    assert align_prefix_boundary("") == ""
    sample = "You are Nexora system prompt.\n\n"
    aligned = align_prefix_boundary(sample)
    assert aligned.endswith("\n\n")
    assert not aligned.endswith("\n\n\n")

    dirty = "System prompt without trailing newline"
    dirty_aligned = align_prefix_boundary(dirty)
    assert dirty_aligned == "System prompt without trailing newline\n\n"


def test_tool_schemas_deterministic_sorting():
    router = ToolRouter()
    schemas1 = router.get_all_schemas(underscore_safe=True)
    schemas2 = router.get_all_schemas(underscore_safe=True)

    # 1. Byte-for-byte deterministic equality across repeated calls
    assert schemas1 == schemas2

    # 2. Tool names are strictly sorted alphabetically to maximize KV-cache prefix hits
    names = [s["function"]["name"] for s in schemas1]
    assert names == sorted(names)
    assert len(names) >= 5


def test_inference_response_tracks_cached_tokens():
    resp = InferenceResponse(
        content="Clean response",
        input_tokens=2000,
        output_tokens=500,
        total_tokens=2500,
        cached_tokens=1600,
    )
    assert resp.cached_tokens == 1600

    # Test normalization from raw provider payload
    adapter = get_provider_adapter()
    raw_payload = {
        "id": "chatcmpl-test",
        "model": "nvidia/nemotron-ultra",
        "choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 3000,
            "completion_tokens": 200,
            "total_tokens": 3200,
            "prompt_tokens_details": {
                "cached_tokens": 2400
            }
        }
    }
    normalized = adapter._normalize_response(raw_payload, "nvidia/nemotron-ultra")
    assert normalized.cached_tokens == 2400
    assert normalized.input_tokens == 3000


def test_context_optimizer_sorts_included_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create files in arbitrary order
        (root / "z_module.py").write_text("def z(): pass\n", encoding="utf-8")
        (root / "a_module.py").write_text("def a(): pass\n", encoding="utf-8")
        (root / "m_module.py").write_text("def m(): pass\n", encoding="utf-8")

        optimizer = ContextOptimizer(str(root))
        model = ModelRegistry(
            model_id="test-model",
            display_name="Test",
            context_capacity=8192,
            max_output_tokens=1024,
        )

        pkg = optimizer.build_context(
            task_query="Inspect modules in workspace",
            model=model,
        )

        paths = [f["path"] for f in pkg.included_files]
        # Paths must be deterministically sorted
        assert paths == sorted(paths)


def test_is_conversational_prompt_detection():
    from nexora.backend.app.worker.orchestrator import _is_conversational_prompt

    # Greetings and general chat should be detected as conversational
    assert _is_conversational_prompt("hi") is True
    assert _is_conversational_prompt("Hi!") is True
    assert _is_conversational_prompt("hello") is True
    assert _is_conversational_prompt("Hey there") is True
    assert _is_conversational_prompt("who are you?") is True
    assert _is_conversational_prompt("what can you do?") is True
    assert _is_conversational_prompt("good morning") is True
    assert _is_conversational_prompt("help") is True
    assert _is_conversational_prompt("") is True

    # Real coding or repository tasks should NOT be conversational
    assert _is_conversational_prompt("Create index.html with a 3D animated hero section") is False
    assert _is_conversational_prompt("Fix bug in the authentication module") is False
    assert _is_conversational_prompt("Write a python script to parse logs") is False
    assert _is_conversational_prompt("Run test suite on api endpoints") is False


def test_tool_argument_normalization_and_aliasing():
    from nexora.backend.app.core.tools import ToolRouter, ToolName

    router = ToolRouter()

    # 1. repo.write: 'code' alias mapped to 'content'
    req1 = router.validate_and_route(
        "repo.write",
        {"path": "index.html", "code": "<h1>Test</h1>"},
        "req-1",
    )
    assert req1.arguments["path"] == "index.html"
    assert req1.arguments["content"] == "<h1>Test</h1>"
    assert "code" not in req1.arguments

    # 2. repo.write: 'file' alias mapped to 'path'
    req2 = router.validate_and_route(
        "repo.write",
        {"file": "app.js", "content": "console.log(1);"},
        "req-2",
    )
    assert req2.arguments["path"] == "app.js"
    assert req2.arguments["content"] == "console.log(1);"

    # 3. repo.write: missing content must FAIL CLOSED (never silently write an
    # empty file) — empty overwrites were reported as a false "Written 0 bytes"
    # success for large payloads.
    with pytest.raises(ToolValidationError) as missing_content:
        router.validate_and_route("repo.write", {"path": "empty.txt"}, "req-3")
    assert "content" in str(missing_content.value)

    # 4. repo.write: harmless LLM metadata (explanation, comment) stripped safely
    req4 = router.validate_and_route(
        "repo.write",
        {"path": "solution.py", "content": "x = 1", "explanation": "setting x to 1"},
        "req-4",
    )
    assert req4.arguments["path"] == "solution.py"
    assert req4.arguments["content"] == "x = 1"
    assert "explanation" not in req4.arguments

    # 5. test.run: 'cmd' alias mapped to 'command'
    req5 = router.validate_and_route(
        "test.run",
        {"cmd": "pytest -v tests/test_api.py"},
        "req-5",
    )
    assert req5.arguments["command"] == "pytest -v tests/test_api.py"


