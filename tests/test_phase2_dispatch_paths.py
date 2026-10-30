"""
Phase 2 — Alternate dispatch path regression guard
==================================================
Phase 2 requires that authorization cannot be bypassed via direct calls or an
alternate dispatch path, not only the normal orchestrator flow.

In this codebase the authoritative policy gate lives in
``RunOrchestrator._process_tool_call``: it evaluates the (fail-closed) policy
engine and only forwards ``ALLOW`` / ``ALLOW_WITH_MONITORING`` to
``sandbox_adapter.execute``. ``SandboxAdapter.execute`` itself performs no
policy evaluation, so it must be reachable **only** from the orchestrator.

These tests are static source guards: they fail if a new call site starts
invoking the sandbox (or constructs a sandbox adapter directly) from anywhere
else -- which would create a policy-bypass path.
"""
import inspect
import re

import nexora.backend.app.api.package_routes as package_routes
import nexora.backend.app.api.routes as routes
import nexora.backend.app.core.tools as tools
import nexora.backend.app.worker.orchestrator as orchestrator

_BACKEND_MODULES = [
    routes,
    package_routes,
    tools,
    orchestrator,
]

# Realistic forms of a sandbox execution call site.
_SANDBOX_CALL = re.compile(
    r"(?:sandbox_adapter|get_sandbox_adapter\(\)|LocalSandboxAdapter[^.\n]*)"
    r"\s*\.\s*execute\s*\(",
    re.IGNORECASE,
)

# Direct construction of the sandbox adapter outside tools.py would also be an
# alternate path into privileged execution.
_SANDBOX_CONSTRUCT = re.compile(r"LocalSandboxAdapter\s*\(")


def test_sandbox_is_only_invoked_by_the_orchestrator():
    offenders = []
    for module in _BACKEND_MODULES:
        if module is orchestrator:
            continue
        source = inspect.getsource(module)
        if _SANDBOX_CALL.search(source):
            offenders.append(module.__name__)
    assert not offenders, (
        "sandbox execution reached from outside the orchestrator (policy bypass): "
        + ", ".join(offenders)
    )


def test_orchestrator_is_the_expected_single_call_site():
    """Sanity: the orchestrator must still actually call the sandbox."""
    source = inspect.getsource(orchestrator)
    assert _SANDBOX_CALL.search(source), (
        "expected the orchestrator to invoke sandbox_adapter.execute"
    )


def test_sandbox_adapter_is_not_constructed_outside_tools():
    offenders = []
    for module in _BACKEND_MODULES:
        if module is tools:
            continue
        source = inspect.getsource(module)
        if _SANDBOX_CONSTRUCT.search(source):
            offenders.append(module.__name__)
    assert not offenders, (
        "LocalSandboxAdapter constructed outside tools.py: " + ", ".join(offenders)
    )


def test_sandbox_adapter_performs_no_policy_evaluation():
    """Document the contract: the sandbox is NOT an authorization boundary.

    If this ever becomes false, the orchestrator gate is no longer the sole
    enforcement point and the dispatch tests above should be revisited.
    """
    source = inspect.getsource(tools)
    adapter_source = source[source.index("class SandboxAdapter"):]
    assert "PolicyEngine" not in adapter_source
    assert "policy_engine" not in adapter_source
