import pytest
from nexora.backend.app.worker.orchestrator import (
    _clean_reasoning_and_tags,
    _is_internal_reasoning,
    extract_clean_code,
    calculate_dynamic_output_budget,
)


def test_clean_reasoning_and_tags_strips_thinking_blocks_and_orphans():
    raw_leak = (
        "Personal Introduction\n"
        "My name is Azizullah Memon. I am interested in Artificial Intelligence (AI), Machine\n"
        "Loaded repository context (2 files) to guide execution.\n"
        "Validating implementation against syntax and automated checks.. </thought>\n"
        "Task executed successfully."
    )
    cleaned = _clean_reasoning_and_tags(raw_leak)
    assert "</thought>" not in cleaned
    assert "<thought>" not in cleaned
    assert "Azizullah Memon" in cleaned
    assert "Task executed successfully." in cleaned


def test_clean_reasoning_strips_complete_thought_tags():
    raw = "<thought>Thinking about the portfolio structure and styles...</thought>\n\n<!DOCTYPE html><html><body><h1>Azizullah Memon</h1></body></html>"
    cleaned = _clean_reasoning_and_tags(raw)
    assert "<thought>" not in cleaned
    assert "</thought>" not in cleaned
    assert "Thinking about the portfolio" not in cleaned
    assert "<!DOCTYPE html>" in cleaned


def test_extract_clean_code_from_markdown_intro():
    intro_and_code = (
        "Here is the requested responsive portfolio in HTML, CSS and JS:\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head><title>Portfolio</title></head>\n"
        "<body><h1>Azizullah Memon</h1></body>\n"
        "</html>\n"
        "```\n"
        "You can customize the sections as needed."
    )
    code = extract_clean_code(intro_and_code)
    assert code.startswith("<!DOCTYPE html>")
    assert "<h1>Azizullah Memon</h1>" in code
    assert "Here is the requested" not in code


def test_portfolio_prompt_gets_4096_output_budget():
    portfolio_prompt = (
        "Portfolio Requirements\n"
        "When creating my personal portfolio:\n"
        "1. Highlight my AI coding-agent security work and Pakistan Multimodal AI Product Search Engine concept.\n"
        "2. Distinguish implemented projects from proposals and work in progress.\n"
        "generate a portfolio code in proper manner in html css and in javascript"
    )
    budget = calculate_dynamic_output_budget(portfolio_prompt, ceiling=4096)
    assert budget == 4096


def test_follow_up_detailed_code_with_thread_context_gets_4096_output_budget():
    follow_up_prompt = "can you give me detailed code not simple"
    thread_context = "generate a portfolio website in html css and javascript for Azizullah Memon"
    
    # Without thread context, short prompt with 'detailed' and 'not simple' still gets 4096
    direct_budget = calculate_dynamic_output_budget(follow_up_prompt, ceiling=4096)
    assert direct_budget == 4096

    # With thread context passed
    budget = calculate_dynamic_output_budget(follow_up_prompt, ceiling=4096, thread_context=thread_context)
    assert budget == 4096


def test_is_code_task_prompt_matches_give_detailed_code():
    from nexora.backend.app.worker.orchestrator import _is_code_task_prompt

    assert _is_code_task_prompt("can you give me detailed code not simple") is True
    assert _is_code_task_prompt("show me the complete portfolio layout") is True
    assert _is_code_task_prompt("make a responsive navigation component") is True


def test_orchestrator_healing_mode_defaults_and_override():
    from nexora.backend.app.worker.orchestrator import RunOrchestrator
    from unittest.mock import MagicMock

    mock_db = MagicMock()
    orch = RunOrchestrator(mock_db)
    assert orch.healing_mode == "auto"

    orch.healing_mode = "off"
    assert orch.healing_mode == "off"

    orch.healing_mode = "on"
    assert orch.healing_mode == "on"


