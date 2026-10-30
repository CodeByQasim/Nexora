import pytest
from nexora.backend.app.worker.orchestrator import (
    _detect_code_content,
    _is_code_task_prompt,
    _is_conversational_prompt,
    extract_clean_code,
)


def test_detect_code_content_css():
    css_sample = """/* ---- Header/Hero ---- */
header {
  background: #004466;
  color: #fff;
  text-align: center;
  padding: 4rem 1rem;
}
header h1 {
  font-size: 2.8rem;
}"""
    assert _detect_code_content(css_sample) is True


def test_detect_code_content_html():
    html_sample = """<div class="hero-container">
  <h1>Welcome</h1>
  <p>Intro text</p>
</div>"""
    assert _detect_code_content(html_sample) is True


def test_detect_code_content_fenced():
    fenced_sample = "```css\nbody { margin: 0; }\n```"
    assert _detect_code_content(fenced_sample) is True


def test_detect_code_content_plain_chat():
    chat_sample = "Hello there! I am Nexora AI. How can I help you today?"
    assert _detect_code_content(chat_sample) is False
    assert _is_conversational_prompt("hello") is True


def test_code_task_target_expansion():
    assert _is_code_task_prompt("create a navbar with css") is True
    assert _is_code_task_prompt("build a hero header section") is True
    assert _is_code_task_prompt("make a responsive button style") is True

