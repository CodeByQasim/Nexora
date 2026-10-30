import pytest
from nexora.backend.app.core.context import (
    ConversationTurn,
    HistoryCompactor,
    compact_code_blocks_in_text,
    prune_conversation_history,
    distill_session_state,
    compact_thread_history,
)


def test_compact_code_blocks_in_text():
    sample_markdown = """Here is your implementation:
```python
class Node:
    '''Node representation.'''
    def __init__(self, val: int):
        self.val = val
        self.next = None

def reverse_linked_list(head: Node) -> Node:
    '''Reverse a singly linked list.'''
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```
Please let me know if you need unit tests!"""

    compacted = compact_code_blocks_in_text(sample_markdown)

    # Class and function signatures + docstrings must be preserved
    assert "class Node:" in compacted
    assert "Node representation." in compacted
    assert "def reverse_linked_list(head: Node) -> Node:" in compacted
    assert "Reverse a singly linked list." in compacted
    assert "pass" in compacted

    # Heavy function bodies must be stripped
    assert "curr.next = prev" not in compacted
    assert "prev = curr" not in compacted
    assert len(compacted) < len(sample_markdown)


def test_history_compactor_retains_latest_turn_full_fidelity():
    turn = ConversationTurn(
        user_prompt="Explain binary search",
        assistant_response="Binary search works by repeatedly halving the search interval.",
    )
    compactor = HistoryCompactor(max_history_tokens=2000)
    messages, telemetry = compactor.prune_and_compact([turn])

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Explain binary search"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Binary search works by repeatedly halving the search interval."
    assert telemetry["total_turns_input"] == 1
    assert telemetry["turns_retained"] == 1


def test_history_compactor_skeletonizes_older_turns():
    turn1 = ConversationTurn(
        user_prompt="Write a stack implementation",
        assistant_response="""```python
class Stack:
    '''LIFO Stack.'''
    def __init__(self):
        self._items = []

    def push(self, item):
        '''Push item.'''
        self._items.append(item)
        return True

    def pop(self):
        '''Pop item.'''
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()
```
Here is the stack code.""",
    )
    turn2 = ConversationTurn(
        user_prompt="Now add peek method",
        assistant_response="```python\ndef peek(self):\n    return self._items[-1]\n```",
    )

    compactor = HistoryCompactor(max_history_tokens=2000)
    messages, telemetry = compactor.prune_and_compact([turn1, turn2])

    assert len(messages) == 4
    # Turn 1 code should be skeletonized
    turn1_assistant = messages[1]["content"]
    assert "class Stack:" in turn1_assistant
    assert "self._items.append(item)" not in turn1_assistant
    assert "pass" in turn1_assistant

    # Turn 2 (latest turn) must remain uncompacted with full body
    turn2_assistant = messages[3]["content"]
    assert "return self._items[-1]" in turn2_assistant

    assert telemetry["total_turns_input"] == 2
    assert telemetry["turns_retained"] == 2
    assert telemetry["tokens_saved_pct"] > 0


def test_history_compactor_enforces_budget_sliding_window():
    # Create 5 large turns
    turns = []
    for i in range(5):
        turns.append(ConversationTurn(
            user_prompt=f"Task number {i} " + ("word " * 50),
            assistant_response=f"Response number {i} " + ("answer " * 150),
        ))

    # Strict token budget of 400 tokens
    compactor = HistoryCompactor(max_history_tokens=400)
    messages, telemetry = compactor.prune_and_compact(turns)

    # Should retain the latest turns and drop oldest turns to fit budget
    assert telemetry["turns_retained"] < 5
    assert telemetry["turns_retained"] >= 1
    # Latest turn must always be present
    assert "Task number 4" in messages[-2]["content"]


def test_distill_session_state_extracts_files_and_milestones():
    turns = [
        ConversationTurn(
            user_prompt="Build 3D hero in index.html with styles.css",
            assistant_response="Created index.html with canvas and styles.css for styling.",
        ),
        ConversationTurn(
            user_prompt="Add particle physics in app.js",
            assistant_response="Implemented particle physics engine in app.js.",
        ),
        ConversationTurn(
            user_prompt="Fix camera rotation bounds",
            assistant_response="Resolved issue with camera orbit bounds.",
        ),
    ]

    session_state = distill_session_state(turns)

    assert "<session_state>" in session_state
    assert "</session_state>" in session_state
    assert "Build 3D hero in index.html" in session_state
    # Check that referenced files were discovered
    assert "index.html" in session_state
    assert "styles.css" in session_state
    assert "app.js" in session_state
    # Check that historical milestones are logged
    assert "Turn 1" in session_state
    assert "Turn 2" in session_state
    assert "Turn 3" in session_state


def test_maybe_compact_generates_session_state_anchor():
    turns = [
        ConversationTurn(
            user_prompt=f"Feature step {i} for module_{i}.py",
            assistant_response=f"Implemented feature step {i} in module_{i}.py with code:\n" + ("def helper(): pass\n" * 15),
        )
        for i in range(6)
    ]

    compactor = HistoryCompactor(max_history_tokens=600)
    messages, telemetry = compactor.maybe_compact(turns, preserve_recent_turns=2)

    # 1. Telemetry confirms session state generated
    assert telemetry["session_state_generated"] is True
    assert telemetry["turns_condensed"] > 0
    assert telemetry["turns_retained"] == 2
    assert telemetry["tokens_saved_pct"] > 0

    # 2. First message must be the <session_state> block
    assert messages[0]["role"] == "user"
    assert "<session_state>" in messages[0]["content"]

    # 3. Second message must be the assistant acknowledgement
    assert messages[1]["role"] == "assistant"
    assert "Acknowledged" in messages[1]["content"]

    # 4. Recent turns preserved at the tail
    assert "Feature step 5" in messages[-2]["content"]
    assert "module_5.py" in messages[-1]["content"]


def test_compact_thread_history_compresses_chat_messages():
    # 5 turns (10 messages in OpenAI role-content format)
    thread = []
    for i in range(5):
        thread.append({"role": "user", "content": f"Task {i}: add component_{i}.jsx"})
        thread.append({"role": "assistant", "content": f"Created component_{i}.jsx with full implementation details."})

    compacted_messages, telemetry = compact_thread_history(thread, max_history_tokens=500, preserve_recent_turns=2)

    # Output should include session_state and recent turns
    assert telemetry["session_state_generated"] is True
    assert any("<session_state>" in m["content"] for m in compacted_messages)
    # The newest prompt must be in the output
    assert any("Task 4" in m["content"] for m in compacted_messages)

