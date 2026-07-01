from unittest.mock import MagicMock

from moatless.agent.agent import ActionAgent
from moatless.flow.loop import AgenticLoop
from moatless.node import Node


def _build_flow_with_two_nodes(child_terminal: bool) -> AgenticLoop:
    root = Node.create_root(user_message="root")
    child = Node(node_id=1, parent=root, terminal=child_terminal)
    root.add_child(child)

    mock_agent = MagicMock(spec=ActionAgent)
    return AgenticLoop.create(
        root=root,
        agent=mock_agent,
        project_id="test-project",
        trajectory_id="test-trajectory",
        max_iterations=2,
    )


def test_is_finished_prefers_terminal_over_max_iterations():
    flow = _build_flow_with_two_nodes(child_terminal=True)
    assert flow.is_finished() == "terminal"


def test_is_finished_returns_max_iterations_when_not_terminal():
    flow = _build_flow_with_two_nodes(child_terminal=False)
    assert flow.is_finished() == "max_iterations"

