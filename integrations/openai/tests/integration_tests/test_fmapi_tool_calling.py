"""
End-to-end FMAPI tool calling tests mirroring app-templates CUJs.

These tests replicate the exact user code patterns from app-templates
(agent-openai-agents-sdk) to verify that single-turn and multi-turn
conversations don't break.

Naturally exercises regressions like:
  - PR #269: Agents SDK adds strict:True -> our client strips it -> FMAPI
  - PR #333: Multi-turn agent loop replays assistant messages with empty
    content + tool_calls -> our client fixes content -> FMAPI

Prerequisites:
- FMAPI endpoints must be available on the test workspace
- echo_message UC function in integration_testing.databricks_ai_bridge_mcp_test
"""

from __future__ import annotations

import os

import pytest
from databricks.sdk import WorkspaceClient

from databricks_openai import AsyncDatabricksOpenAI

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FMAPI_TOOL_CALLING_TESTS") != "1",
    reason="FMAPI tool calling tests disabled. Set RUN_FMAPI_TOOL_CALLING_TESTS=1 to enable.",
)

_FOUNDATION_MODELS = [
    "databricks-claude-3-7-sonnet",
    "databricks-meta-llama-3-3-70b-instruct",
]

# MCP test infrastructure
_MCP_CATALOG = "integration_testing"
_MCP_SCHEMA = "databricks_ai_bridge_mcp_test"
_MCP_FUNCTION = "echo_message"


@pytest.fixture(scope="module")
def workspace_client():
    return WorkspaceClient()


@pytest.fixture(scope="module")
def async_client(workspace_client):
    return AsyncDatabricksOpenAI(workspace_client=workspace_client)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
class TestAgentToolCalling:
    """End-to-end agent tests mirroring app-templates/agent-openai-agents-sdk.

    Each test follows the exact pattern users deploy:
      AsyncDatabricksOpenAI -> set_default_openai_client -> McpServer -> Agent -> Runner.run
    """

    async def test_single_turn(self, async_client, workspace_client, model):
        """Single-turn conversation: user sends one message, agent calls a tool and responds.

        Mirrors the basic app-template @invoke() handler.
        """
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem

        from databricks_openai.agents import McpServer

        set_default_openai_client(async_client)
        set_default_openai_api("chat_completions")

        async with McpServer.from_uc_function(
            catalog=_MCP_CATALOG,
            schema=_MCP_SCHEMA,
            function_name=_MCP_FUNCTION,
            workspace_client=workspace_client,
        ) as server:
            agent = Agent(
                name="echo-agent",
                instructions="Use the echo_message tool to echo messages when asked.",
                model=model,
                mcp_servers=[server],
            )
            result = await Runner.run(agent, "Echo the message 'hello from FMAPI test'")

            # Agent should have produced a final text output containing the echoed message
            assert result.final_output is not None
            assert "hello from FMAPI test" in result.final_output

            # The agent loop should have generated: tool_call -> tool_output -> message
            item_types = [type(item) for item in result.new_items]
            assert ToolCallItem in item_types, f"Expected a tool call, got: {item_types}"
            assert ToolCallOutputItem in item_types, f"Expected tool output, got: {item_types}"
            assert MessageOutputItem in item_types, f"Expected a message, got: {item_types}"

            # to_input_list should produce a valid conversation history
            # (this is what gets sent to FMAPI on the next turn)
            input_list = result.to_input_list()
            assert len(input_list) > 1, "Expected multi-item conversation history"

    async def test_multi_turn(self, async_client, workspace_client, model):
        """Multi-turn conversation: simulates a chat UI sending conversation history.

        First turn: user asks to echo a message, agent calls the tool.
        Second turn: user sends a followup with the full conversation history
        (including the assistant's prior tool-calling turn), agent calls the tool again.

        This is how the app-templates chat UI works: each request includes the
        full conversation history. The second FMAPI call replays the assistant
        message from the first turn, which may have empty content + tool_calls.
        """
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.items import ToolCallItem

        from databricks_openai.agents import McpServer

        set_default_openai_client(async_client)
        set_default_openai_api("chat_completions")

        async with McpServer.from_uc_function(
            catalog=_MCP_CATALOG,
            schema=_MCP_SCHEMA,
            function_name=_MCP_FUNCTION,
            workspace_client=workspace_client,
        ) as server:
            agent = Agent(
                name="echo-agent",
                instructions="Use the echo_message tool to echo messages when asked.",
                model=model,
                mcp_servers=[server],
            )

            # Turn 1: user sends first message
            first_result = await Runner.run(agent, "Echo the message 'hello'")
            assert first_result.final_output is not None
            assert "hello" in first_result.final_output

            # Verify first turn produced a tool call
            first_item_types = [type(item) for item in first_result.new_items]
            assert ToolCallItem in first_item_types

            # Turn 2: user sends followup with full conversation history
            # (mirrors how the chat UI accumulates messages)
            history = first_result.to_input_list()
            history.append({"role": "user", "content": "Now echo the message 'world'"})

            second_result = await Runner.run(agent, history)
            assert second_result.final_output is not None
            assert "world" in second_result.final_output

            # Second turn should also have called the tool
            second_item_types = [type(item) for item in second_result.new_items]
            assert ToolCallItem in second_item_types

            # The accumulated history should be longer than after the first turn
            second_history = second_result.to_input_list()
            assert len(second_history) > len(history), (
                f"Expected history to grow: {len(history)} -> {len(second_history)}"
            )

    async def test_streaming(self, async_client, workspace_client, model):
        """Streaming conversation: mirrors the app-template @stream() handler.

        Uses Runner.run_streamed() which is the streaming path in app-templates.
        Verifies that stream events arrive in the expected order and contain
        the expected item types.
        """
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.stream_events import RunItemStreamEvent

        from databricks_openai.agents import McpServer

        set_default_openai_client(async_client)
        set_default_openai_api("chat_completions")

        async with McpServer.from_uc_function(
            catalog=_MCP_CATALOG,
            schema=_MCP_SCHEMA,
            function_name=_MCP_FUNCTION,
            workspace_client=workspace_client,
        ) as server:
            agent = Agent(
                name="echo-agent",
                instructions="Use the echo_message tool to echo messages when asked.",
                model=model,
                mcp_servers=[server],
            )
            result = Runner.run_streamed(agent, input="Echo the message 'streaming test'")

            # Consume the stream and collect events
            # (mirrors process_agent_stream_events in app-templates)
            run_item_events = []
            event_count = 0
            async for event in result.stream_events():
                event_count += 1
                if isinstance(event, RunItemStreamEvent):
                    run_item_events.append(event)

            # Stream should have produced events
            assert event_count > 0, "No stream events received"

            # Should have tool_called and tool_output events (the agent called echo_message)
            event_names = [e.name for e in run_item_events]
            assert "tool_called" in event_names, f"Expected tool_called event, got: {event_names}"
            assert "tool_output" in event_names, f"Expected tool_output event, got: {event_names}"
            assert "message_output_created" in event_names, (
                f"Expected message_output_created event, got: {event_names}"
            )

            # After stream completes, final_output should be available
            assert result.final_output is not None
            assert "streaming test" in result.final_output
