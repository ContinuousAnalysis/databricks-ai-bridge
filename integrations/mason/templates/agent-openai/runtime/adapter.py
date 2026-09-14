"""Generated OpenAI-to-Mason glue; most applications should not edit this file."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from agents import Runner, RunResultStreaming, RunState
from agents.items import ToolApprovalItem
from agents.mcp import MCPServerManager
from openai.types.responses import ResponseTextDeltaEvent

from agent.agent import create_agent
from agent.mcps import build_mcp_servers
from databricks_mason import DurableAgentContext
from databricks_mason.openai import mcp_servers, session_store, start_trace

logger = logging.getLogger(__name__)

# OpenAI Sessions persist transcript history, not a paused RunState. Keep pending approvals local.
_pending_runs: dict[str, RunState] = {}


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"messages": value}
    if not isinstance(value, dict):
        raise ValueError("input must be a message list or an object")
    return value


def _session_id(payload: dict[str, Any], context: DurableAgentContext) -> str:
    value = payload.get("session_id") or context.session_id
    if not isinstance(value, str) or not value:
        raise ValueError("session_id must be a non-empty string")
    return value


def _actor(payload: dict[str, Any], session_id: str) -> str:
    value = payload.get("actor") or session_id
    if not isinstance(value, str) or not value:
        raise ValueError("actor must be a non-empty string")
    return value


async def invoke(value: Any, context: DurableAgentContext) -> dict:
    return await _run_agent(_payload(value), context)


async def recover(value: Any, context: DurableAgentContext) -> dict:
    return await _run_agent(_payload(value), context)


async def _run_agent(payload: dict[str, Any], context: DurableAgentContext) -> dict:
    session_id = _session_id(payload, context)
    actor = _actor(payload, session_id)
    with start_trace(name="invoke", inputs=payload, session_id=session_id) as span:
        outputs = [
            event
            async for event in _runtime_events(payload, context, session_id, actor)
            if event.get("type") in ("message", "interrupt")
        ]
        interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
        result = {
            "output": [
                event["message"] if event["type"] == "message" else event for event in outputs
            ],
            "session_id": session_id,
            "status": "interrupted" if interrupted else "completed",
        }
        if span is not None:
            span.set_outputs(result)
        return result


async def _runtime_events(
    payload: dict[str, Any],
    context: DurableAgentContext,
    session_id: str,
    actor: str,
) -> AsyncGenerator[dict, None]:
    async for event in _agent_events(payload, session_id, actor):
        await context.emit(event)
        yield event


async def _agent_events(
    payload: dict[str, Any], session_id: str, actor: str
) -> AsyncGenerator[dict, None]:
    servers = await mcp_servers(build_mcp_servers())
    async with MCPServerManager(servers) as manager:
        mcp = []
        for server in manager.active_servers:
            tool_filter = server.tool_filter
            try:
                server.tool_filter = None
                server.cache_tools_list = True
                async with asyncio.timeout(manager.connect_timeout_seconds):
                    await server.list_tools()
            except Exception:
                logger.warning(
                    "Failed to list tools from MCP server %r; continuing without it.",
                    server.name,
                    exc_info=True,
                )
            else:
                mcp.append(server)
            finally:
                server.tool_filter = tool_filter

        model = payload.get("model")
        agent = create_agent(actor, mcp, model=model if isinstance(model, str) else None)
        resume = payload.get("resume")
        if resume is not None:
            if not isinstance(resume, dict):
                raise ValueError("resume must be an object")
            run_input: Any = _apply_decisions(session_id, resume)
            result = Runner.run_streamed(agent, run_input)
        else:
            messages = payload.get("messages") or []
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            result = Runner.run_streamed(
                agent,
                messages,
                session=session_store(session_id, actor),
            )

        async for event in _serialize_events(result, session_id):
            yield event


def _apply_decisions(session_id: str, resume: dict) -> RunState:
    state = _pending_runs.pop(session_id, None)
    if state is None:
        raise RuntimeError(
            "No paused run for this session. HITL pauses are in-process only, so a restart or a "
            "different replica loses them; retry the turn."
        )
    decisions = resume.get("decisions") or []
    for decision, item in zip(decisions, state.get_interruptions(), strict=False):
        if decision.get("type") == "approve":
            state.approve(item)
        else:
            state.reject(item, rejection_message=decision.get("message"))
    return state


async def _serialize_events(
    result: RunResultStreaming, session_id: str
) -> AsyncGenerator[dict, None]:
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                yield {"type": "delta", "content": event.data.delta, "id": event.data.item_id}
        elif event.type == "run_item_stream_event":
            if message := _normalize_item(event.item):
                yield {"type": "message", "message": message}

    if result.interruptions:
        _pending_runs[session_id] = result.to_state()
        for item in result.interruptions:
            yield {"type": "interrupt", "id": item.call_id, "value": _approval_value(item)}


def _approval_value(item: ToolApprovalItem) -> dict:
    return {"action_requests": [{"name": item.tool_name, "args": _tool_args(item)}]}


def _tool_args(item: ToolApprovalItem) -> Any:
    import json

    args = item.arguments
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"arguments": args}
    return args or {}


def _normalize_item(item: Any) -> dict | None:
    from agents import ItemHelpers
    from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem

    if isinstance(item, MessageOutputItem):
        return {"role": "assistant", "content": ItemHelpers.text_message_output(item)}
    if isinstance(item, ToolCallItem):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": item.tool_name, "args": _tool_args_from_call(item)}],
        }
    if isinstance(item, ToolCallOutputItem):
        return {"role": "tool", "name": _tool_call_name(item), "content": str(item.output)}
    return None


def _tool_args_from_call(item: Any) -> Any:
    import json

    raw = item.raw_item
    args = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"arguments": args}
    return args or {}


def _tool_call_name(item: Any) -> str | None:
    raw = item.raw_item
    return raw.get("name") if isinstance(raw, dict) else getattr(raw, "name", None)
