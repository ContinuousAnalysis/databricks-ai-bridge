"""Adapt the framework-native LangGraph agent to Mason Runtime."""

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from langchain.messages import AIMessageChunk
from langgraph.types import Command

from agent.agent import create_agent_graph
from databricks_mason import DurableAgentContext
from databricks_mason.langgraph import checkpointer, start_trace, thread_config

logger = logging.getLogger(__name__)

_INVOCATION_METADATA_KEY = "databricks_mason.invocation_id"


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


def _invocation_input(payload: dict[str, Any]) -> Any:
    resume = payload.get("resume")
    if resume is not None:
        return Command(resume=resume)
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    return {"messages": messages}


async def invoke(value: Any, context: DurableAgentContext) -> dict:
    payload = _payload(value)
    return await _run_agent(_invocation_input(payload), payload, context)


async def recover(value: Any, context: DurableAgentContext) -> dict:
    payload = _payload(value)
    session_id = _session_id(payload, context)
    actor = _actor(payload, session_id)
    checkpoint = await checkpointer().aget_tuple(thread_config(session_id, actor))
    current_invocation_checkpointed = bool(
        checkpoint and checkpoint.metadata.get(_INVOCATION_METADATA_KEY) == context.invocation_id
    )
    agent_input = None if current_invocation_checkpointed else _invocation_input(payload)
    return await _run_agent(agent_input, payload, context)


async def _run_agent(
    agent_input: Any,
    payload: dict[str, Any],
    context: DurableAgentContext,
) -> dict:
    session_id = _session_id(payload, context)
    actor = _actor(payload, session_id)
    with start_trace(name="invoke", inputs=agent_input, session_id=session_id) as span:
        outputs = [
            event
            async for event in _runtime_events(
                agent_input,
                context,
                session_id=session_id,
                actor=actor,
                model=payload.get("model"),
            )
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
    agent_input: Any,
    context: DurableAgentContext,
    *,
    session_id: str,
    actor: str,
    model: Any,
) -> AsyncGenerator[dict, None]:
    agent = await create_agent_graph(actor, model if isinstance(model, str) else None)
    async for event in _serialize_events(
        agent.astream(
            input=agent_input,
            config={
                **thread_config(session_id, actor),
                "metadata": {_INVOCATION_METADATA_KEY: context.invocation_id},
            },
            stream_mode=["updates", "messages"],
            durability="sync",
        )
    ):
        await context.emit(event)
        yield event


async def _serialize_events(async_stream: AsyncIterator[Any]) -> AsyncGenerator[dict, None]:
    async for event in async_stream:
        mode, payload = event[0], event[1]
        if mode == "updates":
            if interrupts := payload.get("__interrupt__"):
                for item in interrupts:
                    yield {"type": "interrupt", "id": item.id, "value": item.value}
                continue
            for node_data in payload.values():
                messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                for message in messages:
                    yield {"type": "message", "message": message.model_dump()}
        elif mode == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield {"type": "delta", "content": content, "id": chunk.id}
            except Exception:
                logger.exception("Error processing agent stream chunk")
