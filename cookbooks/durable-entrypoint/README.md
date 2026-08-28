# Durable Entrypoint

This cookbook shows the decorator-style durable app option with an OpenAI
Agents SDK loop.

```python
@app.entrypoint
async def agent(payload, context): ...

@app.on_resume
async def resume_agent(payload, context): ...
```

The app owns background execution, heartbeat recovery, final-result storage,
and cursor-based SSE replay. The agent owns its `AsyncDatabricksSession`, maps
JSON into the OpenAI SDK, and emits SDK events through `context.emit()`.

## Background, streaming, and client impact

| Capability | Developer contract | Client contract |
| --- | --- | --- |
| Background | Return the final JSON result. The app starts the task and stores its status and result. | With `background=true, stream=false`, `POST /runs` returns `202`; poll `GET /runs/{run_id}`. |
| Durable streaming | Convert SDK events to JSON and call `await context.emit(event)`. | With `stream=true`, the POST is SSE. Save each SSE `id`, reconnect through `GET /runs/{run_id}/events?after=<id>`, and poll the run for the authoritative result. |

`background=true, stream=true` starts one durable run and immediately opens its
event stream. Disconnecting the client does not cancel the run.

### OpenAI Agents SDK: before and after

The OpenAI Agents SDK is an in-process library; it does not define a remote
deployment client. Before this app, the developer implemented both the server
protocol and its client around `Runner.run_streamed()`:

```python
result = Runner.run_streamed(agent, input=message, session=session)
async for sdk_event in result.stream_events():
    await send_using_my_server_protocol(sdk_event)

async with http.stream("POST", "/my-agent", json={"message": "hello"}): ...
```

With the durable app, the agent loop stays the same, but the client adopts the
standard run envelope and runtime routes:

```python
async with http.stream(
    "POST",
    "/runs",
    json={
        "run_id": "run-1",
        "session_id": "conversation-1",
        "background": True,
        "stream": True,
        "payload": {"message": "hello"},
    },
) as response:
    async for line in response.aiter_lines():
        last_event_id = remember_sse_id(line, last_event_id)

final = (await http.get("/runs/run-1")).json()
```

### LangGraph SDK: before and after

A native LangGraph deployment already has a framework-specific remote client:

```python
thread = await langgraph.threads.create()
run = await langgraph.runs.create(
    thread["thread_id"], "agent", input={"messages": messages}
)
result = await langgraph.runs.join(thread["thread_id"], run["run_id"])

async for event in langgraph.runs.stream(
    thread["thread_id"], "agent", input={"messages": messages}
):
    consume(event)
```

Behind this generic app, `langgraph_sdk` is not wire-compatible. The client uses
the same `/runs` call shown above, normally passing `thread_id` as `session_id`;
the developer maps `payload` to graph input inside `@app.entrypoint`. A native
LangGraph client would require a separate protocol adapter.

References: [OpenAI Agents SDK streaming](https://openai.github.io/openai-agents-python/streaming/),
[LangGraph background runs](https://docs.langchain.com/langsmith/runs), and
[LangGraph resumable streaming](https://docs.langchain.com/langsmith/streaming).

## Durable HITL flow

HITL is modeled as two durable runs. The runtime does not keep a worker alive
while waiting for a person.

1. A background streamed proposal completes with `requires_action`.
2. The client reviews the persisted result.
3. The client submits approval as another background streamed run using the
   same `session_id`.

Start the proposal and watch its persisted event stream:

```bash
curl -N -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{
    "run_id": "proposal-1",
    "session_id": "approval-session-1",
    "background": true,
    "stream": true,
    "payload": {"action": "publish the release notes"}
  }'
```

Poll `GET /runs/proposal-1`. Its persisted result contains
`result.status=requires_action`. Then approve it:

```bash
curl -N -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{
    "run_id": "approval-1",
    "session_id": "approval-session-1",
    "background": true,
    "stream": true,
    "payload": {
      "action": "publish the release notes",
      "decision": "approve",
      "wait_seconds": 60
    }
  }'
```

Stop the process while the approved action is waiting. A new process reclaims
the stale run, calls `@app.on_resume` with the original payload and same
`session_id`, and appends events to the existing durable stream. Reconnect with:

```bash
curl -N 'localhost:8000/runs/approval-1/events?after=<last-event-id>'
```

Poll `GET /runs/approval-1` for the authoritative final result. External side
effects remain at-least-once and must be idempotent.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export LAKEBASE_AUTOSCALING_ENDPOINT=projects/.../endpoints/...
python agent.py
```
