# Mason OpenAI Agent

An OpenAI Agents SDK agent served by `databricks_mason.AgentApp`. Mason keeps invocation state and
events in memory during `mason dev`. On deployment, Mason attaches a Runtime Store for persisted
invocation state, events, and recovery.

For normal agent development, edit `agent/agent.py` and leave the generated files under `runtime/`
unchanged. The Runtime adapter translates Mason requests and events to the OpenAI Agents SDK; it
does not own the agent's model, instructions, or tools.

## Run locally

```bash
mason dev
```

The API is available at `http://localhost:8000/api/invocations`. Every request supplies a UUID `id`.
That ID is the invocation identifier and idempotency key. Agent-specific values live inside the
opaque `input` object:

```bash
SESSION_ID=$(uuidgen)
INVOCATION_ID=$(uuidgen)

curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"session_id\":\"$SESSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it? Use your tool.\"}]}}"
```

Reuse `SESSION_ID` for multi-turn conversation history. Generate a new `INVOCATION_ID` for each
turn. Retrying the same request with the same invocation ID returns the persisted result; changing
the request while reusing the ID returns `409`.

## Invocation modes

- Foreground: omit `background` and `stream`; the response contains the agent result under `output`.
- Foreground streaming: set `stream: true`; the response is SSE backed by persisted events.
- Background: set `background: true`; poll the returned `status_url`.
- Background streaming: set both flags; the `202` response includes `status_url` and `events_url`.

```bash
INVOCATION_ID=$(uuidgen)
curl -sN http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"session_id\":\"$SESSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}]},\"stream\":true}"

INVOCATION_ID=$(uuidgen)
curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"session_id\":\"$SESSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize durable agents.\"}]},\"background\":true}" | jq
curl -sS "http://localhost:8000/api/invocations/$INVOCATION_ID" | jq
```

SSE records contain the events translated by `runtime/adapter.py`: token `delta`s, completed `message`s,
and HITL `interrupt`s. Replay from a cursor with
`GET /api/invocations/{id}/events?after={sequence}`.

## Human approval

`send_message` requires approval. When output or the event stream contains an `interrupt`, submit a
new invocation with the same application session:

```json
{
  "id": "<new-uuid>",
  "input": {
    "session_id": "<same-session-id>",
    "resume": {"decisions": [{"type": "approve"}]}
  }
}
```

The paused Agents SDK `RunState` is process-local. A managed Session Store preserves transcript
history, but not a pending approval across restarts or replicas.

## Crash recovery

`runtime/main.py` registers the hooks from `runtime/adapter.py`. OpenAI Agents SDK does not currently
expose LangGraph-style node checkpoints, so recovery replays the persisted application input against
the same session. Invocation state and emitted events survive process loss in the deployed Runtime
Store, but tool calls and other external side effects remain at-least-once and must be idempotent.

## Chat app

The browser UI is included by default. It generates a stable application session ID in local
storage, places it inside each invocation's `input`, and generates a fresh invocation UUID per turn.
Use `mason init --framework openai --disable-chat-app` for API-only output.

## Configure and deploy

- Change the model, instructions, and tools in `agent/agent.py`.
- Leave `runtime/adapter.py` unchanged unless you intentionally need a custom wire protocol or
  recovery policy.
- Add local tools under `agent/tools/`; modules are auto-discovered.
- Add MCP servers in `agent/mcps.py` or with `mason tools add mcp`.
- Bind long-term memory with `mason memory bind <store>`.
- Bind durable transcript history with `mason sessions bind <store>`.

```bash
mason --profile <profile> deploy agent-openai --source .
```

Deployment provisions or reuses the app's dedicated Runtime Store database. Mason manages its
runtime schema and tables; developers do not configure a Runtime Store binding in `agent.toml`.

The `__Host-databricks-app-router` cookie may be supplied independently for sticky replica routing.
It is not authentication and is not used as the template's application session ID.
