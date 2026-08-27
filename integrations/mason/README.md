# `databricks-mason`

Mason is an experimental CLI for Databricks custom agent preview APIs and
deployments. It manages memory, sessions, tracing, and deployments from one
authenticated command.

> The underlying APIs are in preview and may need workspace enablement.

## Installation

From PyPI:

```sh
pip install databricks-mason
```

From source:

```sh
pip install 'git+https://github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/mason'
```

For tracing commands, install Mason with tracing extras:

```sh
pip install 'databricks-mason[tracing]'
```

## Authentication

Mason uses [Databricks authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
If you do not already have credentials, authenticate a named profile first. You can
then ask Mason to validate and remember that profile:

```sh
databricks auth login --profile <profile>
mason login --profile <profile>
mason sessions stores list
```

`mason login` does not create credentials; it stores the selected profile in
`~/.mason/config.json`. `mason logout` forgets that selection without revoking the
underlying credentials. If Databricks SDK default authentication is already configured,
you can skip `mason login`. You can also pass `--profile/-p` for an individual command.
Use `--output json` for scripting.

## Python SDK

Besides the CLI, Mason ships a typed, importable SDK for the same memory and session
APIs. It wraps the CLI's `AgentApiClient` (so it shares profile-based auth), returns typed
handles instead of raw dicts, auto-consumes pagination, and adds convenience lookups.

```python
from databricks_mason import DatabricksAgentClient

client = DatabricksAgentClient(profile="my-profile")  # or default SDK auth

# Memory: bound store handles, get-or-create, and read-modify-write append
store = client.memory_store.get(
    display_name="coding_agent_memory",
    create_if_not_exists=True,
    description="Long-term coding-agent memory",
)
store.add(
    actor_id="alice",
    session_id="project-sess-1",
    path="/memories/preferences.md",
    content="The user prefers concise answers.",
)
hits = store.search(actor_id="alice", query="response preferences")

# Sessions: bound stores/sessions and durable transcript items
sstore = client.session_store.create(session_store_name="support-agent-sessions")
session = sstore.create_session(actor_id="customer-123", session_id="case-456")
session.append(
    [
        {"type": "message", "role": "user", "content": "I need help with my cluster."},
        {"type": "message", "role": "assistant", "content": "Let's take a look."},
    ]
)
page = session.list_items(page_size=100, order_by="create_time asc")
```

`memory_store.list(...)`, `session_store.list()`, and `store.list_sessions()` consume all
server pages; `list_sessions()` defaults to `order_by="create_time desc"` for exactly-once
enumeration. `session.list_items()` returns one `SessionItemPage`; pass its `next_page_token`
for the next page. Every list/search `page_size` must be between 1 and 100. `session.fork(...)`
creates an independent copy, optionally through a specific item; deleting a session cascades to
its descendants.

## Commands

```text
mason [-p <profile>] [-o text|json]
  login        [--profile P]
  logout
  memory
    stores     create | list | get | update | delete
    entries    create | get | list | search | update | delete
  sessions     create | list | get | update | delete | fork
    stores     create | list | get | update | delete
    items      list | append | pop | clear
  tracing
    setup      --catalog C --schema S [--experiment E]
    list | get | instrument
  deploy       <name> --source PATH [--with-memory-store N]
               [--with-session-store N] [--with-traces C.S] [--create-stores]
  deployments  list | get | logs | start | stop | delete
```
