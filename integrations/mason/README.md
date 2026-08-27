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

## Commands

```text
mason [-p <profile>] [-o text|json]
  login        [--profile P]
  logout
  add
    ui         [--enable-crash] [directory]
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
               [--with-session-store N] [--actor-id ID]
               [--with-traces C.S] [--create-stores]
  deployments  list | get | logs | start | stop | delete
```

## Add the demo UI

From a LangGraph scratch agent project, add the zero-build browser client:

```sh
cd ./my-agent
mason add ui
uv run start-server
```

The UI exercises streaming, sticky background polling, same-ID session resume, managed Memory Store
entries, managed Session Store transcript items, agent memory tools, human approval, and runtime status.
`mason add ui --enable-crash` also enables a demo-only endpoint that terminates the process so an
auto-restarting dev server or deployed Databricks App can prove that the durable checkpointer resumes
the same conversation after restart.

For the full deployed demo, connect both managed stores:

```sh
mason add ui --enable-crash
mason --profile <profile> deploy mason-agent-demo --source . \
  --with-session-store mason-demo-sessions \
  --with-memory-store mason-demo-memory \
  --actor-id alice \
  --create-stores
```

`mason deploy` provisions or resolves the stores and injects their names plus the shared actor id.
The UI can create, list, and search memory entries for the actor; it also creates a managed session
and mirrors user/assistant turns into Session Store items. It can pause on the sample approval-gated
tool, crash the app, wait for a new process, and approve the same paused run.
