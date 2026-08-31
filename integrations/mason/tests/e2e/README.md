# Mason df1 E2E matrices

These drivers produce machine-verifiable JSON evidence. They are real workspace tests, not mocked
integration tests. Keep the generated output outside the repository.

## Agent-skills matrix

`skill_matrix.py` proves CLI and direct-`agent.toml` authoring parity across local `mason dev` and
deployed Databricks Apps. Each lane exercises a project-local skill body, a project-local skill's
referenced file, and an irrelevant prompt that must activate no skill (12 semantic rows total).

The driver first proves the full repository/template shape, both auth profiles, model endpoint,
Apps control plane, dynamic local ports, and exact skill listing for both authoring paths. It creates
two uniquely named Apps. Every command and HTTP operation records a redacted request class/argv,
status or return code, timing, and decisive excerpt. Generated projects receive a unique temporary
runtime print plus activation prints; the driver proves those prints in local and deployed logs,
proves prompt construction caused zero body reads, and proves the unique instrumentation never
touched tracked or untracked repository files. Tokens and authorization headers are never placed in
evidence or captured logs.

Run from the Mason package after committing the exact harness/product revision to test. Before any
workspace mutation, the driver builds and hashes a fresh wheel from that repository:

```bash
cd integrations/mason
uv run python tests/e2e/skill_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --output /tmp/mason-agent-skills-df1 \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
uv run python tests/e2e/skill_matrix.py \
  --verify-evidence /tmp/mason-agent-skills-df1/evidence.json
```

Verifier-only mode reads local JSON and does not initialize a Databricks client, so it works without
workspace credentials. Success requires all 12 matrix cells, both authoring and runtime lanes,
body/file selection, irrelevant nonactivation, zero eager body loads, local/deployed freshness,
current commit and wheel hashes, successful deletion plus verified absence for both Apps, and a
clean redaction scan.

Resources are deleted only after every semantic row passes. App absence accepts only explicit
not-found results; auth, rate-limit, network, and 5xx failures remain cleanup failures. A semantic or
setup failure preserves resources for diagnosis and exits nonzero; `--keep-resources` also suppresses
cleanup and therefore cannot produce verifier-green evidence. Review `commands.log` and
`evidence.json`, then explicitly delete retained resources after diagnosing a failed run. The driver
emits terse timestamped `START`, `PROGRESS`, `PASS`, and `ERROR` lines throughout long operations for
a one-minute external monitor.

The evidence JSON is the canonical report input. Do not hand-edit it or hand-author HTML in this
directory; generate the HTML summary separately with the metric-report tooling after the verifier
passes.

Repeated runs may reuse the same `--output`: each run gets an isolated
`runs/mason-skills-<date>-<id>/` directory while `<output>/evidence.json` is the latest canonical
evidence. Local dev ports are dynamically selected and preflighted rather than fixed.

## Agent-tool matrix

This suite proves that CLI edits and direct `agent.toml` edits reach the same runtime code.
It creates two LangGraph projects (CLI/direct), runs each with `mason dev`, deploys each to
Databricks Apps, and semantically exercises sandbox, `system.ai.web_search`, a local Python tool,
and a temporary Unity Catalog function. The result is 16 evidence rows.

## Run

```bash
cd integrations/mason
uv build --wheel --out-dir /tmp/mason-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/mason-tooling-dist/databricks_mason-0.1.0.dev0-py3-none-any.whl \
  --output /tmp/mason-tool-matrix-df1 \
  --uc-schema aifx_benchmarks.mason_agent_tools_e2e \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.web_search`, and permission to create a schema/function. The suite discovers and starts
a SQL warehouse. Override its defaults with `--warehouse-id` or `--uc-schema catalog.schema`.
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
The template repo/ref flags make `mason init` read the exact checkout under test and avoid remote
clone throttling; provide both or omit both to test the default upstream template.

Direct authoring does not call `mason tools add`: it replaces `agent.toml` with
`fixtures/direct_agent.toml` and creates the user-owned Python tool file. CLI authoring invokes all
four `mason tools add ...` commands, then implements the generated Python stub. Every exact command
and generated-file step is captured in `commands.log`.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/mason-tool-matrix-df1/evidence.json
```

Success is exactly `16 passed, 0 failed, 0 skipped`. Temporary Apps and the UC function are deleted
after a successful run. Pass `--keep-resources` while debugging.
