"""Unit tests for manifest-backed ``mason tools`` commands."""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject, ToolSpec
from databricks_mason.cli.tools import tools
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata


class _Ctx:
    def __init__(self, output: str = "text", *, client=None):
        self.output = output
        self._client = client or _Client()

    def client(self):
        return self._client


class _Client:
    def __init__(self, service="system.ai.web_search", error=None):
        self.service = service
        self.error = error
        self.calls = []

    def get_mcp_service(self, service):
        self.calls.append(service)
        if self.error:
            raise self.error
        if service != self.service:
            raise AgentCliError("MCP service does not exist.", error_code="NOT_FOUND")
        return {"name": f"mcp-services/{service}"}


def _project(
    tmp_path: pathlib.Path,
    framework: str = "langgraph",
    *,
    template: str | None = None,
) -> pathlib.Path:
    project = tmp_path / f"agent-{framework}"
    (project / "agent" / "tools").mkdir(parents=True)
    (project / "tests" / "tools").mkdir(parents=True)
    (project / "agent" / "mcps.py").write_text("ORIGINAL = True\n", encoding="utf-8")
    write_project_metadata(project, framework=framework, template=template or f"agent-{framework}")
    server = "custom" if (template or "").startswith("custom-agent-") else "mason"
    AgentProject.create(project, framework=framework, server=server).write()
    return project


def test_add_sandbox_only_updates_manifest(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    loaded = AgentProject.load(project)
    assert loaded.tools[0].source.kind == "sandbox"
    assert loaded.tools[0].policy.downscope[0].resource == "table:samples.nyctaxi.trips"
    assert (project / "agent" / "mcps.py").read_text(encoding="utf-8") == "ORIGINAL = True\n"


def test_generic_mcp_rejects_sandbox_scope(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--scope",
            "table:samples.nyctaxi.trips",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--scope" in result.output
    assert AgentProject.load(project).tools == []


def test_add_mcp_and_uc_function_write_typed_manifest_records(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()

    mcp = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--name", "web", "--source", str(project)],
        obj=_Ctx(),
    )
    uc = runner.invoke(
        tools,
        [
            "add",
            "uc-function",
            "main.tools.lookup_ticket",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert mcp.exit_code == 0, mcp.output
    assert uc.exit_code == 0, uc.output
    loaded = AgentProject.load(project)
    assert [(tool.id, tool.source.kind) for tool in loaded.tools] == [
        ("web", "mcp"),
        ("lookup_ticket", "uc_function"),
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips"],
        ["add", "mcp", "system.ai.web_search"],
        ["add", "uc-function", "main.tools.lookup_ticket"],
    ],
)
def test_add_manifest_tool_works_for_any_framework(tmp_path: pathlib.Path, command: list[str]):
    # mcp / uc_function / sandbox are framework-neutral agent.toml entries; adding them must succeed
    # regardless of framework (every runtime adapter reads them from the manifest).
    project = _project(tmp_path, framework="openai")

    result = CliRunner().invoke(
        tools,
        [*command, "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).tools, "expected the tool to be written to the manifest"


@pytest.mark.parametrize(
    ("framework", "template"),
    [
        ("langgraph", "custom-agent-langgraph"),
        ("openai", "custom-agent-openai"),
    ],
)
@pytest.mark.parametrize(
    "command",
    [
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips"],
        ["add", "mcp", "system.ai.web_search"],
        ["add", "uc-function", "main.tools.lookup_ticket"],
    ],
)
def test_add_managed_tool_rejects_custom_server_template_without_manifest_change(
    tmp_path: pathlib.Path,
    framework: str,
    template: str,
    command: list[str],
):
    project = _project(tmp_path, framework, template=template)
    manifest = project / "agent.toml"
    before = manifest.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        tools,
        [*command, "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "require a Mason server template" in output
    assert "mason init --server mason" in output
    assert "agent/agent.py" in output
    assert manifest.read_text(encoding="utf-8") == before


def test_add_python_is_not_a_cli_command_and_does_not_mutate_project(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    manifest = project / "agent.toml"
    before = manifest.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        tools,
        ["add", "python", "lookup-ticket", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    # The facelift renders unknown commands in the diagnostic grammar (with a did-you-mean hint).
    assert "unknown command `python`" in result.output
    assert manifest.read_text(encoding="utf-8") == before
    assert list((project / "agent" / "tools").iterdir()) == []


def test_add_is_idempotent_and_json_reports_changed_files(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    args = ["add", "mcp", "system.ai.web_search", "--source", str(project)]
    runner = CliRunner()

    first = runner.invoke(tools, args, obj=_Ctx(output="json"))
    second = runner.invoke(tools, args, obj=_Ctx(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first_payload["changed"] is True
    assert first_payload["changed_files"] == [str(project / "agent.toml")]
    assert second_payload["changed"] is False
    assert second_payload["changed_files"] == []
    assert len(AgentProject.load(project).tools) == 1


def test_add_missing_mcp_leaves_project_unchanged(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}

    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.missing_service", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 1, result.output
    assert "NOT_FOUND" in result.output
    assert "system.ai.missing_service" in result.output
    assert "mason mcp list" in result.output
    assert {path: path.read_bytes() for path in project.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("error_code", ["PERMISSION_DENIED", "UNAVAILABLE", "UNAUTHENTICATED"])
def test_add_mcp_lookup_error_preserves_existing_manifest(tmp_path: pathlib.Path, error_code):
    project = _project(tmp_path)
    manifest = AgentProject.load(project)
    manifest.add_tool(ToolSpec.uc_function("existing", function="main.tools.existing"))
    manifest.write()
    before = manifest.path.read_bytes()
    client = _Client(error=AgentCliError("Lookup failed.", error_code=error_code))

    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(client=client),
    )

    assert result.exit_code == 1, result.output
    assert error_code in result.output
    assert "Lookup failed" in result.output
    assert manifest.path.read_bytes() == before


def test_add_mcp_looks_up_exact_service_in_custom_schema(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    client = _Client(service="main.tools.ticket_search")

    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "main.tools.ticket_search", "--name", "tickets", "--source", str(project)],
        obj=_Ctx(client=client),
    )

    assert result.exit_code == 0, result.output
    assert client.calls == ["main.tools.ticket_search"]
    assert AgentProject.load(project).tools == [
        ToolSpec.mcp("tickets", service="main.tools.ticket_search")
    ]


@pytest.mark.parametrize("service", ["web_search", "system.ai", "system..web_search"])
def test_add_malformed_mcp_rejects_before_lookup(tmp_path: pathlib.Path, service):
    project = _project(tmp_path)
    client = _Client()
    before = (project / "agent.toml").read_bytes()

    result = CliRunner().invoke(
        tools, ["add", "mcp", service, "--source", str(project)], obj=_Ctx(client=client)
    )

    assert result.exit_code == 1, result.output
    assert client.calls == []
    assert (project / "agent.toml").read_bytes() == before


def test_remove_tool_updates_only_the_manifest(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    manifest = AgentProject.load(project)
    manifest.add_tool(ToolSpec.mcp("broken", service="system.ai.missing_service"))
    manifest.write()

    result = runner.invoke(
        tools,
        ["remove", "broken", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert "Removed broken" in result.output
    assert AgentProject.load(project).tools == []
    assert (project / "agent" / "mcps.py").read_text(encoding="utf-8") == "ORIGINAL = True\n"


def test_remove_mcp_accepts_the_service_from_the_add_command(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--name",
            "web",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(
        tools,
        ["remove", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert "Removed web" in result.output
    assert AgentProject.load(project).tools == []


def test_remove_tool_is_idempotent_and_reports_json_changes(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output
    args = ["remove", "web_search", "--source", str(project)]

    first = runner.invoke(tools, args, obj=_Ctx(output="json"))
    second = runner.invoke(tools, args, obj=_Ctx(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == {
        "schema_version": 1,
        "changed": True,
        "changed_files": [str(project / "agent.toml")],
        "tool_id": "web_search",
    }
    assert json.loads(second.output) == {
        "schema_version": 1,
        "changed": False,
        "changed_files": [],
        "tool_id": "web_search",
    }


def test_tools_list_emits_manifest_records_as_json(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tools"] == [
        {
            "id": "web_search",
            "kind": "mcp",
            "source": "system.ai.web_search",
        }
    ]
