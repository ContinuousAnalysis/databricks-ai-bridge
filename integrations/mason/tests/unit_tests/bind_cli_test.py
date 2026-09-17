"""CLI tests for `mason memory bind/unbind` and `mason sessions bind/unbind`.

Backfills coverage for the local store-binding command handlers using a real temp AgentProject.
"""

from __future__ import annotations

import pathlib

from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.cli.memory import memory
from databricks_mason.cli.sessions import sessions
from databricks_mason.project_config import write_project_metadata


class _Ctx:
    def __init__(self, output="text"):
        self.output = output

    def client(self):
        raise AssertionError("Binding stores must not require workspace access.")


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "agent-langgraph"
    (project / "agent").mkdir(parents=True)
    write_project_metadata(project, framework="langgraph", template="agent-langgraph")
    AgentProject.create(project, framework="langgraph", server="mason").write()
    return project


def test_memory_bind_and_unbind(tmp_path):
    project = _project(tmp_path)
    r = CliRunner().invoke(memory, ["bind", "my-mem", "--source", str(project)], obj=_Ctx())
    assert r.exit_code == 0, r.output
    reloaded = AgentProject.load(project)
    assert reloaded.memory_store == "my-mem"

    r2 = CliRunner().invoke(memory, ["unbind", "--source", str(project)], obj=_Ctx())
    assert r2.exit_code == 0, r2.output
    assert AgentProject.load(project).memory_store is None


def test_memory_bind_rejects_removed_create_option_without_manifest_change(tmp_path):
    project = _project(tmp_path)
    before = (project / "agent.toml").read_bytes()
    result = CliRunner().invoke(
        memory, ["bind", "ghost", "--no-create-stores", "--source", str(project)], obj=_Ctx()
    )
    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--no-create-stores" in result.output
    assert (project / "agent.toml").read_bytes() == before


def test_sessions_bind_and_unbind(tmp_path):
    project = _project(tmp_path)
    r = CliRunner().invoke(sessions, ["bind", "my-sess", "--source", str(project)], obj=_Ctx())
    assert r.exit_code == 0, r.output
    assert AgentProject.load(project).session_store == "my-sess"

    r2 = CliRunner().invoke(sessions, ["unbind", "--source", str(project)], obj=_Ctx())
    assert r2.exit_code == 0, r2.output
    assert AgentProject.load(project).session_store is None


def test_unbind_when_nothing_bound_is_graceful(tmp_path):
    project = _project(tmp_path)
    r = CliRunner().invoke(memory, ["unbind", "--source", str(project)], obj=_Ctx())
    assert r.exit_code == 0
    assert "No memory store binding" in r.output or "Removed" in r.output
