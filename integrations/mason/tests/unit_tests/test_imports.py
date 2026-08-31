import importlib.resources

from click.testing import CliRunner


def test_package_import() -> None:
    import databricks_mason

    assert databricks_mason.__doc__


def test_public_surface() -> None:
    import databricks_mason

    for name in ("MasonClient", "AgentCliError", "memory_store_path", "memory_entry_path"):
        assert name in databricks_mason.__all__
        assert hasattr(databricks_mason, name)


def test_agentapiclient_alias() -> None:
    from databricks_mason import MasonClient
    from databricks_mason.client import AgentApiClient

    assert AgentApiClient is MasonClient


def test_root_help_lists_skills_command() -> None:
    from databricks_mason.cli import mason

    result = CliRunner().invoke(mason, ["--help"])

    assert result.exit_code == 0
    assert "skills       Inspect and attach project-local agent skills." in result.output


def test_skill_runtime_templates_are_packaged() -> None:
    templates = importlib.resources.files("databricks_mason.templates")

    assert templates.joinpath("skill_manifest_runtime.py").is_file()
    assert templates.joinpath("skill_runtime_langgraph.py").is_file()
