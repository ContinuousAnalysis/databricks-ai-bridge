"""Unit tests for `agentbricks login` / `logout` and the saved-profile helpers.

`AGENTBRICKS_CONFIG_HOME` redirects the config file into a tmp dir, and `_AgentBricksApiClient`
is stubbed so login never touches the network.
"""

from __future__ import annotations

import json
import sys
from unittest import mock

from click.testing import CliRunner
from databricks.sdk.errors import Unauthenticated

from databricks_agentbricks.cli import auth


class _Ctx:
    """Stand-in for CliContext: auth commands read only .profile and .output."""

    def __init__(self, profile=None, output="text"):
        self.profile = profile
        self.output = output


def _stub_client(monkeypatch, user="me@example.com", host="https://ws"):
    fake = mock.Mock()
    fake.current_user = user
    fake.host = host
    monkeypatch.setattr(auth, "_AgentBricksApiClient", lambda profile: fake)


def test_load_default_profile_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    assert auth.load_default_profile() is None


def test_login_persists_and_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    _stub_client(monkeypatch)
    databricks_login = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)
    result = CliRunner().invoke(auth.login, ["--profile", "my-workspace"], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() == "my-workspace"
    databricks_login.assert_not_called()


def test_login_json_output(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    _stub_client(monkeypatch)
    result = CliRunner().invoke(auth.login, ["-p", "prof"], obj=_Ctx(output="json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "profile": "prof",
        "user": "me@example.com",
        "host": "https://ws",
    }


def test_login_falls_back_to_global_profile(tmp_path, monkeypatch):
    # When the command's own --profile is omitted, the global -p (obj.profile) is saved.
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    _stub_client(monkeypatch)
    result = CliRunner().invoke(auth.login, [], obj=_Ctx(profile="from-global"))
    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() == "from-global"


def test_login_without_any_profile_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    result = CliRunner().invoke(auth.login, [], obj=_Ctx(profile=None))
    assert result.exit_code != 0
    assert auth.load_default_profile() is None


def test_login_configures_invalid_profile_then_revalidates(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    validated = mock.Mock(current_user="me@example.com", host="https://ws")
    agentbricks_client = mock.Mock(side_effect=[auth.AgentCliError("no credentials"), validated])
    monkeypatch.setattr(auth, "_AgentBricksApiClient", agentbricks_client)
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    databricks_login = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    result = CliRunner().invoke(auth.login, ["--profile", "prof"], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() == "prof"
    assert agentbricks_client.call_args_list == [mock.call("prof"), mock.call("prof")]
    databricks_login.assert_called_once_with(
        ["databricks", "auth", "login", "--profile", "prof"],
        text=True,
        check=False,
        stdout=mock.ANY,
    )


def test_databricks_login_routes_child_stdout_to_stderr(monkeypatch):
    databricks_login = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    auth._run_databricks_login("prof")

    databricks_login.assert_called_once_with(
        ["databricks", "auth", "login", "--profile", "prof"],
        text=True,
        check=False,
        stdout=sys.stderr,
    )


def test_login_reauthenticates_unauthenticated_api_response(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    validated = mock.Mock(current_user="me@example.com", host="https://ws")
    validate_profile = mock.Mock(
        side_effect=[Unauthenticated("expired credentials"), (validated, "me@example.com")]
    )
    monkeypatch.setattr(auth, "_validate_profile", validate_profile)
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    databricks_login = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    result = CliRunner().invoke(auth.login, ["--profile", "prof"], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert validate_profile.call_count == 2
    databricks_login.assert_called_once()


def test_login_does_not_reauthenticate_non_auth_validation_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        auth, "_validate_profile", mock.Mock(side_effect=RuntimeError("service unavailable"))
    )
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    databricks_login = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    result = CliRunner().invoke(auth.login, ["--profile", "prof"], obj=_Ctx())

    assert result.exit_code != 0
    assert "service unavailable" in result.output
    assert auth.load_default_profile() is None
    databricks_login.assert_not_called()


def test_login_does_not_launch_browser_when_noninteractive(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        auth, "_AgentBricksApiClient", mock.Mock(side_effect=auth.AgentCliError("no credentials"))
    )
    monkeypatch.setattr(auth, "_is_interactive", lambda: False)
    databricks_login = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    result = CliRunner().invoke(auth.login, ["--profile", "prof"], obj=_Ctx())

    assert result.exit_code != 0
    assert "interactive terminal" in result.output
    assert auth.load_default_profile() is None
    databricks_login.assert_not_called()


def test_login_reports_when_databricks_cli_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        auth, "_AgentBricksApiClient", mock.Mock(side_effect=auth.AgentCliError("no credentials"))
    )
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    monkeypatch.setattr(auth.subprocess, "run", mock.Mock(side_effect=FileNotFoundError))

    result = CliRunner().invoke(auth.login, ["--profile", "prof"], obj=_Ctx())

    assert result.exit_code != 0
    assert "Could not configure Databricks authentication" in result.output
    assert "not found" in result.output
    assert auth.load_default_profile() is None


def test_logout_clears_saved_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"profile": "x"}))
    result = CliRunner().invoke(auth.logout, [], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() is None


# --- profile resolution (see resolve_profile) -------------------------------------


def _hermetic_resolution(monkeypatch, tmp_path):
    """Keep resolve_profile off this machine's env / saved login; layers are added by each test."""
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.setenv("AGENTBRICKS_CONFIG_HOME", str(tmp_path / "agentbricks-home"))


def test_resolve_profile_precedence_matrix(tmp_path, monkeypatch):
    _hermetic_resolution(monkeypatch, tmp_path)

    # Nothing anywhere: the Databricks SDK's own default chain.
    assert auth.resolve_profile(None) == auth.ResolvedProfile(None, "Databricks SDK default")

    home = tmp_path / "agentbricks-home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"profile": "saved"}))
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=from-dotenv\n")

    # The saved login outranks the env var and the project .env; an empty env var counts as unset.
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "")
    assert auth.resolve_profile(None, project) == auth.ResolvedProfile("saved", "agentbricks login")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env")
    assert auth.resolve_profile(None, project) == auth.ResolvedProfile("saved", "agentbricks login")

    # Without a saved login: the env var beats the .env.
    (home / "config.json").unlink()
    assert auth.resolve_profile(None, project) == auth.ResolvedProfile(
        "from-env", "DATABRICKS_CONFIG_PROFILE"
    )

    # The .env applies only below the env var, and only when its dir is given (project-aware commands).
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE")
    assert auth.resolve_profile(None, project) == auth.ResolvedProfile("from-dotenv", ".env")
    assert auth.resolve_profile(None) == auth.ResolvedProfile(None, "Databricks SDK default")

    # The -p flag beats everything.
    (home / "config.json").write_text(json.dumps({"profile": "saved"}))
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env")
    assert auth.resolve_profile("flag", project) == auth.ResolvedProfile("flag", "--profile")


def test_parse_env_file_edge_cases(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "export EXPORTED=exported-value\n"
        'DQ="double quoted"\n'
        "SQ='single quoted'\n"
        "SPACED =  padded  \n"
        "INLINE=prod  # team shared\n"
        'QUOTED_HASH="a # b"\n'
        "NO_SPACE_HASH=prod#literal\n"
        "a line without an equals sign\n"
    )
    assert auth._parse_env_file(env) == {
        "PLAIN": "value",
        "EXPORTED": "exported-value",
        "DQ": "double quoted",
        "SQ": "single quoted",
        "SPACED": "padded",
        # dotenv strips an unquoted inline comment (a `#` after whitespace), keeps a quoted `#`
        # literal, and leaves a `#` with no preceding whitespace alone.
        "INLINE": "prod",
        "QUOTED_HASH": "a # b",
        "NO_SPACE_HASH": "prod#literal",
    }
    assert auth._parse_env_file(tmp_path / "absent.env") == {}


def test_profile_host_honors_databricks_config_file(tmp_path, monkeypatch):
    config = tmp_path / "databrickscfg"
    config.write_text(
        "[primary]\nhost = https://primary.databricks.com\n\n"
        "[other]\nhost = https://other.databricks.com\n"
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(config))

    assert auth.profile_host("other") == "https://other.databricks.com"
    assert auth.profile_host("no-such-profile") is None
    assert auth.profile_host(None) is None

    # Display-only: a missing file or a broken config yields None, never an error.
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(tmp_path / "absent"))
    assert auth.profile_host("primary") is None
    broken = tmp_path / "broken"
    broken.write_text("= = =\nnot ini at all\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(broken))
    assert auth.profile_host("primary") is None

    # A tilde path is expanded like the SDK does (HOME redirected to tmp).
    monkeypatch.setenv("HOME", str(tmp_path))
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "databrickscfg").write_text("[primary]\nhost = https://primary.databricks.com\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "~/configs/databrickscfg")
    assert auth.profile_host("primary") == "https://primary.databricks.com"
