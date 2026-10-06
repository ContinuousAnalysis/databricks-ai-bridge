"""`agentbricks login` / `logout` — remember an optional Databricks profile.

`login` validates a named profile and persists the selection; when credentials are missing or
rejected in an interactive terminal, it delegates setup to `databricks auth login` and retries.
Commands resolve their profile with `resolve_profile`: the `-p` flag, then the saved `agentbricks login`
selection, then `DATABRICKS_CONFIG_PROFILE`, then the project `.env` (project-aware commands only),
then the Databricks SDK's own default authentication. `logout` removes only Agent Bricks' saved
selection, not the underlying credentials. State lives in a small JSON file under `~/.agentbricks`
(override the directory with `AGENTBRICKS_CONFIG_HOME`, mainly for tests).
"""

from __future__ import annotations

import configparser
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Optional

import click

from databricks_agentbricks import render
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit._api_client import _AgentBricksApiClient


@dataclasses.dataclass(frozen=True)
class ResolvedProfile:
    """The profile a command runs with, and where it came from (for display and re-resolution)."""

    name: Optional[str]
    source: str


def _parse_env_file(path: pathlib.Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for a project `.env` (python-dotenv is not a CLI dependency)."""
    values: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # Match dotenv for unquoted values: a `#` that follows whitespace starts a comment.
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if key:
            values[key] = value
    return values


def resolve_profile(
    flag: Optional[str], project_dir: Optional[pathlib.Path] = None
) -> ResolvedProfile:
    """Pick the Databricks profile for a command, most to least specific.

    1. the `-p/--profile` flag,
    2. the profile saved by `agentbricks login`,
    3. the `DATABRICKS_CONFIG_PROFILE` environment variable,
    4. the project `.env`'s `DATABRICKS_CONFIG_PROFILE` (only when `project_dir` is given — the
       project-aware commands `dev` and `deploy` pass their source dir so the CLI and the locally
       running agent, which reads the same `.env`, agree on one profile),
    5. none — the Databricks SDK's own default authentication resolution.
    """
    if flag:
        return ResolvedProfile(flag, "--profile")
    saved = load_default_profile()
    if saved:
        return ResolvedProfile(saved, "agentbricks login")
    env_profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if env_profile:
        return ResolvedProfile(env_profile, "DATABRICKS_CONFIG_PROFILE")
    if project_dir is not None:
        file_profile = _parse_env_file(project_dir / ".env").get("DATABRICKS_CONFIG_PROFILE")
        if file_profile:
            return ResolvedProfile(file_profile, ".env")
    return ResolvedProfile(None, "Databricks SDK default")


def profile_host(profile: Optional[str]) -> Optional[str]:
    """Host configured for a profile in the Databricks config file, for display only.

    Honors `DATABRICKS_CONFIG_FILE` (default `~/.databrickscfg`) the same way the SDK client does.
    Display-only, so any problem — missing file, missing profile, missing host — yields None.
    """
    if not profile:
        return None
    # The SDK expands `~` in DATABRICKS_CONFIG_FILE; without it a tilde path reads nothing.
    config_path = pathlib.Path(
        os.getenv("DATABRICKS_CONFIG_FILE", str(pathlib.Path.home() / ".databrickscfg"))
    ).expanduser()
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path)
        return parser.get(profile, "host", fallback=None)
    except (OSError, configparser.Error):
        return None


def _config_file() -> pathlib.Path:
    base = os.environ.get("AGENTBRICKS_CONFIG_HOME")
    root = pathlib.Path(base) if base else pathlib.Path.home() / ".agentbricks"
    return root / "config.json"


def load_default_profile() -> Optional[str]:
    """The profile saved by `agentbricks login`, or None if the user never logged in."""
    try:
        return json.loads(_config_file().read_text()).get("profile")
    except (OSError, json.JSONDecodeError):
        return None


def _save_default_profile(profile: str) -> None:
    path = _config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile": profile}, indent=2) + "\n")


def _validate_profile(profile: str) -> tuple[_AgentBricksApiClient, str]:
    client = _AgentBricksApiClient(profile)
    return client, client.current_user


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _run_databricks_login(profile: str) -> None:
    command = ["databricks", "auth", "login", "--profile", profile]
    try:
        # Keep the child process interactive while preserving stdout for Agent Bricks JSON output.
        result = subprocess.run(command, text=True, check=False, stdout=sys.stderr)
    except FileNotFoundError as exc:
        raise AgentCliError(
            "Could not configure Databricks authentication: the `databricks` CLI was not found.",
            hint=f"Install the Databricks CLI, then retry `agentbricks login --profile {profile}`.",
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(
            f"`databricks auth login --profile {profile}` failed (exit {result.returncode})."
        )


def _authenticate_profile(profile: str) -> tuple[_AgentBricksApiClient, str]:
    # Local import: pulling databricks.sdk.errors loads the full SDK (~0.7s), which we defer off
    # the CLI startup path. This function already builds a client, so the cost lands here anyway.
    from databricks.sdk.errors import Unauthenticated

    try:
        return _validate_profile(profile)
    except (AgentCliError, Unauthenticated) as initial_error:
        if not _is_interactive():
            raise AgentCliError(
                f"Could not validate Databricks profile {profile!r}: {initial_error}",
                hint="Run this command in an interactive terminal so Agent Bricks can open "
                "Databricks login, or authenticate first with "
                f"`databricks auth login --profile {profile}`.",
            ) from initial_error
    except Exception as validation_error:  # noqa: BLE001 - normalize unexpected API failures
        raise AgentCliError(
            f"Could not validate Databricks profile {profile!r}: {validation_error}"
        ) from validation_error

    _run_databricks_login(profile)
    try:
        return _validate_profile(profile)
    except Exception as retry_error:  # noqa: BLE001 - normalize the post-login failure
        raise AgentCliError(
            f"Databricks login completed, but profile {profile!r} could not be validated: "
            f"{retry_error}"
        ) from retry_error


@click.command()
@click.option(
    "--profile",
    "-p",
    default=None,
    help="Profile to authenticate with and remember as the default.",
)
@click.pass_obj
def login(obj, profile) -> None:
    """Authenticate a profile and save it as the default, so later commands can omit -p."""
    profile = profile or obj.profile
    if not profile:
        raise AgentCliError(
            "No profile to save.",
            hint="Pass one to remember, e.g. `agentbricks login --profile <profile>`.",
        )
    client, user = _authenticate_profile(profile)
    _save_default_profile(profile)
    if obj.output == "json":
        render.emit_json({"profile": profile, "user": user, "host": client.host})
        return
    render.success(
        f"Logged in as {user}",
        fields={"Profile": profile, "Host": client.host},
        next_steps=[
            ("agentbricks init my-agent", "Scaffold a new agent project"),
        ],
    )


@click.command()
@click.pass_obj
def logout(obj) -> None:
    """Forget the saved profile selection without deleting its credentials."""
    path = _config_file()
    existed = path.exists()
    path.unlink(missing_ok=True)
    if obj.output == "json":
        render.emit_json({"logged_out": existed})
        return
    render.success("Logged out" if existed else "No saved login to clear")
