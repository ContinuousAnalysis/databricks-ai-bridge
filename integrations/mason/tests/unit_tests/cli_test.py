"""Unit tests for the mason command tree: flattened sessions + login/logout wired in."""

from __future__ import annotations

import click

from databricks_mason import cli


def test_sessions_verbs_are_flat_no_redundant_subgroup():
    names = set(cli.sessions.commands)
    # Session verbs are direct subcommands of `sessions` (no `mason sessions sessions`).
    assert {"create", "list", "get", "update", "delete", "fork"} <= names
    assert "sessions" not in names
    # Sub-resources remain their own groups.
    assert {"stores", "items"} <= names


def test_root_registers_login_and_logout():
    names = set(cli.mason.commands)
    assert {"login", "logout", "memory", "sessions", "tracing", "deploy", "deployments"} <= names


def test_memory_search_uses_canonical_page_size_option():
    entries = cli.memory.commands["entries"]
    assert isinstance(entries, click.Group)
    search = entries.commands["search"]
    parameter_names = {parameter.name for parameter in search.params}

    assert "page_size" in parameter_names
    assert "limit" not in parameter_names


def test_session_delete_has_no_removed_force_option():
    delete = cli.sessions.commands["delete"]

    assert "force" not in {parameter.name for parameter in delete.params}
