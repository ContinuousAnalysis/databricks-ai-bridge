"""Contract tests for the agent-skills df1 evidence verifier and recorder helpers."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import pytest
import tomli

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))

import skill_matrix as skill_matrix_module  # noqa: E402  # ty: ignore[unresolved-import]
from skill_matrix import (  # noqa: E402  # ty: ignore[unresolved-import]
    Runner,
    _activation_events,
    _app_health_class,
    _assistant_response_excerpt,
    _classify_cli_absence,
    _cli_authoring_argvs,
    _direct_skill_toml,
    _find_marker_paths,
    _instrument_runtime_text,
    _redact_output,
    _reserve_port,
    _run_process,
    _terminate_process_group,
    _wheel_build_argv,
    expected_matrix_cells,
    redact_argv,
    redact_text,
    verify_evidence,
)


def _activations(prompt_kind: str) -> list[dict[str, str]]:
    if prompt_kind == "body":
        return [{"op": "load", "skill_id": "body-guidance"}]
    if prompt_kind == "file":
        return [
            {"op": "load", "skill_id": "file-guidance"},
            {"op": "read", "skill_id": "file-guidance", "path": "facts.txt"},
        ]
    return []


def valid_evidence() -> dict:
    artifact_root = Path(tempfile.mkdtemp(prefix="mason-evidence-fixture-"))
    wheel = artifact_root / "mason.whl"
    log = artifact_root / "run.log"
    wheel.write_bytes(b"fixture wheel")
    log.write_text("fixture log\n", encoding="utf-8")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    log_hash = hashlib.sha256(log.read_bytes()).hexdigest()
    rows = []
    for authoring, runtime, prompt_kind in sorted(expected_matrix_cells()):
        rows.append(
            {
                "authoring": authoring,
                "runtime": runtime,
                "prompt_kind": prompt_kind,
                "source": {
                    "body": "local",
                    "file": "local",
                    "irrelevant": "none",
                }[prompt_kind],
                "status": "pass",
                "expected_marker": f"EXPECTED_{prompt_kind.upper()}",
                "actual_excerpt": f"EXPECTED_{prompt_kind.upper()}",
                "activations": _activations(prompt_kind),
                "freshness_marker": f"MASON_E2E_FRESH_{authoring}_{runtime}",
                "freshness_observed": True,
                "duration_seconds": 1.25,
                "app_name": f"mason-sk-{authoring}-abcdef12" if runtime == "deploy" else None,
                "app_url_class": "databricks_app" if runtime == "deploy" else None,
                "app_service_principal": "application-service-principal"
                if runtime == "deploy"
                else None,
                "artifact_paths": [str(log)],
                "error": None,
            }
        )
    checks = [
        "capabilities_probed_before_mutation",
        "preflight_repository_clean",
        "preflight_template_assembly",
        "preflight_dynamic_ports",
        "preflight_auth_profiles",
        "preflight_model_available",
        "preflight_apps_available",
        "custom_skill_listing",
        "prompt_construction_metadata_only",
        "freshness_local_and_deployed",
        "repository_instrumentation_absent",
        "git_commit_current",
        "wheel_hash_current",
        "all_steps_redacted",
    ]
    return {
        "schema_version": 1,
        "run_status": "passed",
        "failures": [],
        "run_id": "mason-skills-20260831-abcdef12",
        "generated_at": "2026-08-31T01:02:03+00:00",
        "environment": {
            "profile": "df1",
            "app_auth_profile": "df1-oauth-mcp",
            "host_class": "databricks_workspace",
            "python": "3.12.0",
            "platform": "linux",
        },
        "source": {
            "git_commit": "a" * 40,
            "template_ref_commit": "a" * 40,
            "git_status_clean_before": True,
            "git_status_clean_after": True,
            "wheel": {
                "path": str(wheel),
                "sha256": wheel_hash,
            },
        },
        "resources": {"apps": ["mason-sk-cli-abcdef12", "mason-sk-direct-abcdef12"]},
        "commands": [
            {
                "kind": "command",
                "label": "mason-skills-list",
                "argv": ["mason", "skills", "list", "--schema", "catalog.schema"],
                "cwd": "/tmp/project",
                "started_at": "2026-08-31T01:02:03+00:00",
                "duration_seconds": 0.5,
                "return_code": 0,
                "status": "pass",
                "stdout_excerpt": "catalog.schema.mason-skill-abcdef12",
                "stderr_excerpt": "",
            },
            {
                "kind": "http",
                "label": "invoke-cli-dev-body",
                "request_class": "agent_invocation",
                "method": "POST",
                "path_class": "/invocations",
                "started_at": "2026-08-31T01:02:04+00:00",
                "duration_seconds": 1.0,
                "http_status": 200,
                "status": "pass",
                "response_excerpt": "EXPECTED_BODY",
            },
        ],
        "validation_checks": [
            {"name": name, "status": "pass", "evidence": "deterministic proof"} for name in checks
        ],
        "rows": rows,
        "cleanup": {
            "semantic_passed": True,
            "attempted": True,
            "apps": [
                {
                    "name": "mason-sk-cli-abcdef12",
                    "delete_status": "pass",
                    "absence_verified": True,
                },
                {
                    "name": "mason-sk-direct-abcdef12",
                    "delete_status": "pass",
                    "absence_verified": True,
                },
            ],
        },
        "metrics": {
            "eager_body_loads": 0,
            "passed_rows": 12,
            "failed_rows": 0,
            "total_rows": 12,
        },
        "artifacts": [
            {
                "kind": "wheel",
                "path": str(wheel),
                "sha256": wheel_hash,
            },
            {
                "kind": "log",
                "path": str(log),
                "sha256": log_hash,
            },
        ],
        "redactions": {
            "scan_status": "pass",
            "findings": [],
            "credentials_recorded": False,
        },
    }


def _write_evidence(tmp_path: Path, evidence: dict) -> Path:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_verify_accepts_complete_cli_and_direct_matrix(tmp_path: Path) -> None:
    assert verify_evidence(_write_evidence(tmp_path, valid_evidence())) == 0


def test_verify_requires_progressive_loading_and_cleanup(tmp_path: Path) -> None:
    evidence = valid_evidence()
    evidence["metrics"]["eager_body_loads"] = 1
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    evidence["cleanup"]["apps"][0]["absence_verified"] = False
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    evidence["resources"]["apps"] = evidence["resources"]["apps"][:1]
    evidence["cleanup"]["apps"] = evidence["cleanup"]["apps"][:1]
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1


def test_verify_rejects_missing_or_duplicate_matrix_cells(tmp_path: Path) -> None:
    evidence = valid_evidence()
    evidence["rows"].pop()
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    evidence["rows"].append(copy.deepcopy(evidence["rows"][0]))
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1


def test_verify_rejects_wrong_activation_selection(tmp_path: Path) -> None:
    evidence = valid_evidence()
    irrelevant = next(row for row in evidence["rows"] if row["prompt_kind"] == "irrelevant")
    irrelevant["activations"] = [{"op": "load", "skill_id": "body-guidance"}]
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    file_row = next(row for row in evidence["rows"] if row["prompt_kind"] == "file")
    file_row["activations"] = [{"op": "load", "skill_id": "file-guidance"}]
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1


def test_verify_requires_current_hash_and_validation_proofs(tmp_path: Path) -> None:
    evidence = valid_evidence()
    evidence["source"]["git_commit"] = "stale"
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    evidence["validation_checks"] = evidence["validation_checks"][:-1]
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    evidence["artifacts"][0]["sha256"] = "c" * 64
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    missing = tmp_path / "missing.whl"
    evidence["source"]["wheel"] = {"path": str(missing), "sha256": "b" * 64}
    evidence["artifacts"][0] = {
        "kind": "wheel",
        "path": str(missing),
        "sha256": "b" * 64,
    }
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1


def test_verify_rehashes_accessible_wheel_artifact(tmp_path: Path) -> None:
    wheel = tmp_path / "fresh.whl"
    wheel.write_bytes(b"fresh wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    evidence = valid_evidence()
    evidence["source"]["wheel"] = {"path": str(wheel), "sha256": digest}
    evidence["artifacts"][0] = {"kind": "wheel", "path": str(wheel), "sha256": digest}
    path = _write_evidence(tmp_path, evidence)
    assert verify_evidence(path) == 0

    wheel.write_bytes(b"stale wheel")
    assert verify_evidence(path) == 1


def test_verify_independently_detects_unredacted_credentials(tmp_path: Path, monkeypatch) -> None:
    evidence = valid_evidence()
    evidence["commands"][0]["stderr_excerpt"] = "Authorization: Bearer super-secret-token"
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    evidence = valid_evidence()
    evidence["commands"][0]["stdout_excerpt"] = json.dumps(
        {"access_token": "plain-access", "refresh_token": "plain-refresh"}
    )
    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1

    monkeypatch.setattr(skill_matrix_module, "_SENSITIVE_KEY", re.compile(r"$^"))
    for leaked in (
        "client_secret=hunter2",
        "token=plain-value",
        "password: open-sesame",
        '"api_key": "plain-api-key"',
    ):
        evidence = valid_evidence()
        evidence["redactions"] = {
            "scan_status": "pass",
            "findings": [],
            "credentials_recorded": False,
        }
        evidence["commands"][0]["stderr_excerpt"] = leaked
        assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1


@pytest.mark.parametrize(
    "leaked",
    [
        "secret=hunter2",
        "secret: hunter2",
        '{"outer":{"secret":"hunter2"}}',
    ],
)
def test_verify_independently_rejects_bare_secret_contexts(tmp_path: Path, leaked: str) -> None:
    evidence = valid_evidence()
    evidence["redactions"] = {
        "scan_status": "pass",
        "findings": [],
        "credentials_recorded": False,
    }
    evidence["commands"][0]["stderr_excerpt"] = leaked

    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 1


def test_verify_does_not_treat_secret_prose_as_an_assignment(tmp_path: Path) -> None:
    evidence = valid_evidence()
    evidence["commands"][0]["stderr_excerpt"] = "the secret hunter2 is diagnostic prose"

    assert verify_evidence(_write_evidence(tmp_path, evidence)) == 0


@pytest.mark.parametrize(
    "leaked",
    [
        "secret=hunter2",
        "secret: hunter2",
        '{"outer":{"secret":"hunter2"}}',
    ],
)
def test_producer_redacts_bare_secret_contexts(leaked: str) -> None:
    assert "hunter2" not in _redact_output(leaked)


def test_producer_does_not_redact_secret_prose_or_similar_keys() -> None:
    prose = "the secret hunter2 is diagnostic prose"
    assert redact_text(prose) == prose
    assert json.loads(_redact_output('{"secretary":"hunter2"}')) == {"secretary": "hunter2"}


def test_redaction_helpers_remove_credentials_without_hiding_decisive_output() -> None:
    text = redact_text(
        "ok marker=MASON_LOCAL_OK Authorization: Bearer abc.def client_secret=hunter2 "
        "token=plain-value access_token: access-value refresh_token = refresh-value "
        'password: "open-sesame" api_key=plain-api dapi0123456789abcdef'
    )
    assert "MASON_LOCAL_OK" in text
    assert "abc.def" not in text
    assert "hunter2" not in text
    assert "plain-value" not in text
    assert "access-value" not in text
    assert "refresh-value" not in text
    assert "open-sesame" not in text
    assert "plain-api" not in text
    assert "dapi0123456789abcdef" not in text
    assert text.count("<redacted>") >= 3

    argv = redact_argv(["tool", "--token", "abc", "--profile", "df1", "secret=dapi12345678"])
    assert argv == ["tool", "--token", "<redacted>", "--profile", "df1", "<redacted>"]


def test_redaction_structurally_preserves_json_and_removes_adversarial_secrets() -> None:
    raw = json.dumps(
        {
            "access_token": "plain-access",
            "nested": {
                "refreshToken": "plain-refresh",
                "client_secret": "plain-client",
                "authorization": "Bearer plain-bearer",
            },
            "result": "MASON_OK",
        }
    )

    safe = _redact_output(raw)
    parsed = json.loads(safe)

    assert parsed["result"] == "MASON_OK"
    assert parsed["access_token"] == "<redacted>"
    assert set(parsed["nested"].values()) == {"<redacted>"}
    assert not any(
        value in safe for value in ("plain-access", "plain-refresh", "plain-client", "plain-bearer")
    )


def test_absence_classification_accepts_only_specific_not_found() -> None:
    assert _classify_cli_absence(1, "", "Error: RESOURCE_DOES_NOT_EXIST: app is absent")
    assert _classify_cli_absence(1, "", '{"error_code":"RESOURCE_DOES_NOT_EXIST","message":"gone"}')
    assert _classify_cli_absence(
        1, "", "Error: App with name mason-sk-cli-abcdef12 does not exist or is deleted.\n"
    )
    assert not _classify_cli_absence(
        1, "", "Error: App with name mason-sk-cli-abcdef12 is unavailable.\n"
    )
    assert not _classify_cli_absence(1, "", "Error: app not found")
    assert not _classify_cli_absence(1, "", "Error: proxy target not found")
    assert not _classify_cli_absence(1, "", "Error: DNS host not found")
    assert not _classify_cli_absence(1, "", "Error: upstream returned Not Found")
    assert not _classify_cli_absence(1, "", "Error: 429 rate limit")
    assert not _classify_cli_absence(1, "", "Error: authentication failed")
    assert not _classify_cli_absence(1, "", "Error: connection reset")
    assert not _classify_cli_absence(0, "{}", "")


def test_deployed_health_class_distinguishes_transient_and_terminal() -> None:
    assert _app_health_class(200) == "ready"
    assert _app_health_class(404) == "transient"
    assert _app_health_class(429) == "transient"
    assert _app_health_class(503) == "transient"
    assert _app_health_class(401) == "terminal"
    assert _app_health_class(403) == "terminal"
    assert _app_health_class(422) == "terminal"


def test_project_assembly_helpers_cover_cli_direct_freshness_and_activation(tmp_path: Path) -> None:
    cli = _cli_authoring_argvs(Path("/venv/mason"), tmp_path)
    assert cli == [
        [
            "/venv/mason",
            "skills",
            "add",
            "custom",
            ".claude/skills/body-guidance",
            "--source",
            str(tmp_path),
        ],
        [
            "/venv/mason",
            "skills",
            "add",
            "custom",
            ".claude/skills/file-guidance",
            "--source",
            str(tmp_path),
        ],
    ]
    direct = _direct_skill_toml()
    assert tomli.loads(direct) == {
        "skills": [
            {
                "id": "body-guidance",
                "source": {"kind": "local", "path": ".claude/skills/body-guidance"},
            },
            {
                "id": "file-guidance",
                "source": {"kind": "local", "path": ".claude/skills/file-guidance"},
            },
        ]
    }

    original = (
        "from __future__ import annotations\n"
        "    async def load_skill(skill_id: str) -> str:\n"
        "        return await provider_for(skill_id).load()\n"
        "    async def read_skill_file(skill_id: str, path: str) -> str:\n"
        "        return await provider_for(skill_id).read_file(path)\n"
        "    return context, _tools(immutable)\n"
    )
    instrumented = _instrument_runtime_text(original, "FRESH", "MASON_E2E_ACTIVATION_token_cli")
    assert instrumented == (
        "from __future__ import annotations\n"
        '\nprint("FRESH module_loaded", flush=True)\n'
        "    async def load_skill(skill_id: str) -> str:\n"
        '        print(f"MASON_E2E_ACTIVATION_token_cli op=load skill_id={skill_id}", flush=True)\n'
        "        return await provider_for(skill_id).load()\n"
        "    async def read_skill_file(skill_id: str, path: str) -> str:\n"
        '        print(f"MASON_E2E_ACTIVATION_token_cli op=read skill_id={skill_id} path={path}", flush=True)\n'
        "        return await provider_for(skill_id).read_file(path)\n"
        '    print("FRESH context_ready", flush=True)\n'
        "    return context, _tools(immutable)\n"
    )

    events = _activation_events(
        "MASON_E2E_ACTIVATION_token_cli op=load skill_id=body-guidance\n"
        "MASON_E2E_ACTIVATION_token_cli op=read skill_id=file-guidance path=facts.txt\n",
        "MASON_E2E_ACTIVATION_token_cli",
    )
    assert events == [
        {"op": "load", "skill_id": "body-guidance"},
        {"op": "read", "skill_id": "file-guidance", "path": "facts.txt"},
    ]
    assert expected_matrix_cells() == {
        ("cli", "dev", "body"),
        ("cli", "dev", "file"),
        ("cli", "dev", "irrelevant"),
        ("cli", "deploy", "body"),
        ("cli", "deploy", "file"),
        ("cli", "deploy", "irrelevant"),
        ("direct", "dev", "body"),
        ("direct", "dev", "file"),
        ("direct", "dev", "irrelevant"),
        ("direct", "deploy", "body"),
        ("direct", "deploy", "file"),
        ("direct", "deploy", "irrelevant"),
    }


def test_default_heartbeat_interval_has_scheduling_margin() -> None:
    assert skill_matrix_module.HEARTBEAT_SECONDS < 60
    assert inspect.signature(_run_process).parameters["heartbeat_seconds"].default < 60


def test_repository_marker_scan_includes_untracked_and_ignores_sdd_scratch(tmp_path: Path) -> None:
    (tmp_path / "untracked.py").write_text("UNIQUE_MARKER", encoding="utf-8")
    scratch = tmp_path / ".superpowers" / "sdd" / "plan"
    scratch.mkdir(parents=True)
    (scratch / "notes.md").write_text("UNIQUE_MARKER", encoding="utf-8")

    assert _find_marker_paths(tmp_path, "UNIQUE_MARKER") == [tmp_path / "untracked.py"]


def _runner_args(tmp_path: Path) -> Namespace:
    return Namespace(
        profile="df1",
        app_auth_profile="df1-oauth-mcp",
        output=tmp_path / "output",
        template_repo=str(tmp_path),
        template_ref="branch",
        keep_resources=False,
        verify_evidence=None,
    )


def test_fresh_wheel_command_and_repeated_output_are_unique(tmp_path: Path) -> None:
    first = Runner(_runner_args(tmp_path))
    second = Runner(_runner_args(tmp_path))

    assert first.run_output != second.run_output
    assert first.output == second.output
    assert _wheel_build_argv(tmp_path / "repo", first.run_output / "dist") == [
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(first.run_output / "dist"),
    ]


def test_dynamic_port_selection_binds_loopback_ephemeral_port() -> None:
    calls: list[tuple[str, int]] = []

    class Listener:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def bind(self, address):
            calls.append(address)

        def getsockname(self):
            return ("127.0.0.1", 43123)

    assert _reserve_port(lambda *_args: Listener()) == 43123
    assert calls == [("127.0.0.1", 0)]


def test_failure_evidence_tolerates_missing_wheel(tmp_path: Path) -> None:
    runner = Runner(_runner_args(tmp_path))
    runner.record_failure("bootstrap", "fresh wheel was not produced")

    runner._write_evidence()
    evidence = json.loads((runner.output / "evidence.json").read_text())

    assert evidence["run_status"] == "failed"
    assert evidence["failures"] == [{"phase": "bootstrap", "error": "fresh wheel was not produced"}]
    assert evidence["source"]["wheel"] == {"path": None, "sha256": None}


def test_process_runner_emits_heartbeats_and_timeout_termination_reaps_process() -> None:
    heartbeats: list[float] = []
    result = _run_process(
        [sys.executable, "-c", "import time; time.sleep(0.16); print('done')"],
        timeout=2,
        heartbeat_seconds=0.04,
        on_heartbeat=heartbeats.append,
    )
    assert result.returncode == 0
    assert "done" in result.stdout
    assert len(heartbeats) >= 2

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    _terminate_process_group(process, grace_seconds=0.05)
    deadline = time.monotonic() + 1
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert process.poll() == -signal.SIGKILL
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("terminated subprocess still exists")


def test_semantic_excerpt_uses_assistant_output_not_echoed_user_input() -> None:
    payload = json.dumps(
        {
            "output": [
                {"type": "human", "content": "echo-only marker MASON_FALSE_PASS"},
                {"type": "ai", "content": "The actual MASON_ASSISTANT_OK result"},
            ]
        }
    ).encode()

    excerpt = _assistant_response_excerpt(payload)

    assert "MASON_ASSISTANT_OK" in excerpt
    assert "MASON_FALSE_PASS" not in excerpt
