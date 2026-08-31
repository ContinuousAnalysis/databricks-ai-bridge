#!/usr/bin/env python3
"""Run or verify the deterministic Mason agent-skills df1 proof matrix.

Verifier-only mode reads JSON and never initializes Databricks credentials.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

AUTHORING_PATHS = ("cli", "direct")
RUNTIMES = ("dev", "deploy")
PROMPT_KINDS = ("body", "file", "irrelevant")
SKILL_IDS = {
    "body": "body-guidance",
    "file": "file-guidance",
}
REQUIRED_CHECKS = {
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
}
HEARTBEAT_SECONDS = 55.0
_HASH = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SECRET_VALUE_FLAGS = {
    "--access-token",
    "--client-secret",
    "--password",
    "--secret",
    "--token",
}
_BEARER = re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)(?!<redacted>)[^\s,;]+")
_DAPI = re.compile(r"(?i)\bdapi[a-z0-9_-]{8,}\b")
_NAMED_SECRET = re.compile(
    r"(?i)((?:client[_-]?secret|access[_-]?token|refresh[_-]?token|password|secret|token|"
    r"api[_-]?key)"
    r"[\"']?\s*[:=]\s*[\"']?)(?!<redacted>)[^\s,;\"']+"
)
_SENSITIVE_KEY = re.compile(
    r"(?i)^(?:authorization|credential|password|secret|token|client[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|access[_-]?key|api[_-]?key)$"
)
_VERIFIER_SENSITIVE_KEYS = {
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "apikey",
    "accesskey",
}
_VERIFIER_ASSIGNMENT = re.compile(
    r"(?i)(?:client[_-]?secret|access[_-]?token|refresh[_-]?token|password|secret|token|"
    r"api[_-]?key|access[_-]?key)[\"']?\s*(?:=|:)\s*[\"']?"
    r"(?!<redacted>)[^\s,;\"']+"
)
_ACTIVATION = re.compile(
    r"(?P<prefix>MASON_E2E_ACTIVATION_[a-z0-9_]+) "
    r"op=(?P<op>load|read) skill_id=(?P<skill>[A-Za-z0-9_.-]+)"
    r"(?: path=(?P<path>[^\s]+))?"
)


class MatrixError(RuntimeError):
    """A reproducible setup, semantic, or cleanup failure."""


@dataclasses.dataclass(frozen=True)
class ProjectCase:
    authoring: str
    path: pathlib.Path
    app_name: str
    freshness_marker: str
    activation_prefix: str
    port: int


@dataclasses.dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def progress(state: str, label: str, detail: str = "") -> None:
    suffix = f" | {redact_text(detail)[:300]}" if detail else ""
    sys.stdout.write(f"{_now()} | {state} | {label}{suffix}\n")
    sys.stdout.flush()


def redact_text(value: object) -> str:
    """Remove credential-shaped values while preserving useful diagnostics."""
    text = str(value)
    text = _BEARER.sub(r"\1<redacted>", text)
    text = _DAPI.sub("<redacted>", text)
    return _NAMED_SECRET.sub(r"\1<redacted>", text)


def _redact_structure(value: Any, key: str | None = None) -> Any:
    if (
        key is not None
        and _SENSITIVE_KEY.search(key)
        and value not in (None, False, "", "<redacted>")
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(item_key): _redact_structure(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_structure(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_structure(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


def _redact_output(value: object) -> str:
    """Redact structured JSON without making command stdout invalid JSON."""
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return redact_text(text)
    return json.dumps(_redact_structure(parsed), sort_keys=True)


def _independent_secret_findings(value: Any, path: str = "$") -> list[str]:
    """Scan evidence independently from the producer's redaction expressions."""
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            sensitive_key = normalized_key in _VERIFIER_SENSITIVE_KEYS
            if sensitive_key and item not in (None, False, "", "<redacted>"):
                findings.append(child)
            findings.extend(_independent_secret_findings(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_independent_secret_findings(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "bearer " in lowered and "bearer <redacted>" not in lowered:
            findings.append(path)
        if re.search(r"\bdapi[a-z0-9_-]{8,}\b", value, re.IGNORECASE):
            findings.append(path)
        if _VERIFIER_ASSIGNMENT.search(value):
            findings.append(path)
    return sorted(set(findings))


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Return a safe argv representation for evidence and transcripts."""
    safe: list[str] = []
    redact_next = False
    for raw in argv:
        value = str(raw)
        if redact_next:
            safe.append("<redacted>")
            redact_next = False
            continue
        if value.lower() in _SECRET_VALUE_FLAGS:
            safe.append(value)
            redact_next = True
            continue
        redacted = redact_text(value)
        if redacted != value and "<redacted>" in redacted:
            safe.append("<redacted>" if "=" in value else redacted)
        else:
            safe.append(redacted)
    return safe


def _classify_cli_absence(returncode: int, stdout: str, stderr: str) -> bool:
    if returncode == 0:
        return False
    detail = f"{stdout}\n{stderr}"
    try:
        parsed = json.loads((stderr or stdout).removeprefix("Error:").strip())
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("error_code") == "RESOURCE_DOES_NOT_EXIST":
        return True
    if re.fullmatch(
        r"Error: App with name [a-z0-9][a-z0-9-]* does not exist or is deleted\.\s*", stderr
    ):
        return True
    return re.search(r"(?i)\bRESOURCE_DOES_NOT_EXIST\b", detail) is not None


def _app_health_class(status: int) -> str:
    if status == 200:
        return "ready"
    if status in {404, 409, 429, 502, 503, 504}:
        return "transient"
    return "terminal"


def _unexpected_repo_status(status: str) -> list[str]:
    ignored_prefixes = ("?? .superpowers/sdd/",)
    return [
        line
        for line in status.splitlines()
        if line.strip() and not line.startswith(ignored_prefixes)
    ]


def _cli_authoring_argvs(mason: pathlib.Path, project: pathlib.Path) -> list[list[str]]:
    commands = (
        ["skills", "add", "custom", f".claude/skills/{SKILL_IDS['body']}"],
        ["skills", "add", "custom", f".claude/skills/{SKILL_IDS['file']}"],
    )
    return [[str(mason), *args, "--source", str(project)] for args in commands]


def _direct_skill_toml() -> str:
    return (
        "\n[[skills]]\n"
        f'id = "{SKILL_IDS["body"]}"\n'
        f'source = {{ kind = "local", path = ".claude/skills/{SKILL_IDS["body"]}" }}\n'
        "\n[[skills]]\n"
        f'id = "{SKILL_IDS["file"]}"\n'
        f'source = {{ kind = "local", path = ".claude/skills/{SKILL_IDS["file"]}" }}\n'
    )


def _instrument_runtime_text(text: str, freshness: str, activation: str) -> str:
    future = "from __future__ import annotations\n"
    load = "    async def load_skill(skill_id: str) -> str:\n"
    read = "    async def read_skill_file(skill_id: str, path: str) -> str:\n"
    returned = "    return context, _tools(immutable)\n"
    if any(needle not in text for needle in (future, load, read, returned)):
        raise MatrixError("generated runtime shape changed; cannot add E2E instrumentation")
    text = text.replace(future, future + f'\nprint("{freshness} module_loaded", flush=True)\n', 1)
    text = text.replace(
        load,
        load + f'        print(f"{activation} op=load skill_id={{skill_id}}", flush=True)\n',
        1,
    )
    text = text.replace(
        read,
        read
        + f'        print(f"{activation} op=read skill_id={{skill_id}} path={{path}}", flush=True)\n',
        1,
    )
    return text.replace(
        returned, f'    print("{freshness} context_ready", flush=True)\n' + returned, 1
    )


def _find_marker_paths(root: pathlib.Path, marker: str) -> list[pathlib.Path]:
    matches: list[pathlib.Path] = []
    for directory, names, files in os.walk(root):
        relative = pathlib.Path(directory).relative_to(root)
        names[:] = [
            name
            for name in names
            if name != ".git" and not (relative == pathlib.Path(".superpowers") and name == "sdd")
        ]
        for name in files:
            path = pathlib.Path(directory) / name
            try:
                if path.stat().st_size <= 2 * 1024 * 1024 and marker in path.read_text(
                    encoding="utf-8", errors="ignore"
                ):
                    matches.append(path)
            except OSError:
                continue
    return sorted(matches)


def _wheel_build_argv(repo: pathlib.Path, destination: pathlib.Path) -> list[str]:
    del repo
    return ["uv", "build", "--wheel", "--out-dir", str(destination)]


def _reserve_port(socket_factory: Any = socket.socket) -> int:
    with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _terminate_process_group(process: subprocess.Popen[str], *, grace_seconds: float = 20) -> None:
    if process.poll() is not None:
        process.wait()
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=max(1.0, grace_seconds))


def _run_process(
    argv: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    on_heartbeat: Any = None,
) -> ProcessResult:
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout: list[str] = []
    stderr: list[str] = []

    def read(stream: Any, destination: list[str]) -> None:
        for line in stream:
            destination.append(line)

    readers = [
        threading.Thread(target=read, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=read, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    started = time.monotonic()
    next_heartbeat = heartbeat_seconds
    timed_out = False
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            timed_out = True
            _terminate_process_group(process)
            break
        if elapsed >= next_heartbeat:
            if on_heartbeat is not None:
                on_heartbeat(elapsed)
            next_heartbeat += heartbeat_seconds
        time.sleep(min(0.05, max(0.005, heartbeat_seconds / 4)))
    process.wait()
    for reader in readers:
        reader.join(timeout=2)
        if reader.is_alive():
            raise MatrixError("subprocess output reader did not terminate")
    return ProcessResult(process.returncode or 0, "".join(stdout), "".join(stderr), timed_out)


def _credential_findings(text: str) -> list[str]:
    findings: list[str] = []
    for label, pattern in (
        ("bearer credential", _BEARER),
        ("Databricks PAT", _DAPI),
        ("named credential", _NAMED_SECRET),
    ):
        if pattern.search(text):
            findings.append(label)
    return findings


def expected_matrix_cells() -> set[tuple[str, str, str]]:
    return {
        (authoring, runtime, prompt_kind)
        for authoring in AUTHORING_PATHS
        for runtime in RUNTIMES
        for prompt_kind in PROMPT_KINDS
    }


def _assistant_response_excerpt(payload: bytes) -> str:
    """Return only the final assistant message, excluding echoed human input."""
    try:
        response = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MatrixError("agent response was not valid UTF-8 JSON") from exc
    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(output, list):
        raise MatrixError("agent response did not contain an output message list")
    assistant = [
        item
        for item in output
        if isinstance(item, dict)
        and (item.get("type") in {"ai", "AIMessage"} or item.get("role") == "assistant")
    ]
    if not assistant:
        raise MatrixError("agent response did not contain an assistant message")
    return redact_text(json.dumps(assistant[-1], sort_keys=True, default=str))[-6000:]


def _expected_activations(prompt_kind: str) -> set[tuple[str, str, str | None]]:
    if prompt_kind == "body":
        return {("load", SKILL_IDS["body"], None)}
    if prompt_kind == "file":
        return {
            ("load", SKILL_IDS["file"], None),
            ("read", SKILL_IDS["file"], "facts.txt"),
        }
    return set()


def _object(value: object) -> dict[str, Any] | None:
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _validation_errors(document: object) -> list[str]:
    root = _object(document)
    if root is None:
        return ["evidence root must be an object"]
    errors: list[str] = []
    if root.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if root.get("run_status") != "passed" or root.get("failures") != []:
        errors.append("run did not reach a clean passed state")

    source = _object(root.get("source"))
    if source is None:
        errors.append("source must be an object")
    else:
        commit = source.get("git_commit")
        ref_commit = source.get("template_ref_commit")
        if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
            errors.append("source.git_commit must be a full SHA-1")
        if commit != ref_commit:
            errors.append("template ref commit is not the tested git commit")
        if source.get("git_status_clean_before") is not True:
            errors.append("tracked repository was not clean before the run")
        if source.get("git_status_clean_after") is not True:
            errors.append("tracked repository was not clean after the run")
        wheel = source.get("wheel")
        if not isinstance(wheel, dict) or not isinstance(wheel.get("sha256"), str):
            errors.append("source wheel hash is missing")
        elif _HASH.fullmatch(wheel["sha256"]) is None:
            errors.append("source wheel hash is not SHA-256")

    checks = root.get("validation_checks")
    if not isinstance(checks, list):
        errors.append("validation_checks must be an array")
    else:
        by_name: dict[str, dict[str, Any]] = {}
        for value in checks:
            item = _object(value)
            if item is not None and isinstance(item.get("name"), str):
                by_name[item["name"]] = item
        missing = REQUIRED_CHECKS - set(by_name)
        if missing:
            errors.append(f"missing validation checks: {sorted(missing)}")
        failed = sorted(
            name for name in REQUIRED_CHECKS & set(by_name) if by_name[name].get("status") != "pass"
        )
        if failed:
            errors.append(f"failed validation checks: {failed}")

    commands = root.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must contain recorded steps")
    else:
        for index, value in enumerate(commands):
            item = _object(value)
            if item is None:
                errors.append(f"commands[{index}] must be an object")
                continue
            for field in ("label", "started_at", "duration_seconds", "status"):
                if field not in item:
                    errors.append(f"commands[{index}] is missing {field}")
            if item.get("kind") == "command":
                if not isinstance(item.get("argv"), list) or "return_code" not in item:
                    errors.append(f"commands[{index}] has an incomplete command record")
            elif item.get("kind") == "http":
                if not item.get("request_class") or "http_status" not in item:
                    errors.append(f"commands[{index}] has an incomplete HTTP record")
            else:
                errors.append(f"commands[{index}] has an unknown kind")

    rows = root.get("rows")
    expected = expected_matrix_cells()
    if not isinstance(rows, list):
        errors.append("rows must be an array")
        rows = []
    cells: list[tuple[str, str, str]] = []
    deployed_row_apps: set[str] = set()
    expected_sources = {"body": "local", "file": "local", "irrelevant": "none"}
    for index, value in enumerate(rows):
        row = _object(value)
        if row is None:
            errors.append(f"rows[{index}] must be an object")
            continue
        cell = (str(row.get("authoring")), str(row.get("runtime")), str(row.get("prompt_kind")))
        cells.append(cell)
        if row.get("status") != "pass":
            errors.append(f"matrix row failed: {cell}")
        prompt_kind = cell[2]
        if row.get("source") != expected_sources.get(prompt_kind):
            errors.append(f"matrix row has wrong source: {cell}")
        marker = row.get("expected_marker")
        if not isinstance(marker, str) or marker not in str(row.get("actual_excerpt", "")):
            errors.append(f"matrix row lacks its semantic marker: {cell}")
        if row.get("freshness_observed") is not True or not row.get("freshness_marker"):
            errors.append(f"matrix row lacks freshness proof: {cell}")
        if cell[1] == "deploy" and (
            not row.get("app_name")
            or row.get("app_url_class") != "databricks_app"
            or not row.get("app_service_principal")
        ):
            errors.append(f"deployed row lacks App identity proof: {cell}")
        elif cell[1] == "deploy":
            deployed_row_apps.add(str(row["app_name"]))
        raw_activations = row.get("activations")
        if not isinstance(raw_activations, list):
            errors.append(f"matrix row lacks activation evidence: {cell}")
            continue
        actual_activations = {
            (item.get("op"), item.get("skill_id"), item.get("path"))
            for item in raw_activations
            if isinstance(item, dict)
        }
        if actual_activations != _expected_activations(prompt_kind):
            errors.append(f"matrix row selected wrong skill operations: {cell}")
    actual = set(cells)
    if actual != expected:
        errors.append(
            f"matrix cells differ; missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    if len(cells) != len(actual):
        errors.append("matrix contains duplicate cells")

    metrics = _object(root.get("metrics"))
    if metrics is None:
        errors.append("metrics must be an object")
    else:
        if metrics.get("eager_body_loads") != 0:
            errors.append("progressive-loading proof has eager body loads")
        if metrics.get("passed_rows") != len(expected) or metrics.get("failed_rows") != 0:
            errors.append("row metrics do not prove the complete matrix")
        if metrics.get("total_rows") != len(expected):
            errors.append("row total does not match the complete matrix")

    resources = _object(root.get("resources"))
    resource_apps = set(resources.get("apps", [])) if resources is not None else set()
    if len(resource_apps) != len(AUTHORING_PATHS) or resource_apps != deployed_row_apps:
        errors.append("resources must identify exactly the two Apps exercised by deployed rows")
    cleanup = _object(root.get("cleanup"))
    if cleanup is None:
        errors.append("cleanup must be an object")
    else:
        if cleanup.get("semantic_passed") is not True or cleanup.get("attempted") is not True:
            errors.append("cleanup did not follow a successful semantic matrix")
        apps = cleanup.get("apps")
        if (
            not isinstance(apps, list)
            or {item.get("name") for item in apps if isinstance(item, dict)} != resource_apps
        ):
            errors.append("cleanup does not cover both temporary Apps")
        else:
            for value in apps:
                item = _object(value)
                if item is None or (
                    item.get("delete_status") != "pass" or item.get("absence_verified") is not True
                ):
                    errors.append(f"App cleanup failed: {item.get('name') if item else None}")

    artifacts = root.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must contain hashed proof files")
    else:
        for index, value in enumerate(artifacts):
            artifact = _object(value)
            if artifact is None or not isinstance(artifact.get("sha256"), str):
                errors.append(f"artifacts[{index}] lacks a SHA-256")
            elif _HASH.fullmatch(artifact["sha256"]) is None:
                errors.append(f"artifacts[{index}] has an invalid SHA-256")
            else:
                artifact_path = artifact.get("path")
                if not isinstance(artifact_path, str) or not artifact_path:
                    errors.append(f"artifacts[{index}] lacks a nonempty path")
                    continue
                local_path = pathlib.Path(artifact_path)
                if not local_path.is_file():
                    errors.append(f"artifacts[{index}] is not an accessible file")
                elif _sha256(local_path) != artifact["sha256"]:
                    errors.append(f"artifacts[{index}] does not match its accessible file")
        source_wheel = source.get("wheel") if isinstance(source, dict) else None
        wheel_artifacts = []
        for value in artifacts:
            artifact = _object(value)
            if artifact is not None and artifact.get("kind") == "wheel":
                wheel_artifacts.append(artifact)
        if (
            len(wheel_artifacts) != 1
            or not isinstance(source_wheel, dict)
            or wheel_artifacts[0].get("path") != source_wheel.get("path")
            or wheel_artifacts[0].get("sha256") != source_wheel.get("sha256")
        ):
            errors.append("source wheel does not match the canonical wheel artifact hash")

    redactions = _object(root.get("redactions"))
    if redactions is None or (
        redactions.get("scan_status") != "pass"
        or redactions.get("findings") != []
        or redactions.get("credentials_recorded") is not False
    ):
        errors.append("redaction proof did not pass")
    findings = _independent_secret_findings(root)
    if findings:
        errors.append(f"independent credential scan found: {findings}")
    return errors


def verify_evidence(path: pathlib.Path) -> int:
    """Verify canonical evidence without consulting credentials or external services."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        sys.stdout.write(f"evidence invalid: {exc}\n")
        return 1
    errors = _validation_errors(document)
    rows = document.get("rows", []) if isinstance(document, dict) else []
    passed = sum(isinstance(row, dict) and row.get("status") == "pass" for row in rows)
    failed = len(rows) - passed if isinstance(rows, list) else 0
    skipped = len(expected_matrix_cells()) - len(
        {
            (row.get("authoring"), row.get("runtime"), row.get("prompt_kind"))
            for row in rows
            if isinstance(row, dict)
        }
        & expected_matrix_cells()
    )
    sys.stdout.write(f"{passed} passed, {failed} failed, {skipped} skipped\n")
    for error in errors:
        sys.stdout.write(f"- {error}\n")
    return 1 if errors else 0


class Transcript:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, value: object) -> None:
        line = redact_text(value).rstrip() + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)

    def command(self, argv: Sequence[str], cwd: pathlib.Path | None) -> None:
        prefix = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
        self.write(f"$ {prefix}{shlex.join(redact_argv(argv))}")


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.run_token = uuid.uuid4().hex[:8]
        self.run_id = f"mason-skills-{dt.datetime.now(dt.timezone.utc):%Y%m%d}-{self.run_token}"
        self.run_output = self.output / "runs" / self.run_id
        self.run_output.mkdir(parents=True, exist_ok=False)
        self.transcript = Transcript(self.run_output / "commands.log")
        self.repo = pathlib.Path(args.template_repo).resolve()
        self.wheel: pathlib.Path | None = None
        self.runner_venv = self.run_output / "runner-venv"
        self.mason = self.runner_venv / "bin" / "mason"
        self.commands: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.cleanup_evidence: dict[str, Any] = {
            "semantic_passed": False,
            "attempted": False,
            "apps": [],
        }
        self.artifacts: list[dict[str, str]] = []
        self.workspace_headers: dict[str, str] = {}
        self.app_headers: dict[str, str] = {}
        self.host = ""
        self.apps: list[str] = []
        self.cases: list[ProjectCase] = []
        self.eager_body_loads = 0
        self.git_commit = ""
        self.template_ref_commit = ""
        self.clean_before = False
        self.clean_after = False
        self.mutation_started_at_step: int | None = None
        self.run_status = "pending"
        self.failures: list[dict[str, str]] = []
        self.ports: dict[str, int] = {}

    def _record_check(self, name: str, passed: bool, evidence: str) -> None:
        self.checks = [item for item in self.checks if item["name"] != name]
        self.checks.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "evidence": redact_text(evidence),
            }
        )
        self._write_evidence()

    def record_failure(self, phase: str, error: object) -> None:
        self.run_status = "failed"
        self.failures.append({"phase": phase, "error": redact_text(error)})

    def run_command(
        self,
        label: str,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 300,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        progress("START", label)
        self.transcript.command(argv, cwd)
        started_at = _now()
        started = time.monotonic()
        result = _run_process(
            argv,
            cwd=cwd,
            env=env,
            timeout=timeout,
            heartbeat_seconds=HEARTBEAT_SECONDS,
            on_heartbeat=lambda elapsed: progress("PROGRESS", label, f"running {elapsed:.0f}s"),
        )
        stdout = _redact_output(result.stdout)
        stderr = _redact_output(result.stderr)
        duration = round(time.monotonic() - started, 3)
        if result.timed_out:
            self._append_command(
                label, argv, cwd, started_at, duration, 124, "fail", stdout, stderr
            )
            progress("ERROR", label, f"timed out after {duration:.0f}s")
            raise MatrixError(f"{label} timed out after {timeout:.0f}s")
        status = "pass" if result.returncode == 0 else "fail"
        self._append_command(
            label, argv, cwd, started_at, duration, result.returncode, status, stdout, stderr
        )
        self.transcript.write(stdout)
        self.transcript.write(stderr)
        progress("PASS" if result.returncode == 0 else "ERROR", label, f"rc={result.returncode}")
        if check and result.returncode != 0:
            raise MatrixError(f"{label} failed ({result.returncode}): {(stderr or stdout)[-2000:]}")
        return subprocess.CompletedProcess(
            list(argv), result.returncode, stdout=stdout, stderr=stderr
        )

    def _append_command(
        self,
        label: str,
        argv: Sequence[str],
        cwd: pathlib.Path | None,
        started_at: str,
        duration: float,
        return_code: int,
        status: str,
        stdout: str,
        stderr: str,
    ) -> None:
        self.commands.append(
            {
                "kind": "command",
                "label": label,
                "argv": redact_argv(argv),
                "cwd": str(cwd) if cwd else None,
                "started_at": started_at,
                "duration_seconds": duration,
                "return_code": return_code,
                "status": status,
                "stdout_excerpt": stdout[-2000:],
                "stderr_excerpt": stderr[-2000:],
            }
        )
        self._write_evidence()

    def run_long(
        self,
        label: str,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        timeout: float = 1800,
    ) -> str:
        log_path = self.run_output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.run_command(label, argv, cwd=cwd, timeout=timeout)
        text = result.stdout + ("\n" + result.stderr if result.stderr else "")
        log_path.write_text(text, encoding="utf-8")
        return text

    def http(
        self,
        label: str,
        method: str,
        url: str,
        *,
        request_class: str,
        headers: Mapping[str, str] | None = None,
        body: dict[str, Any] | bytes | None = None,
        expected_statuses: set[int] | None = None,
        timeout: float = 360,
    ) -> tuple[int, bytes]:
        expected_statuses = expected_statuses or {200}
        progress("START", label)
        encoded: bytes | None
        content_type: str | None = None
        if isinstance(body, dict):
            encoded = json.dumps(body).encode("utf-8")
            content_type = "application/json"
        else:
            encoded = body
            if isinstance(body, bytes):
                content_type = "application/octet-stream"
        private_headers = dict(headers or {})
        if content_type:
            private_headers["Content-Type"] = content_type
        request = urllib.request.Request(url, data=encoded, headers=private_headers, method=method)
        started_at = _now()
        started = time.monotonic()
        stop_heartbeat = threading.Event()

        def heartbeat() -> None:
            while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
                progress("PROGRESS", label, f"HTTP running {time.monotonic() - started:.0f}s")

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = response.status
                    payload = response.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                payload = exc.read()
            except (OSError, TimeoutError) as exc:
                duration = round(time.monotonic() - started, 3)
                self._append_http(
                    label, method, url, request_class, started_at, duration, 0, "fail", str(exc)
                )
                raise MatrixError(f"{label} transport failed: {redact_text(exc)}") from exc
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2)
            if heartbeat_thread.is_alive():
                raise MatrixError(f"{label} heartbeat did not terminate")
        duration = round(time.monotonic() - started, 3)
        excerpt = _redact_output(payload.decode("utf-8", errors="replace"))[-2000:]
        passed = status in expected_statuses
        self._append_http(
            label,
            method,
            url,
            request_class,
            started_at,
            duration,
            status,
            "pass" if passed else "fail",
            excerpt,
        )
        progress("PASS" if passed else "ERROR", label, f"http={status}")
        if not passed:
            raise MatrixError(f"{label} returned HTTP {status}: {excerpt}")
        return status, payload

    def _append_http(
        self,
        label: str,
        method: str,
        url: str,
        request_class: str,
        started_at: str,
        duration: float,
        http_status: int,
        status: str,
        excerpt: str,
    ) -> None:
        parsed = urllib.parse.urlsplit(url)
        self.commands.append(
            {
                "kind": "http",
                "label": label,
                "request_class": request_class,
                "method": method,
                "path_class": parsed.path,
                "started_at": started_at,
                "duration_seconds": duration,
                "http_status": http_status,
                "status": status,
                "response_excerpt": _redact_output(excerpt),
            }
        )
        self._write_evidence()

    def bootstrap(self) -> None:
        progress("START", "bootstrap")
        self.git_commit = self.run_command(
            "git-head", ["git", "rev-parse", "HEAD"], cwd=self.repo
        ).stdout.strip()
        self.template_ref_commit = self.run_command(
            "git-template-ref",
            ["git", "rev-parse", f"{self.args.template_ref}^{{commit}}"],
            cwd=self.repo,
        ).stdout.strip()
        before = self.run_command(
            "git-status-before",
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.repo,
        ).stdout
        unexpected = _unexpected_repo_status(before)
        self.clean_before = not unexpected
        marker_paths = _find_marker_paths(self.repo, self.run_token)
        output_outside_repo = True
        try:
            self.output.relative_to(self.repo)
            output_outside_repo = False
        except ValueError:
            pass
        self._record_check(
            "preflight_repository_clean",
            self.clean_before and not marker_paths and output_outside_repo,
            f"unexpected_status={unexpected}; marker_paths={marker_paths}; "
            f"output_outside_repo={output_outside_repo}",
        )
        self._record_check(
            "git_commit_current",
            self.git_commit == self.template_ref_commit and self.clean_before,
            f"HEAD={self.git_commit}; ref={self.template_ref_commit}; full_clean={self.clean_before}",
        )
        ports = [_reserve_port(), _reserve_port()]
        if len(set(ports)) != len(ports):
            ports[1] = _reserve_port()
        self.ports = dict(zip(AUTHORING_PATHS, ports, strict=True))
        self._record_check(
            "preflight_dynamic_ports",
            len(set(self.ports.values())) == len(AUTHORING_PATHS)
            and all(port > 0 for port in self.ports.values()),
            f"reserved loopback ports={sorted(self.ports.values())}",
        )

        dist = self.run_output / "dist"
        dist.mkdir(parents=True, exist_ok=False)
        self.run_long(
            "build-current-wheel",
            _wheel_build_argv(self.repo, dist),
            cwd=self.repo / "integrations" / "mason",
            timeout=900,
        )
        wheels = sorted(dist.glob("databricks_mason-*.whl"))
        if len(wheels) != 1 or not wheels[0].is_file():
            self._record_check("wheel_hash_current", False, f"fresh wheel candidates={wheels}")
            raise MatrixError(f"fresh build produced {len(wheels)} Mason wheels")
        self.wheel = wheels[0].resolve()
        wheel_hash = _sha256(self.wheel)
        self._record_check(
            "wheel_hash_current",
            bool(_HASH.fullmatch(wheel_hash)),
            f"fresh_wheel={self.wheel}; wheel_sha256={wheel_hash}",
        )
        self.run_command("runner-venv", ["uv", "venv", str(self.runner_venv)], timeout=300)
        self.run_command(
            "runner-install-wheel",
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.runner_venv / "bin" / "python"),
                str(self.wheel),
            ],
            timeout=600,
        )
        self.run_command("mason-skills-help", [str(self.mason), "skills", "--help"])

        from databricks.sdk import WorkspaceClient

        workspace = WorkspaceClient(profile=self.args.profile)
        app_auth = WorkspaceClient(profile=self.args.app_auth_profile)
        if not workspace.config.host or not app_auth.config.host:
            self._record_check(
                "preflight_auth_profiles", False, "one or both profiles have no workspace host"
            )
            raise MatrixError("could not resolve both workspace hosts")
        self.host = workspace.config.host.rstrip("/")
        if app_auth.config.host.rstrip("/") != self.host:
            self._record_check(
                "preflight_auth_profiles", False, "workspace profiles target different hosts"
            )
            raise MatrixError("workspace and App-auth profiles target different hosts")
        if app_auth.config.auth_type == "pat":
            self._record_check("preflight_auth_profiles", False, "App auth profile resolves to PAT")
            raise MatrixError("deployed App /api calls require an OAuth App-auth profile")
        self.workspace_headers = dict(workspace.config.authenticate())
        self.app_headers = dict(app_auth.config.authenticate())
        if not self.workspace_headers.get("Authorization") or not self.app_headers.get(
            "Authorization"
        ):
            self._record_check(
                "preflight_auth_profiles", False, "one or both profiles did not authenticate"
            )
            raise MatrixError("profile authentication did not produce authorization headers")
        self._record_check(
            "preflight_auth_profiles",
            app_auth.config.auth_type != "pat",
            f"profiles resolve same host; app_auth_type={app_auth.config.auth_type}",
        )
        progress("PASS", "bootstrap")

    def probe_capabilities(self) -> None:
        """Run all read-only/local probes before the first workspace mutation."""
        start = len(self.commands)
        self.run_command("probe-databricks-version", ["databricks", "--version"])
        self.run_command("probe-apps-create-help", ["databricks", "apps", "create", "--help"])
        self.run_command(
            "probe-apps-list",
            ["databricks", "apps", "list", "--profile", self.args.profile, "-o", "json"],
        )
        model = self.run_command(
            "probe-model-endpoint",
            [
                "databricks",
                "serving-endpoints",
                "get",
                "databricks-gpt-5-2",
                "--profile",
                self.args.profile,
                "-o",
                "json",
            ],
        )
        self._record_check(
            "preflight_model_available",
            "databricks-gpt-5-2" in model.stdout,
            "model endpoint databricks-gpt-5-2 is readable",
        )
        if "databricks-gpt-5-2" not in model.stdout:
            raise MatrixError("required model endpoint databricks-gpt-5-2 was not proven")
        apps = next(item for item in self.commands if item.get("label") == "probe-apps-list")
        self._record_check(
            "preflight_apps_available",
            apps.get("status") == "pass",
            "Apps CLI and list control-plane request succeeded",
        )
        self.mutation_started_at_step = len(self.commands)
        labels = [item["label"] for item in self.commands[start:]]
        preflight_names = {name for name in REQUIRED_CHECKS if name.startswith("preflight_")} | {
            "git_commit_current",
            "wheel_hash_current",
        }
        passed_preflights = {
            item["name"]
            for item in self.checks
            if item["name"] in preflight_names and item["status"] == "pass"
        }
        preflight_ok = passed_preflights == preflight_names
        self._record_check(
            "capabilities_probed_before_mutation",
            len(labels) >= 4
            and self.mutation_started_at_step == len(self.commands)
            and preflight_ok,
            f"read-only probes completed before mutation step {self.mutation_started_at_step}: {labels}",
        )
        if not preflight_ok:
            raise MatrixError(
                f"preflight did not prove: {sorted(preflight_names - passed_preflights)}"
            )

    def _markers(self) -> dict[str, str]:
        upper = self.run_token.upper()
        return {
            "body": f"MASON_SKILL_BODY_OK_{upper}",
            "file": f"MASON_SKILL_FILE_OK_{upper}",
            "irrelevant": f"MASON_IRRELEVANT_OK_{upper}_4",
        }

    def create_projects(self) -> None:
        root = self.run_output / "projects"
        root.mkdir(parents=True, exist_ok=True)
        for authoring in AUTHORING_PATHS:
            project = root / f"langgraph-{authoring}"
            init_args = [
                str(self.mason),
                "--profile",
                self.args.profile,
                "init",
                "--framework",
                "langgraph",
                "--profile",
                self.args.profile,
                "--repo",
                str(self.repo),
                "--ref",
                self.args.template_ref,
                str(project),
            ]
            self.run_long(f"init-{authoring}", init_args, timeout=900)
            self._write_local_skills(project)
            if authoring == "cli":
                self._author_cli(project)
            else:
                self._author_direct(project)
            listing = self.run_command(
                f"skills-list-{authoring}",
                [
                    str(self.mason),
                    "--output",
                    "json",
                    "skills",
                    "list",
                    "--source",
                    str(project),
                ],
            )
            listed = json.loads(listing.stdout)
            listed_ids = {item.get("id") for item in listed.get("skills", [])}
            if listed_ids != set(SKILL_IDS.values()):
                raise MatrixError(
                    f"skills list for {authoring} returned {sorted(listed_ids)} instead of "
                    f"{sorted(SKILL_IDS.values())}"
                )
            freshness = f"MASON_E2E_FRESHNESS_{self.run_token}_{authoring}"
            activation = f"MASON_E2E_ACTIVATION_{self.run_token}_{authoring}"
            self._instrument_generated_runtime(project, freshness, activation)
            app_name = f"mason-sk-{authoring[:3]}-{self.run_token}"
            self.apps.append(app_name)
            self.cleanup_evidence["apps"].append(
                {"name": app_name, "delete_status": "not_attempted", "absence_verified": False}
            )
            self.cases.append(
                ProjectCase(
                    authoring,
                    project,
                    app_name,
                    freshness,
                    activation,
                    self.ports[authoring],
                )
            )
        assembled = len(self.cases) == len(AUTHORING_PATHS) and all(
            (case.path / "agent.toml").is_file()
            and case.freshness_marker
            in (case.path / "agent" / "mason" / "skill_runtime.py").read_text(encoding="utf-8")
            for case in self.cases
        )
        self._record_check(
            "preflight_template_assembly",
            assembled,
            "CLI/direct manifests and generated-only runtime instrumentation assembled",
        )
        self._record_check(
            "custom_skill_listing",
            len(self.cases) == len(AUTHORING_PATHS),
            "CLI/direct projects listed both exact project-local skill declarations",
        )
        self._write_evidence()

    def _write_local_skills(self, project: pathlib.Path) -> None:
        markers = self._markers()
        body = project / ".claude" / "skills" / SKILL_IDS["body"]
        body.mkdir(parents=True, exist_ok=True)
        (body / "SKILL.md").write_text(
            "---\n"
            f"name: {SKILL_IDS['body']}\n"
            "description: Provides the deterministic harbor color phrase for Mason validation.\n"
            "---\n"
            "When asked for the harbor color phrase, return exactly this marker and nothing else:\n"
            f"{markers['body']}\n",
            encoding="utf-8",
        )
        file_skill = project / ".claude" / "skills" / SKILL_IDS["file"]
        file_skill.mkdir(parents=True, exist_ok=True)
        (file_skill / "SKILL.md").write_text(
            "---\n"
            f"name: {SKILL_IDS['file']}\n"
            "description: Looks up the deterministic lighthouse fact for Mason validation.\n"
            "---\n"
            "When asked for the lighthouse fact, call read_skill_file for facts.txt and return "
            "the exact marker in that file.\n",
            encoding="utf-8",
        )
        (file_skill / "facts.txt").write_text(markers["file"] + "\n", encoding="utf-8")

    def _author_cli(self, project: pathlib.Path) -> None:
        for index, argv in enumerate(_cli_authoring_argvs(self.mason, project)):
            self.run_command(
                f"author-cli-{index + 1}",
                argv,
            )

    def _author_direct(self, project: pathlib.Path) -> None:
        manifest = project / "agent.toml"
        addition = _direct_skill_toml()
        manifest.write_text(manifest.read_text(encoding="utf-8") + addition, encoding="utf-8")
        self.commands.append(
            {
                "kind": "command",
                "label": "author-direct-agent-toml",
                "argv": ["write-generated-agent-toml", str(manifest)],
                "cwd": str(project),
                "started_at": _now(),
                "duration_seconds": 0.0,
                "return_code": 0,
                "status": "pass",
                "stdout_excerpt": "appended two exact [[skills]] bindings",
                "stderr_excerpt": "",
            }
        )
        self._write_evidence()

    def _instrument_generated_runtime(
        self, project: pathlib.Path, freshness: str, activation: str
    ) -> None:
        runtime = project / "agent" / "mason" / "skill_runtime.py"
        text = runtime.read_text(encoding="utf-8")
        runtime.write_text(_instrument_runtime_text(text, freshness, activation), encoding="utf-8")
        if _find_marker_paths(self.repo, self.run_token):
            raise MatrixError("test-only runtime marker was written into repository files")

    def _prompt_probe(self, case: ProjectCase) -> None:
        probe = (
            "import asyncio; "
            "from agent.mason.skill_runtime import build_skill_context; "
            "context, tools = asyncio.run(build_skill_context()); "
            "print('MASON_E2E_PROBE tools=' + ','.join(t.name for t in tools)); "
            "print(context)"
        )
        environment = dict(os.environ)
        environment["MASON_PROJECT_ROOT"] = str(case.path)
        environment["DATABRICKS_CONFIG_PROFILE"] = self.args.profile
        result = self.run_command(
            f"prompt-probe-{case.authoring}",
            [str(case.path / ".venv" / "bin" / "python"), "-c", probe],
            cwd=case.path,
            timeout=300,
            env=environment,
        )
        output = result.stdout + result.stderr
        eager = len(_activation_events(output, case.activation_prefix))
        self.eager_body_loads += eager
        body_markers = set(self._markers().values()) - {self._markers()["irrelevant"]}
        metadata_only = (
            f"{case.freshness_marker} context_ready" in output
            and eager == 0
            and not any(marker in output for marker in body_markers)
            and "load_skill" in output
            and "read_skill_file" in output
        )
        if not metadata_only:
            raise MatrixError(f"metadata-only prompt probe failed for {case.authoring}")

    def run_dev(self, case: ProjectCase) -> None:
        port = case.port
        label = f"dev-{case.authoring}"
        log_path = self.run_output / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.mason),
            "--profile",
            self.args.profile,
            "dev",
            "--source",
            str(case.path),
            "--app-port",
            str(port),
            "--prepare-environment",
        ]
        progress("START", label)
        self.transcript.command(argv, None)
        started_at = _now()
        started = time.monotonic()
        output = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                safe = redact_text(line)
                output.write(safe)
                output.flush()
                self.transcript.write(safe)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        error: Exception | None = None
        try:
            self._wait_local(process, port, label, log_path)
            self._prompt_probe(case)
            before = log_path.read_text(encoding="utf-8", errors="replace")
            eager = len(_activation_events(before, case.activation_prefix))
            self.eager_body_loads += eager
            if eager:
                raise MatrixError(f"{label} loaded skill bodies before its first prompt")
            self._exercise(case, "dev", f"http://127.0.0.1:{port}", {}, log_path)
        except Exception as exc:
            error = exc
            self._record_runtime_failure(case, "dev", exc, log_path)
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
            reader.join(timeout=10)
            output.close()
            text = log_path.read_text(encoding="utf-8", errors="replace")
            self._append_command(
                label,
                argv,
                None,
                started_at,
                round(time.monotonic() - started, 3),
                process.returncode or 0,
                "fail" if error else "pass",
                text,
                str(error or "intentional shutdown"),
            )
            progress("ERROR" if error else "PASS", label, str(error or "stopped"))

    def _wait_local(
        self,
        process: subprocess.Popen[str],
        port: int,
        label: str,
        log_path: pathlib.Path,
    ) -> None:
        started = time.monotonic()
        next_tick = HEARTBEAT_SECONDS
        while time.monotonic() - started < 1200:
            if process.poll() is not None:
                raise MatrixError(
                    f"{label} exited {process.returncode}: {_last_lines(log_path, 30)}"
                )
            try:
                self.http(
                    f"{label}-health",
                    "GET",
                    f"http://127.0.0.1:{port}/health",
                    request_class="local_health_probe",
                    expected_statuses={200},
                    timeout=5,
                )
                return
            except MatrixError:
                pass
            elapsed = time.monotonic() - started
            if elapsed >= next_tick:
                progress("PROGRESS", label, f"starting {elapsed:.0f}s; {_last_nonempty(log_path)}")
                next_tick += HEARTBEAT_SECONDS
            time.sleep(5)
        raise MatrixError(f"{label} did not become healthy: {_last_lines(log_path, 30)}")

    def deploy(self, case: ProjectCase) -> None:
        label = f"deploy-{case.authoring}"
        try:
            self.run_long(
                label,
                [
                    str(self.mason),
                    "--profile",
                    self.args.profile,
                    "deploy",
                    case.app_name,
                    "--source",
                    str(case.path),
                ],
                timeout=2400,
            )
            app = self._wait_app(case.app_name)
            principal = str(app.get("service_principal_client_id") or "")
            if not principal:
                raise MatrixError(f"App {case.app_name} has no service principal identity")
            url = str(app.get("url") or "").rstrip("/")
            if not url:
                raise MatrixError(f"App {case.app_name} has no URL")
            self._wait_deployed_health(case, url)
            log_path = self._wait_app_log(case, case.freshness_marker)
            before = log_path.read_text(encoding="utf-8", errors="replace")
            eager = len(_activation_events(before, case.activation_prefix))
            self.eager_body_loads += eager
            if eager:
                raise MatrixError(f"{label} loaded skill bodies before its first prompt")
            self._exercise(
                case,
                "deploy",
                url,
                self.app_headers,
                log_path,
                app_name=case.app_name,
                app_principal=principal,
            )
        except Exception as exc:
            self._record_runtime_failure(
                case,
                "deploy",
                exc,
                self.run_output / "logs" / f"app-{case.authoring}.log",
                case.app_name,
            )

    def _wait_app(self, name: str) -> dict[str, Any]:
        started = time.monotonic()
        next_tick = 0.0
        while time.monotonic() - started < 1200:
            result = self.run_command(
                f"app-get-{name}",
                ["databricks", "apps", "get", name, "--profile", self.args.profile, "-o", "json"],
                check=False,
            )
            if result.returncode == 0:
                try:
                    app = json.loads(result.stdout)
                except json.JSONDecodeError:
                    app = {}
                state = app.get("compute_status", {}).get("state")
                if state == "ACTIVE" and app.get("url"):
                    return app
            elapsed = time.monotonic() - started
            if elapsed >= next_tick:
                progress("PROGRESS", f"app-{name}", f"waiting {elapsed:.0f}s")
                next_tick += HEARTBEAT_SECONDS
            time.sleep(15)
        raise MatrixError(f"App {name} did not become ACTIVE")

    def _wait_deployed_health(self, case: ProjectCase, base_url: str) -> None:
        started = time.monotonic()
        while time.monotonic() - started < 600:
            try:
                status, _ = self.http(
                    f"app-health-{case.authoring}",
                    "GET",
                    f"{base_url}/api/health",
                    request_class="deployed_authenticated_health",
                    headers=self.app_headers,
                    expected_statuses={200, 401, 403, 404, 409, 422, 429, 500, 502, 503, 504},
                    timeout=60,
                )
            except MatrixError as exc:
                progress("PROGRESS", f"app-health-{case.authoring}", f"transport: {exc}")
                time.sleep(10)
                continue
            classification = _app_health_class(status)
            if classification == "ready":
                return
            if classification == "terminal":
                raise MatrixError(f"authenticated App health returned terminal HTTP {status}")
            progress("PROGRESS", f"app-health-{case.authoring}", f"transient HTTP {status}")
            time.sleep(10)
        raise MatrixError(f"App {case.app_name} /api/health did not become ready")

    def _fetch_app_log(self, case: ProjectCase) -> pathlib.Path:
        result = self.run_command(
            f"app-logs-{case.authoring}",
            [
                "databricks",
                "apps",
                "logs",
                case.app_name,
                "--tail-lines",
                "1000",
                "--profile",
                self.args.profile,
            ],
            timeout=180,
        )
        path = self.run_output / "logs" / f"app-{case.authoring}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact_text(result.stdout + result.stderr), encoding="utf-8")
        return path

    def _wait_app_log(self, case: ProjectCase, needle: str) -> pathlib.Path:
        started = time.monotonic()
        next_tick = HEARTBEAT_SECONDS
        path = self.run_output / "logs" / f"app-{case.authoring}.log"
        while time.monotonic() - started < 300:
            path = self._fetch_app_log(case)
            text = path.read_text(encoding="utf-8", errors="replace")
            if needle in text:
                return path
            elapsed = time.monotonic() - started
            if elapsed >= next_tick:
                progress("PROGRESS", f"app-log-{case.authoring}", f"waiting {elapsed:.0f}s")
                next_tick += HEARTBEAT_SECONDS
            time.sleep(10)
        raise MatrixError(f"App log did not show freshness marker {needle}")

    def _exercise(
        self,
        case: ProjectCase,
        runtime: str,
        base_url: str,
        headers: Mapping[str, str],
        log_path: pathlib.Path,
        *,
        app_name: str | None = None,
        app_principal: str | None = None,
    ) -> None:
        prompts = self._prompts()
        invocation_path = "/api/invocations" if runtime == "deploy" else "/invocations"
        for prompt_kind in PROMPT_KINDS:
            before = log_path.read_text(encoding="utf-8", errors="replace")
            before_events = _activation_events(before, case.activation_prefix)
            expected_marker = self._markers()[prompt_kind]
            started = time.monotonic()
            error: str | None = None
            response_excerpt = ""
            activations: list[dict[str, str]] = []
            freshness = case.freshness_marker in before
            try:
                _, payload = self.http(
                    f"invoke-{runtime}-{case.authoring}-{prompt_kind}",
                    "POST",
                    f"{base_url}{invocation_path}",
                    request_class="agent_invocation",
                    headers=headers,
                    body={"input": [{"role": "user", "content": prompts[prompt_kind]}]},
                    expected_statuses={200},
                    timeout=360,
                )
                response_excerpt = _assistant_response_excerpt(payload)
                if expected_marker not in response_excerpt:
                    raise MatrixError(f"response lacks semantic marker {expected_marker}")
                if runtime == "deploy":
                    log_path = self._wait_for_activation_delta(
                        case, before_events, _expected_activations(prompt_kind)
                    )
                else:
                    time.sleep(1)
                after = log_path.read_text(encoding="utf-8", errors="replace")
                after_events = _activation_events(after, case.activation_prefix)
                activations = after_events[len(before_events) :]
                actual = {(item["op"], item["skill_id"], item.get("path")) for item in activations}
                if actual != _expected_activations(prompt_kind):
                    raise MatrixError(
                        f"activation mismatch for {prompt_kind}: expected "
                        f"{sorted(_expected_activations(prompt_kind))}, actual {sorted(actual)}"
                    )
                freshness = case.freshness_marker in after
                if not freshness:
                    raise MatrixError(f"freshness marker absent from {runtime} logs")
                status = "pass"
            except Exception as exc:
                status = "fail"
                error = redact_text(exc)
            self.rows.append(
                {
                    "authoring": case.authoring,
                    "runtime": runtime,
                    "prompt_kind": prompt_kind,
                    "source": {
                        "body": "local",
                        "file": "local",
                        "irrelevant": "none",
                    }[prompt_kind],
                    "status": status,
                    "expected_marker": expected_marker,
                    "actual_excerpt": response_excerpt,
                    "activations": activations,
                    "freshness_marker": case.freshness_marker,
                    "freshness_observed": freshness,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "app_name": app_name,
                    "app_url_class": "databricks_app" if app_name else None,
                    "app_service_principal": app_principal,
                    "artifact_paths": [str(log_path)],
                    "error": error,
                }
            )
            self._write_evidence()

    def _wait_for_activation_delta(
        self,
        case: ProjectCase,
        before_events: list[dict[str, str]],
        expected: set[tuple[str, str, str | None]],
    ) -> pathlib.Path:
        started = time.monotonic()
        path = self.run_output / "logs" / f"app-{case.authoring}.log"
        stable_matches = 0
        while time.monotonic() - started < 180:
            path = self._fetch_app_log(case)
            events = _activation_events(
                path.read_text(encoding="utf-8", errors="replace"), case.activation_prefix
            )
            delta = events[len(before_events) :]
            actual = {(item["op"], item["skill_id"], item.get("path")) for item in delta}
            if actual == expected:
                stable_matches += 1
                if stable_matches >= 2:
                    return path
            else:
                stable_matches = 0
            time.sleep(5)
        return path

    def _prompts(self) -> dict[str, str]:
        markers = self._markers()
        return {
            "body": (
                "Use the available skill whose description supplies the harbor color phrase. "
                "Load only that matching skill and return its exact validation marker."
            ),
            "file": (
                "Use the available skill that looks up the lighthouse fact. Load that skill, read "
                "only its referenced facts.txt file, and return the exact marker from the file."
            ),
            "irrelevant": (
                "Do not load any skill and do not call any tool. Compute 2 + 2, then reply exactly "
                f"{markers['irrelevant']}"
            ),
        }

    def _record_runtime_failure(
        self,
        case: ProjectCase,
        runtime: str,
        exc: Exception,
        log_path: pathlib.Path,
        app_name: str | None = None,
    ) -> None:
        existing = {
            row["prompt_kind"]
            for row in self.rows
            if row["authoring"] == case.authoring and row["runtime"] == runtime
        }
        for prompt_kind in PROMPT_KINDS:
            if prompt_kind in existing:
                continue
            self.rows.append(
                {
                    "authoring": case.authoring,
                    "runtime": runtime,
                    "prompt_kind": prompt_kind,
                    "source": {
                        "body": "local",
                        "file": "local",
                        "irrelevant": "none",
                    }[prompt_kind],
                    "status": "fail",
                    "expected_marker": self._markers()[prompt_kind],
                    "actual_excerpt": "",
                    "activations": [],
                    "freshness_marker": case.freshness_marker,
                    "freshness_observed": False,
                    "duration_seconds": 0.0,
                    "app_name": app_name,
                    "app_url_class": "databricks_app" if app_name else None,
                    "app_service_principal": None,
                    "artifact_paths": [str(log_path)],
                    "error": redact_text(exc),
                }
            )
        self._write_evidence()

    def semantic_passed(self) -> bool:
        cells = {
            (row["authoring"], row["runtime"], row["prompt_kind"])
            for row in self.rows
            if row["status"] == "pass"
        }
        return cells == expected_matrix_cells() and len(self.rows) == len(cells)

    def cleanup(self) -> None:
        self.cleanup_evidence["semantic_passed"] = self.semantic_passed()
        if not self.semantic_passed():
            progress("ERROR", "cleanup", "semantic failure; resources preserved for diagnosis")
            self._write_evidence()
            return
        if self.args.keep_resources:
            progress("ERROR", "cleanup", "--keep-resources requested; verifier will remain red")
            self._write_evidence()
            return
        self.cleanup_evidence["attempted"] = True
        for item in self.cleanup_evidence["apps"]:
            name = item["name"]
            deleted = self.run_command(
                f"delete-app-{name}",
                ["databricks", "apps", "delete", name, "--profile", self.args.profile],
                timeout=600,
                check=False,
            )
            item["delete_status"] = "pass" if deleted.returncode == 0 else "fail"
            item["absence_verified"] = self._wait_app_absent(name)
        self._write_evidence()

    def _wait_app_absent(self, name: str) -> bool:
        started = time.monotonic()
        while time.monotonic() - started < 600:
            result = self.run_command(
                f"verify-app-absent-{name}",
                ["databricks", "apps", "get", name, "--profile", self.args.profile, "-o", "json"],
                check=False,
            )
            if result.returncode != 0:
                return _classify_cli_absence(result.returncode, result.stdout, result.stderr)
            time.sleep(10)
        return False

    def finalize_local_proofs(self) -> None:
        after = self.run_command(
            "git-status-after",
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.repo,
        ).stdout
        unexpected = _unexpected_repo_status(after)
        self.clean_after = not unexpected
        marker_paths = _find_marker_paths(self.repo, self.run_token)
        instrumentation_absent = not marker_paths
        self._record_check(
            "repository_instrumentation_absent",
            self.clean_before and self.clean_after and instrumentation_absent,
            f"full_clean_before={self.clean_before}; full_clean_after={self.clean_after}; "
            f"unexpected={unexpected}; marker_paths={marker_paths}",
        )
        self._record_check(
            "prompt_construction_metadata_only",
            self.eager_body_loads == 0,
            f"standalone prompt probes and pre-invocation logs observed {self.eager_body_loads} loads",
        )
        freshness_ok = all(
            row.get("freshness_observed") is True for row in self.rows
        ) and expected_matrix_cells() == {
            (row["authoring"], row["runtime"], row["prompt_kind"]) for row in self.rows
        }
        self._record_check(
            "freshness_local_and_deployed",
            freshness_ok,
            "unique generated-runtime markers appeared in every dev/deploy row",
        )
        self._collect_artifacts()
        candidate = self._payload(include_redactions=False)
        findings = _independent_secret_findings(candidate)
        for artifact in self.artifacts:
            path = pathlib.Path(artifact["path"])
            if artifact["kind"] != "wheel" and path.is_file():
                findings.extend(
                    f"artifact:{path}:{finding}"
                    for finding in _independent_secret_findings(
                        path.read_text(encoding="utf-8", errors="replace")
                    )
                )
        self._record_check(
            "all_steps_redacted", not findings, f"credential-shaped findings={findings}"
        )
        self._write_evidence()

    def _collect_artifacts(self) -> None:
        paths = [self.transcript.path]
        if self.wheel is not None:
            paths.append(self.wheel)
        paths.extend(sorted((self.run_output / "logs").glob("*.log")))
        for case in self.cases:
            paths.extend(
                [
                    case.path / "agent.toml",
                    case.path / "agent" / "mason" / "skill_runtime.py",
                ]
            )
        unique = {path.resolve() for path in paths if path.is_file()}
        self.artifacts = [
            {
                "kind": "wheel" if self.wheel is not None and path == self.wheel else "proof",
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in sorted(unique)
        ]

    def _payload(self, *, include_redactions: bool = True) -> dict[str, Any]:
        passed = sum(row.get("status") == "pass" for row in self.rows)
        wheel_path = str(self.wheel) if self.wheel is not None else None
        wheel_hash = (
            _sha256(self.wheel) if self.wheel is not None and self.wheel.is_file() else None
        )
        document: dict[str, Any] = {
            "schema_version": 1,
            "run_status": self.run_status,
            "failures": self.failures,
            "run_id": self.run_id,
            "generated_at": _now(),
            "environment": {
                "profile": self.args.profile,
                "app_auth_profile": self.args.app_auth_profile,
                "host_class": "databricks_workspace",
                "python": platform.python_version(),
                "platform": platform.system().lower(),
            },
            "source": {
                "git_commit": self.git_commit,
                "template_ref_commit": self.template_ref_commit,
                "git_status_clean_before": self.clean_before,
                "git_status_clean_after": self.clean_after,
                "wheel": {"path": wheel_path, "sha256": wheel_hash},
            },
            "resources": {"apps": self.apps},
            "commands": self.commands,
            "validation_checks": self.checks,
            "rows": self.rows,
            "cleanup": self.cleanup_evidence,
            "metrics": {
                "eager_body_loads": self.eager_body_loads,
                "passed_rows": passed,
                "failed_rows": len(self.rows) - passed,
                "total_rows": len(self.rows),
            },
            "artifacts": self.artifacts,
        }
        if include_redactions:
            findings = _independent_secret_findings(document)
            document["redactions"] = {
                "scan_status": "pass" if not findings else "fail",
                "findings": findings,
                "credentials_recorded": bool(findings),
            }
        return document

    def _write_evidence(self) -> None:
        target = self.output / "evidence.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._payload(), indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, target)

    def execute(self) -> int:
        self._write_evidence()
        self.bootstrap()
        self.create_projects()
        self.probe_capabilities()
        for case in self.cases:
            self.run_dev(case)
        for case in self.cases:
            self.deploy(case)
        self.cleanup()
        self.finalize_local_proofs()
        self.run_status = "passed"
        self._write_evidence()
        result = verify_evidence(self.output / "evidence.json")
        if result:
            self.record_failure("verification", "canonical evidence verifier failed")
            self._write_evidence()
        return result


def _activation_events(text: str, prefix: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for match in _ACTIVATION.finditer(text):
        if match.group("prefix") != prefix:
            continue
        event = {"op": match.group("op"), "skill_id": match.group("skill")}
        if match.group("path"):
            event["path"] = match.group("path")
        events.append(event)
    return events


def _last_lines(path: pathlib.Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _last_nonempty(path: pathlib.Path) -> str:
    for line in reversed(_last_lines(path, 20).splitlines()):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="df1")
    parser.add_argument("--app-auth-profile", default="df1-oauth-mcp")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--template-repo")
    parser.add_argument("--template-ref")
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument("--verify-evidence", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.verify_evidence is None:
        if args.output is None:
            parser.error("--output is required unless --verify-evidence is used")
        if not args.template_repo or not args.template_ref:
            parser.error("--template-repo and --template-ref are required for source provenance")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_evidence is not None:
        return verify_evidence(args.verify_evidence)
    runner = Runner(args)
    try:
        return runner.execute()
    except Exception as exc:
        progress("ERROR", "skill-matrix", str(exc))
        runner.record_failure("setup_or_execution", exc)
        runner.transcript.write(f"fatal | {type(exc).__name__} | {exc}")
        runner._write_evidence()
        return 1


if __name__ == "__main__":
    sys.exit(main())
