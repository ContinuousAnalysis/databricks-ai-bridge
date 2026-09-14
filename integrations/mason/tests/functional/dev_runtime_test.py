"""Hermetic dev CUJ: the scaffolded agent boots on the local durable runtime and answers a real
invocation, with the model served by an in-process fake. No cloud, no auth, no ``run-local``.

`mason init` scaffolds a durable-runtime agent; ``uv sync`` builds its venv; then we boot the agent's
real entrypoint (``start-server`` — the exact command ``run-local`` runs) in a from-scratch
environment whose only Databricks target is the fake serving endpoint, and POST an invocation.

Why this shape:
  * It boots the real ``runtime/main.py`` -> ``AgentApp(durable_runtime=...)`` -> durability-store
    selection: the path #550 broke, where the agent aborted at startup because
    ``DATABRICKS_MASON_RUNTIME_LOCAL`` wasn't injected. The older test only asserted manifest text.
  * The boot env is taken from mason dev's OWN local manifest (``_dev_entry_point``), so if mason dev
    stops injecting the local-runtime marker the booted agent crashes exactly as #550 did — and this
    test fails.
  * It deliberately does NOT go through ``databricks apps run-local``: run-local resolves a real
    workspace at startup (config discovery + SCIM user), so it can't run in an empty environment.
    Booting the entrypoint directly runs the same command run-local would, and stays hermetic.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

_MARKER = "MASON_DEV_OK"


class _FakeServingHandler(BaseHTTPRequestHandler):
    """Minimal Databricks model-serving stand-in: config discovery + a streamed chat completion.

    ChatDatabricks calls ``/serving-endpoints/chat/completions`` with ``stream=True``; we answer with
    OpenAI-shaped SSE whose content is the marker, so the agent's reply is deterministic.
    """

    def log_message(self, *args):  # silence the default per-request logging
        pass

    def _json(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({})  # host/config discovery — an empty doc is enough for a direct boot

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": _MARKER},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_listening(port: int, proc: subprocess.Popen, timeout: float = 90) -> bool:
    """True once ``port`` accepts a connection; False if the process exits first or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _terminate(proc: subprocess.Popen) -> None:
    """Kill the server's whole process group (start-server spawns uvicorn workers)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_dev_cuj_agent_boots_and_answers_on_local_runtime(tmp_path: pathlib.Path) -> None:
    mason = pathlib.Path(sys.executable).with_name("mason")
    uv = shutil.which("uv")
    if not mason.is_file() or uv is None:
        pytest.skip("requires the mason CLI and uv on PATH")

    # 1. Scaffold a durable-runtime langgraph agent (durability is on by default -> the #550 path).
    project = tmp_path / "agent"
    subprocess.run(
        [str(mason), "init", "--framework", "langgraph", str(project)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Neutralize the profile mason init seeds into .env, so the booted agent can only reach the fake.
    (project / ".env").write_text("")

    # 2. Build the agent's venv (installs databricks-mason from the editable pin + runtime deps).
    subprocess.run(
        [uv, "sync"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    start_server = project / ".venv" / "bin" / "start-server"
    assert start_server.is_file(), "uv sync did not install the start-server entrypoint"

    # 3. Boot with mason dev's OWN local manifest env, so the test rides on whatever mason dev
    #    actually injects rather than a hard-coded marker. If a #550-style regression drops the
    #    local-durability signal, this env won't carry it and the durable runtime aborts at startup —
    #    the boot below never listens and the test fails, with the DATABRICKS_MASON_RUNTIME_ENDPOINT
    #    error surfaced in the dumped server log.
    from databricks_mason.dev import _dev_entry_point

    manifest = _dev_entry_point(project / "app.yaml")
    manifest_env = {
        entry["name"]: entry["value"]
        for entry in (yaml.safe_load(manifest.read_text()).get("env") or [])
        if isinstance(entry, dict) and "name" in entry
    }
    manifest.unlink(missing_ok=True)  # don't leave the local-only manifest in the project tree

    # 4. Fake model serving.
    fake = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServingHandler)
    fake_port = fake.server_address[1]
    threading.Thread(target=fake.serve_forever, daemon=True).start()

    # 5. Boot the real entrypoint in a from-scratch environment: only PATH/HOME + project root, the
    #    fake as the sole Databricks target, and mason dev's manifest env. `env=` replaces the whole
    #    environment, so no ambient DATABRICKS_* / profile can leak in.
    #
    #    DATABRICKS_APP_NAME is what makes the local-durability marker load-bearing: without an app
    #    name the durable runtime silently falls back to in-memory, but with one it *requires* either
    #    DATABRICKS_MASON_RUNTIME_LOCAL (dev) or a Lakebase endpoint (deploy) — else it aborts at
    #    startup. `run-local` and real Apps both set it, so setting it here reproduces the exact
    #    condition #550 broke: if mason dev's manifest env stops carrying the local marker, this boot
    #    dies with "DATABRICKS_MASON_RUNTIME_ENDPOINT is required" instead of serving.
    app_port = _free_port()
    home = tmp_path / "home"
    home.mkdir()
    log_path = tmp_path / "server.log"
    boot_env = {
        "PATH": f"{start_server.parent}:/usr/bin:/bin",
        "HOME": str(home),
        "MASON_PROJECT_ROOT": str(project),
        "DATABRICKS_APP_NAME": "mason-dev-cuj",
        "DATABRICKS_HOST": f"http://127.0.0.1:{fake_port}",
        "DATABRICKS_TOKEN": "dummy",
        "PORT": str(app_port),
        **manifest_env,
    }
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [str(start_server)],
            cwd=str(project),
            env=boot_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            assert _wait_until_listening(app_port, server), (
                "agent never started listening — it likely failed to boot (local durability marker "
                f"missing?). Server output:\n{log_path.read_text()}"
            )
            # 6. A durable-runtime foreground invocation must return the model's canned answer.
            request = urllib.request.Request(
                f"http://127.0.0.1:{app_port}/api/invocations",
                data=json.dumps(
                    {
                        "id": "00000000-0000-4000-8000-000000000000",
                        "input": [{"role": "user", "content": "hi"}],
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=40) as response:
                body = json.loads(response.read())
        finally:
            _terminate(server)
            fake.shutdown()

    assert body.get("status") == "completed", body
    assert _MARKER in json.dumps(body["output"]), body
