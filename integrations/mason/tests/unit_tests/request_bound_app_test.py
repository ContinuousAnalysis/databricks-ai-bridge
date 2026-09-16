import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from databricks_mason import AgentApp
from databricks_mason.runtime.auth import AuthError, InvocationAuthPolicy


@pytest.fixture
def deployed(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "auth-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")


def headers(subject="user-a", token="token-sentinel"):
    return {"x-forwarded-user": subject, "x-forwarded-access-token": token}


def make_app(handler):
    app = AgentApp(auth_policy=InvocationAuthPolicy(user_tools=("sandbox",)))
    app.invoke(handler)
    return app


@pytest.mark.asyncio
async def test_missing_auth_and_background_never_execute_or_submit(deployed):
    seen = []

    async def handler(value, context):
        seen.append(value)

    app = make_app(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        body = {"id": str(uuid4()), "input": []}
        assert (await client.post("/api/invocations", json=body)).status_code == 401
        response = await client.post(
            "/api/invocations", json={**body, "background": True}, headers=headers()
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED"
    assert not seen
    assert not app._runtime.runtime_store.states


@pytest.mark.asyncio
async def test_principal_isolated_invocations_replay_without_retaining_auth(deployed):
    contexts = []

    async def handler(value, context):
        contexts.append(context.request_auth)
        return {"actor": context.request_auth.namespace("actor", "shared")}

    app = make_app(handler)
    invocation_id = str(uuid4())
    body = {"id": invocation_id, "input": []}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        first = await client.post("/api/invocations", json=body, headers=headers())
        replay = await client.post(
            "/api/invocations", json=body, headers=headers(token="refreshed")
        )
        other = await client.post("/api/invocations", json=body, headers=headers("user-b"))
        assert first.status_code == replay.status_code == other.status_code == 200
        assert first.json() == replay.json()
        assert first.json()["output"] != other.json()["output"]
        denied = await client.get(f"/api/invocations/{invocation_id}", headers=headers("user-c"))
        assert denied.status_code == 404
    assert len(contexts) == 2
    for auth in contexts:
        with pytest.raises(AuthError):
            auth.client_for("user")
    assert not app._runtime.runtime_store.states
    assert "token-sentinel" not in repr(app._request_execution.store.states)


@pytest.mark.asyncio
async def test_stream_has_events_and_redacted_auth_error(deployed):
    async def handler(value, context):
        await context.emit({"type": "delta", "content": "hello"})
        raise AuthError("MCP_PERMISSION_DENIED", "Permission denied", 403, "sandbox")

    app = make_app(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": str(uuid4()), "input": [], "stream": True},
            headers=headers(),
        )
    assert response.status_code == 200
    assert '"content": "hello"' in response.text
    assert "MCP_PERMISSION_DENIED" in response.text
    assert "token-sentinel" not in response.text


@pytest.mark.asyncio
async def test_cancelled_request_closes_auth_and_handler(deployed):
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    contexts = []

    async def handler(value, context):
        contexts.append(context.request_auth)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    app = make_app(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/api/invocations", json={"id": str(uuid4()), "input": []}, headers=headers()
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cleaned.is_set()
    with pytest.raises(AuthError):
        contexts[0].client_for("user")


@pytest.mark.asyncio
async def test_deadline_closes_auth_and_returns_safe_failure(deployed):
    contexts = []

    async def handler(value, context):
        contexts.append(context.request_auth)
        await asyncio.Event().wait()

    app = make_app(handler)
    app._request_execution.timeout_seconds = 0.01
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post(
            "/api/invocations", json={"id": str(uuid4()), "input": []}, headers=headers()
        )
    assert response.status_code == 500
    with pytest.raises(AuthError):
        contexts[0].client_for("user")


@pytest.mark.asyncio
async def test_exception_credentials_never_enter_errors_or_events(deployed):
    async def handler(value, context):
        raise RuntimeError("SDK error containing token-sentinel")

    app = make_app(handler)
    invocation_id = str(uuid4())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post(
            "/api/invocations", json={"id": invocation_id, "input": []}, headers=headers()
        )
        events = await client.get(f"/api/invocations/{invocation_id}/events", headers=headers())
        denied = await client.get(
            f"/api/invocations/{invocation_id}/events", headers=headers("user-b")
        )
    assert response.status_code == 500
    assert denied.status_code == 404
    assert "token-sentinel" not in response.text + events.text
    assert "token-sentinel" not in repr(app._request_execution._errors)


@pytest.mark.asyncio
async def test_concurrent_callers_keep_separate_auth_and_sessions(deployed):
    contexts = []
    both_entered = asyncio.Event()

    async def handler(value, context):
        contexts.append(context)
        if len(contexts) == 2:
            both_entered.set()
        await both_entered.wait()
        return context.session_id

    app = make_app(handler)
    body = {"id": str(uuid4()), "input": []}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        first, second = await asyncio.wait_for(
            asyncio.gather(
                client.post("/api/invocations", json=body, headers=headers("user-a")),
                client.post("/api/invocations", json=body, headers=headers("user-b")),
            ),
            2,
        )
    assert first.status_code == second.status_code == 200
    assert first.json()["output"] != second.json()["output"]
    assert contexts[0].request_auth is not contexts[1].request_auth


@pytest.mark.asyncio
async def test_active_duplicate_and_changed_payload_conflict(deployed):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(value, context):
        entered.set()
        await release.wait()
        return value

    app = make_app(handler)
    body = {"id": str(uuid4()), "input": []}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        running = asyncio.create_task(client.post("/api/invocations", json=body, headers=headers()))
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = await client.post("/api/invocations", json=body, headers=headers())
        assert duplicate.status_code == 409
        release.set()
        assert (await running).status_code == 200
        conflict = await client.post(
            "/api/invocations", json={**body, "input": "changed"}, headers=headers()
        )
        assert conflict.status_code == 409


def test_user_policy_rejects_recovery_and_explicit_store(deployed):
    from databricks_mason.runtime.store import InMemoryRuntimeStore

    with pytest.raises(ValueError, match="Runtime Store"):
        AgentApp(
            auth_policy=InvocationAuthPolicy(("sandbox",)), runtime_store=InMemoryRuntimeStore()
        )
    app = make_app(None)
    with pytest.raises(ValueError, match="recovery"):
        app.recover(lambda value, context: value)


@pytest.mark.asyncio
async def test_stream_disconnect_cancels_handler_and_closes_auth(deployed):
    cleaned = asyncio.Event()
    contexts = []

    async def handler(value, context):
        contexts.append(context.request_auth)
        try:
            await context.emit({"type": "delta", "content": "disconnect now"})
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    app = make_app(handler)
    incoming = asyncio.Queue()
    await incoming.put(
        {
            "type": "http.request",
            "body": json.dumps(
                {
                    "id": str(uuid4()),
                    "input": [],
                    "stream": True,
                }
            ).encode(),
            "more_body": False,
        }
    )

    async def send(message):
        if b"disconnect now" in message.get("body", b""):
            await incoming.put({"type": "http.disconnect"})

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "path": "/api/invocations",
        "raw_path": b"/api/invocations",
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "https",
        "server": ("test", 443),
        "client": ("test", 1234),
        "headers": [(b"content-type", b"application/json")]
        + [(name.encode(), value.encode()) for name, value in headers().items()],
    }
    await asyncio.wait_for(app(scope, incoming.get, send), 2)
    await asyncio.wait_for(cleaned.wait(), 2)
    with pytest.raises(AuthError):
        contexts[0].client_for("user")
    assert all(state.is_terminal for state in app._request_execution.store.states.values())


@pytest.mark.asyncio
async def test_stream_startup_failure_closes_never_started_execution(deployed, monkeypatch):
    from fastapi import Request
    from starlette.requests import ClientDisconnect

    from databricks_mason.runtime.auth import RequestAuthContext

    auth = RequestAuthContext.from_headers(headers())
    monkeypatch.setattr(RequestAuthContext, "from_headers", lambda headers: auth)

    async def handler(value, context):
        await asyncio.Event().wait()

    app = make_app(handler)
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "headers": []}
    response = await app._request_execution.invoke(
        Request(scope),
        str(uuid4()),
        {"session_id": "shared", "input": []},
        handler,
        background=False,
        stream=True,
    )

    async def fail_send(message):
        raise OSError("connection already gone")

    with pytest.raises(ClientDisconnect):
        await response(scope, asyncio.Queue().get, fail_send)
    assert all(state.is_terminal for state in app._request_execution.store.states.values())
    with pytest.raises(AuthError):
        auth.client_for("user")


@pytest.mark.asyncio
async def test_terminal_retention_evicts_state_events_and_errors(deployed):
    async def handler(value, context):
        raise AuthError("MCP_PERMISSION_DENIED", "Permission denied", 403)

    app = make_app(handler)
    app._request_execution.max_records = 2
    invocation_ids = [str(uuid4()) for _ in range(3)]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        for invocation_id in invocation_ids:
            await client.post("/api/invocations", json={"id": invocation_id}, headers=headers())
        assert (
            await client.get(f"/api/invocations/{invocation_ids[0]}", headers=headers())
        ).status_code == 404
        latest = await client.get(
            f"/api/invocations/{invocation_ids[-1]}/events", headers=headers()
        )
        assert "MCP_PERMISSION_DENIED" in latest.text
    execution = app._request_execution
    assert len(execution.store.states) == len(execution._errors) == 2
    assert all(
        event.invocation_id in execution.store.states for event in execution.store.persisted_events
    )
    sequences = [event.sequence_number for event in execution.store.persisted_events]
    assert len(set(sequences)) == len(sequences)


@pytest.mark.asyncio
async def test_terminal_ttl_and_event_limit(deployed):
    async def handler(value, context):
        for index in range(4):
            await context.emit({"type": "delta", "index": index})

    app = make_app(handler)
    app._request_execution.max_events = 2
    invocation_id = str(uuid4())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post(
            "/api/invocations", json={"id": invocation_id}, headers=headers()
        )
        assert response.json()["error"]["code"] == "MCP_USER_EVENT_LIMIT"
        app._request_execution.retention_seconds = 0
        assert (
            await client.get(f"/api/invocations/{invocation_id}", headers=headers())
        ).status_code == 404
    assert not app._request_execution.store.states
    assert not app._request_execution.store.persisted_events
    assert not app._request_execution._errors
