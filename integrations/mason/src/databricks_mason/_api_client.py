"""Internal authenticated REST client for the Mason memory and session APIs.

Wraps a databricks-sdk `WorkspaceClient` so auth/host come from a `.databrickscfg`
profile. Each method maps to one API operation.
Deployment is handled separately (deploy.py) since it wraps the `databricks apps` CLI.
"""

from __future__ import annotations

from typing import Any, Optional

from databricks.sdk import WorkspaceClient

from databricks_mason.errors import AgentCliError, wrap_api_error

_BASE = "/api/2.0/agents"


def _query(**kwargs: Any) -> dict[str, Any]:
    """Build a query dict, dropping None and empty values."""
    return {k: v for k, v in kwargs.items() if v is not None and v != ""}


def _body(**kwargs: Any) -> dict[str, Any]:
    """Build a request body while retaining meaningful empty values."""
    return {k: v for k, v in kwargs.items() if v is not None}


def memory_store_path(name: str) -> str:
    """Normalize a store id or name into the `memory-stores/{id}` resource segment."""
    name = name.strip().strip("/")
    return name if name.startswith("memory-stores/") else f"memory-stores/{name}"


def memory_entry_path(store: str, entry: str) -> str:
    entry = entry.strip().strip("/")
    if entry.startswith("memory-stores/"):
        return entry
    return f"{memory_store_path(store)}/entries/{entry}"


class AgentApiClient:
    """Thin, authenticated wrapper over the Mason REST surface."""

    def __init__(self, profile: Optional[str] = None):
        try:
            self._w = WorkspaceClient(profile=profile)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean CLI error
            raise AgentCliError(
                f"Could not initialize Databricks auth: {exc}",
                hint="Check your profile (`databricks auth login --profile <name>`) "
                "or pass --profile.",
            ) from exc

    @property
    def host(self) -> str:
        return self._w.config.host or "unknown"

    @property
    def current_user(self) -> str:
        """The authenticated user's name (used to derive the app source workspace path)."""
        return str(self._w.current_user.me().user_name or "unknown")

    def _do(
        self, method: str, path: str, *, query: Optional[dict] = None, body: Optional[dict] = None
    ) -> Any:
        try:
            return self._w.api_client.do(method, path, query=query, body=body)
        except Exception as exc:  # noqa: BLE001 - normalized to AgentCliError
            raise wrap_api_error(exc) from exc

    # --- memory stores -------------------------------------------------------

    def create_memory_store(self, display_name: str, description: Optional[str] = None) -> dict:
        body = _body(display_name=display_name, description=description)
        return self._do("POST", f"{_BASE}/memory-stores", body=body)

    def get_memory_store(self, name: str) -> dict:
        return self._do("GET", f"{_BASE}/{memory_store_path(name)}")

    def list_memory_stores(
        self, page_size: Optional[int] = None, page_token: Optional[str] = None
    ) -> dict:
        return self._do(
            "GET",
            f"{_BASE}/memory-stores",
            query=_query(page_size=page_size, page_token=page_token),
        )

    def update_memory_store(
        self, name: str, display_name: Optional[str] = None, description: Optional[str] = None
    ) -> dict:
        body = _body(display_name=display_name, description=description)
        mask = ",".join(body.keys())
        return self._do(
            "PATCH", f"{_BASE}/{memory_store_path(name)}", query=_query(update_mask=mask), body=body
        )

    def delete_memory_store(self, name: str) -> dict:
        return self._do("DELETE", f"{_BASE}/{memory_store_path(name)}")

    # --- memory entries ------------------------------------------------------

    def create_memory_entry(
        self,
        store: str,
        actor_id: str,
        path: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
        session_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> dict:
        body = _body(
            actor_id=actor_id,
            path=path,
            content=content,
            description=description,
            session_id=session_id,
            source_type=source_type,
        )
        return self._do("POST", f"{_BASE}/{memory_store_path(store)}/entries", body=body)

    def get_memory_entry(
        self,
        store: str,
        entry: str,
        read_mask: Optional[str] = None,
    ) -> dict:
        return self._do(
            "GET",
            f"{_BASE}/{memory_entry_path(store, entry)}",
            query=_query(read_mask=read_mask),
        )

    def list_memory_entries(
        self,
        store: str,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        read_mask: Optional[str] = None,
    ) -> dict:
        return self._do(
            "GET",
            f"{_BASE}/{memory_store_path(store)}/entries",
            query=_query(
                actor_id=actor_id,
                path_prefix=path_prefix,
                session_id=session_id,
                page_size=page_size,
                page_token=page_token,
                read_mask=read_mask,
            ),
        )

    def search_memory_entries(
        self,
        store: str,
        actor_id: str,
        query: str,
        page_size: Optional[int] = None,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        read_mask: Optional[str] = None,
    ) -> dict:
        body = _body(
            actor_id=actor_id,
            query=query,
            page_size=page_size,
            path_prefix=path_prefix,
            session_id=session_id,
            read_mask=read_mask,
        )
        return self._do("POST", f"{_BASE}/{memory_store_path(store)}/entries:search", body=body)

    def update_memory_entry(
        self,
        store: str,
        entry: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        body = _body(content=content, description=description)
        return self._do("PATCH", f"{_BASE}/{memory_entry_path(store, entry)}", body=body)

    def delete_memory_entry(self, store: str, entry: str) -> dict:
        return self._do("DELETE", f"{_BASE}/{memory_entry_path(store, entry)}")

    # --- session stores ------------------------------------------------------

    def create_session_store(
        self, name: str, description: Optional[str] = None, metadata: Optional[dict] = None
    ) -> dict:
        body = _body(description=description, metadata=metadata)
        return self._do(
            "POST", f"{_BASE}/session-stores", query={"session_store_name": name}, body=body
        )

    def get_session_store(self, name: str) -> dict:
        return self._do("GET", f"{_BASE}/session-stores/{name}")

    def list_session_stores(
        self, page_size: Optional[int] = None, page_token: Optional[str] = None
    ) -> dict:
        return self._do(
            "GET",
            f"{_BASE}/session-stores",
            query=_query(page_size=page_size, page_token=page_token),
        )

    def update_session_store(
        self, name: str, description: Optional[str] = None, metadata: Optional[dict] = None
    ) -> dict:
        body = _body(description=description, metadata=metadata)
        mask = ",".join(body.keys())
        return self._do(
            "PATCH", f"{_BASE}/session-stores/{name}", query=_query(update_mask=mask), body=body
        )

    def delete_session_store(self, name: str) -> dict:
        return self._do("DELETE", f"{_BASE}/session-stores/{name}")

    # --- sessions ------------------------------------------------------------

    def create_session(
        self,
        store: str,
        actor_id: str,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        body = _body(actor_id=actor_id, parent_session_id=parent_session_id, metadata=metadata)
        return self._do(
            "POST",
            f"{_BASE}/session-stores/{store}/sessions",
            query=_query(session_id=session_id),
            body=body,
        )

    def list_sessions(
        self,
        store: str,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> dict:
        return self._do(
            "GET",
            f"{_BASE}/session-stores/{store}/sessions",
            query=_query(
                filter=filter, order_by=order_by, page_size=page_size, page_token=page_token
            ),
        )

    def get_session(self, session_id: str, store: Optional[str] = None) -> dict:
        if store:
            return self._do("GET", f"{_BASE}/session-stores/{store}/sessions/{session_id}")
        return self._do("GET", f"{_BASE}/sessions/{session_id}")

    def update_session(self, store: str, session_id: str, metadata: dict) -> dict:
        return self._do(
            "PATCH",
            f"{_BASE}/session-stores/{store}/sessions/{session_id}",
            query={"update_mask": "metadata"},
            body=_body(metadata=metadata),
        )

    def delete_session(self, store: str, session_id: str) -> dict:
        return self._do("DELETE", f"{_BASE}/session-stores/{store}/sessions/{session_id}")

    def fork_session(
        self,
        store: str,
        source_session_id: str,
        actor_id: str,
        up_to_item_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        body = _body(
            source_session_id=source_session_id,
            actor_id=actor_id,
            up_to_item_id=up_to_item_id,
            session_id=session_id,
            metadata=metadata,
        )
        return self._do("POST", f"{_BASE}/session-stores/{store}/sessions:fork", body=body)

    # --- session items -------------------------------------------------------

    def list_session_items(
        self,
        store: str,
        session_id: str,
        order_by: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> dict:
        return self._do(
            "GET",
            f"{_BASE}/session-stores/{store}/sessions/{session_id}/items",
            query=_query(order_by=order_by, page_size=page_size, page_token=page_token),
        )

    def append_session_items(self, store: str, session_id: str, items: list[dict]) -> dict:
        body = {"items": [{"data": item} for item in items]}
        return self._do(
            "POST", f"{_BASE}/session-stores/{store}/sessions/{session_id}/items:append", body=body
        )

    def pop_session_item(self, store: str, session_id: str) -> dict:
        return self._do(
            "POST", f"{_BASE}/session-stores/{store}/sessions/{session_id}/items:pop", body={}
        )

    def clear_session_items(self, store: str, session_id: str) -> dict:
        return self._do(
            "POST", f"{_BASE}/session-stores/{store}/sessions/{session_id}/items:clear", body={}
        )
