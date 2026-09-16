"""Live fixture: real Mason durability, deterministic echo, no model or other stores."""

import os

import uvicorn
from sqlalchemy import text

from databricks_mason import AgentApp, DurableAgentContext
from databricks_mason.runtime.durability.store import (
    RUNTIME_DATABASE_ENV,
    RUNTIME_SCHEMA_ENV,
    RUNTIME_USERNAME_ENV,
    LakebaseDurabilityStore,
    default_durability_store,
)
from databricks_mason.runtime.durability.types import JsonValue

configured_store = default_durability_store()
if not isinstance(configured_store, LakebaseDurabilityStore):
    raise RuntimeError(
        "Deploy this fixture with Mason: the test requires a Lakebase Runtime Store."
    )
store: LakebaseDurabilityStore = configured_store
app = AgentApp(durable_runtime=True, durability_store=store)


@app.invoke
@app.on_recovery
async def invoke(value: JsonValue, context: DurableAgentContext) -> JsonValue:
    await context.emit({"marker": "runtime-store-e2e", "input": value})
    return {"echo": value, "invocation_id": context.invocation_id}


@app.get("/api/runtime-store-proof")
async def proof() -> dict:
    """Return database/schema/table ownership as the app SP; never return credentials."""
    async with store._engine.connect() as connection:
        identity = (
            (
                await connection.execute(
                    text(
                        "SELECT current_user AS role, current_database() AS database, "
                        "pg_get_userbyid(datdba) AS database_owner, "
                        "has_database_privilege(current_user, current_database(), 'CREATE') AS can_create "
                        "FROM pg_database WHERE datname = current_database()"
                    )
                )
            )
            .mappings()
            .one()
        )
        schema_owner = (
            await connection.execute(
                text("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = :schema"),
                {"schema": os.environ[RUNTIME_SCHEMA_ENV]},
            )
        ).scalar_one()
        tables = (
            (
                await connection.execute(
                    text(
                        "SELECT tablename, tableowner FROM pg_tables WHERE schemaname = :schema "
                        "AND tablename IN ('executions', 'execution_events') ORDER BY tablename"
                    ),
                    {"schema": os.environ[RUNTIME_SCHEMA_ENV]},
                )
            )
            .mappings()
            .all()
        )
    assert identity["role"] == identity["database_owner"] == os.environ[RUNTIME_USERNAME_ENV]
    assert identity["database"] == os.environ[RUNTIME_DATABASE_ENV]
    assert identity["can_create"]
    assert schema_owner == identity["role"]
    assert [table["tablename"] for table in tables] == ["execution_events", "executions"]
    assert all(table["tableowner"] == identity["role"] for table in tables)
    return {**dict(identity), "schema_owner": schema_owner, "tables": [dict(row) for row in tables]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["DATABRICKS_APP_PORT"]))
