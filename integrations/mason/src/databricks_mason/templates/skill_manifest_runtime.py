"""Read framework-neutral skill bindings from the project's ``agent.toml``."""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, cast

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib


_MAX_SKILLS = 60
_SKILL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SUPPORTED_FRAMEWORKS = {"langgraph", "openai"}


@dataclass(frozen=True)
class SkillRecord:
    id: str
    kind: str
    path: str | None = None


def project_root() -> pathlib.Path:
    """Resolve the project containing ``agent.toml`` without writing it."""
    configured = os.getenv("MASON_PROJECT_ROOT")
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if (root / "agent.toml").is_file():
            return root
        raise RuntimeError(f"MASON_PROJECT_ROOT has no agent.toml: {root}")

    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "agent.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate agent.toml; set MASON_PROJECT_ROOT to the project root.")


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Skill manifest must declare {description}.")
    return value


def _project_relative_path(value: str) -> None:
    if (
        not value
        or pathlib.PurePosixPath(value).is_absolute()
        or pathlib.PureWindowsPath(value).is_absolute()
        or bool(pathlib.PureWindowsPath(value).drive)
        or any(segment in {"", ".", ".."} for segment in re.split(r"[/\\]", value))
    ):
        raise RuntimeError(f"Invalid local skill path {value!r}.")


def _skill(value: object) -> SkillRecord:
    if not isinstance(value, dict):
        raise RuntimeError("Each agent.toml skill must be a table.")
    value = cast(dict[str, Any], value)
    skill_id = _required_string(value.get("id"), "a skill id")
    if not _SKILL_ID.fullmatch(skill_id):
        raise RuntimeError(f"Invalid skill id {skill_id!r}.")
    source = value.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Each agent.toml skill must declare a source table.")
    source = cast(dict[str, Any], source)
    kind = _required_string(source.get("kind"), "a skill source kind")
    if kind != "local":
        raise RuntimeError(f"Unsupported agent.toml skill kind: {kind!r}.")
    if "name" in source:
        raise RuntimeError("Local skills do not accept source.name.")
    path = source.get("path")
    if path is not None and not isinstance(path, str):
        raise RuntimeError("Skill source path must be a string.")

    record = SkillRecord(id=skill_id, kind=kind, path=path)
    if record.path is None:
        raise RuntimeError("Local skills require source.path.")
    _project_relative_path(record.path)
    return record


def load_skills(expected_framework: str) -> tuple[SkillRecord, ...]:
    """Load an immutable skill view so direct manifest edits apply immediately."""
    path = project_root() / "agent.toml"
    try:
        with path.open("rb") as input_file:
            document: dict[str, Any] = tomllib.load(input_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc
    if document.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported agent.toml schema in {path}; expected schema_version = 1.")
    agent = document.get("agent")
    actual_framework = agent.get("framework") if isinstance(agent, dict) else None
    if actual_framework not in _SUPPORTED_FRAMEWORKS:
        raise RuntimeError(f"Unsupported agent.toml framework {actual_framework!r}.")
    if actual_framework != expected_framework:
        raise RuntimeError(
            f"agent.toml framework {actual_framework!r} does not match runtime "
            f"{expected_framework!r}."
        )
    raw_skills = document.get("skills", [])
    if not isinstance(raw_skills, list):
        raise RuntimeError("agent.toml skills must be an array of tables.")
    if len(raw_skills) > _MAX_SKILLS:
        raise RuntimeError(f"agent.toml may declare at most {_MAX_SKILLS} skills.")
    skills = tuple(_skill(item) for item in raw_skills)
    ids = [skill.id for skill in skills]
    if len(ids) != len(set(ids)):
        raise RuntimeError("agent.toml skill ids must be unique.")
    return skills
