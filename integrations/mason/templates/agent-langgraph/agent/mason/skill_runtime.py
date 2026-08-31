# ruff: noqa: I001
"""Resolve declared skills into metadata context and two lazy LangGraph tools."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Protocol

import yaml
from agent.mason.skill_manifest import (  # ty: ignore[unresolved-import]
    SkillRecord,
    load_skills,
    project_root,
)
from langchain_core.tools import BaseTool, StructuredTool  # ty: ignore[unresolved-import]

_MAX_CONTENT_BYTES = 1024 * 1024
_BUNDLE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BUNDLE_NAME_MAX_LENGTH = 64


@dataclass(frozen=True)
class SkillDescriptor:
    id: str
    kind: str
    source: str
    name: str
    description: str


class _Provider(Protocol):
    descriptor: SkillDescriptor

    async def load(self) -> str: ...

    async def read_file(self, path: str) -> str: ...


def _read_utf8(path: pathlib.Path) -> str:
    try:
        with path.open("rb") as input_file:
            content = input_file.read(_MAX_CONTENT_BYTES + 1)
    except OSError as exc:
        raise RuntimeError(f"Could not read skill content {path}: {exc}.") from exc
    if len(content) > _MAX_CONTENT_BYTES:
        raise RuntimeError(f"Skill content {path} exceeds the 1 MiB limit.")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Skill content {path} must be UTF-8.") from exc


def _relative_path(value: str) -> pathlib.PurePosixPath:
    windows = pathlib.PureWindowsPath(value)
    parts = re.split(r"[/\\]", value)
    if (
        not value
        or pathlib.PurePosixPath(value).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"Skill file path {value!r} must be a contained relative path.")
    return pathlib.PurePosixPath(*parts)


def _contained(path: pathlib.Path, root: pathlib.Path, description: str) -> pathlib.Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{description} {root}.") from exc
    return resolved


def _frontmatter(path: pathlib.Path) -> tuple[str, str]:
    lines: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as input_file:
            first = input_file.readline(_MAX_CONTENT_BYTES + 1)
            total += len(first)
            if first.strip() != b"---":
                raise RuntimeError(f"Skill {path} must begin with YAML frontmatter.")
            while total <= _MAX_CONTENT_BYTES:
                line = input_file.readline(_MAX_CONTENT_BYTES - total + 1)
                total += len(line)
                if total > _MAX_CONTENT_BYTES:
                    raise RuntimeError(f"Skill {path} YAML frontmatter exceeds the 1 MiB limit.")
                if not line:
                    raise RuntimeError(f"Skill {path} has unterminated YAML frontmatter.")
                if line.strip() == b"---":
                    break
                lines.append(line)
            else:
                raise RuntimeError(f"Skill {path} YAML frontmatter exceeds the 1 MiB limit.")
    except OSError as exc:
        raise RuntimeError(f"Could not read skill metadata {path}: {exc}.") from exc
    try:
        frontmatter = b"".join(lines).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Skill metadata {path} must be UTF-8.") from exc
    try:
        metadata = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Skill {path} has invalid YAML frontmatter: {exc}.") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Skill {path} YAML frontmatter must be a mapping.")
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Skill {path} YAML frontmatter must declare a name.")
    if len(name) > _BUNDLE_NAME_MAX_LENGTH:
        raise RuntimeError(f"Skill {path} YAML frontmatter name must be at most 64 characters.")
    if _BUNDLE_NAME.fullmatch(name) is None:
        raise RuntimeError(
            f"Skill {path} YAML frontmatter must use a standard lowercase-hyphenated name."
        )
    if not isinstance(description, str) or not description.strip():
        raise RuntimeError(f"Skill {path} YAML frontmatter must declare a description.")
    return name, description.strip()


def _instructions(path: pathlib.Path) -> str:
    content = _read_utf8(path)
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise RuntimeError(f"Skill {path} must begin with YAML frontmatter.")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[index + 1 :])
    raise RuntimeError(f"Skill {path} has unterminated YAML frontmatter.")


class _LocalProvider:
    def __init__(self, record: SkillRecord, root: pathlib.Path):
        self.root = root
        manifest = _contained(
            root / "SKILL.md", root, "Local skill SKILL.md must be a contained relative path within"
        )
        name, description = _frontmatter(manifest)
        if root.name != name:
            raise RuntimeError(
                f"Local skill directory name {root.name!r} must match frontmatter name {name!r}."
            )
        if record.id != name:
            raise RuntimeError(
                f"Local skill id {record.id!r} must match frontmatter name {name!r}."
            )
        self.descriptor = SkillDescriptor(
            id=record.id,
            kind="local",
            source=record.path or "",
            name=name,
            description=description,
        )

    async def load(self) -> str:
        manifest = _contained(
            self.root / "SKILL.md",
            self.root,
            "Local skill SKILL.md must be a contained relative path within",
        )
        return _instructions(manifest)

    async def read_file(self, path: str) -> str:
        relative = _relative_path(path)
        resolved = _contained(
            self.root.joinpath(*relative.parts),
            self.root,
            "Skill file path must be a contained relative path within",
        )
        return _read_utf8(resolved)


def _tools(providers: tuple[_Provider, ...]) -> list[BaseTool]:
    registry = {provider.descriptor.id: provider for provider in providers}

    def provider_for(skill_id: str) -> _Provider:
        provider = registry.get(skill_id)
        if provider is None:
            raise RuntimeError(f"Unknown or undeclared skill ID {skill_id!r}.")
        return provider

    async def load_skill(skill_id: str) -> str:
        """Load the instructions for one declared skill by its ID."""
        return await provider_for(skill_id).load()

    async def read_skill_file(skill_id: str, path: str) -> str:
        """Read a contained relative file referenced by one declared skill."""
        return await provider_for(skill_id).read_file(path)

    return [
        StructuredTool.from_function(
            coroutine=load_skill,
            name="load_skill",
            description="Load instructions for a declared skill by its ID.",
        ),
        StructuredTool.from_function(
            coroutine=read_skill_file,
            name="read_skill_file",
            description="Read a contained relative file referenced by a declared skill.",
        ),
    ]


async def build_skill_context() -> tuple[str, list[BaseTool]]:
    """Resolve exact skill metadata and return a metadata-only prompt plus lazy tools."""
    records = load_skills(expected_framework="langgraph")
    if not records:
        return "", []

    root = project_root().resolve()
    providers: list[_Provider] = []
    for record in records:
        skill_root = _contained(
            root / (record.path or ""),
            root,
            f"Local skill {record.id!r} must resolve within the project root",
        )
        if not skill_root.is_dir():
            raise RuntimeError(f"Local skill {record.id!r} must resolve to a directory.")
        providers.append(_LocalProvider(record, skill_root))

    immutable = tuple(providers)
    lines = [
        f"- [{descriptor.id}] ({descriptor.kind}:{descriptor.source}) {descriptor.description}"
        for descriptor in (provider.descriptor for provider in immutable)
    ]
    context = (
        "Available skills:\n"
        + "\n".join(lines)
        + "\n\nCall load_skill with an ID when a task matches. "
        "Read referenced files only with read_skill_file."
    )
    return context, _tools(immutable)
