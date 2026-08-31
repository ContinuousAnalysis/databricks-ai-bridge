"""Inspect and attach exact project-local agent skills."""

from __future__ import annotations

import pathlib
import re
from typing import Any

import click
import yaml

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, SkillSpec
from databricks_mason.errors import AgentCliError

_BUNDLE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BUNDLE_NAME_MAX_LENGTH = 64


def _skill_source(spec: SkillSpec) -> str:
    return spec.source.path or spec.source.kind


def _manifest_record(spec: SkillSpec) -> dict[str, str]:
    return {
        "id": spec.id,
        "kind": spec.source.kind,
        "source": _skill_source(spec),
    }


def _emit_change(
    obj: Any,
    project: AgentProject,
    spec: SkillSpec,
    changed_files: list[pathlib.Path],
) -> None:
    payload = {
        "schema_version": 1,
        "changed": bool(changed_files),
        "changed_files": [str(path) for path in changed_files],
        "skill": _manifest_record(spec),
    }
    if getattr(obj, "output", "text") == "json":
        render.emit_json(payload)
        return
    if changed_files:
        render.success(
            f"Added {spec.id}",
            fields={"Kind": spec.source.kind, "Manifest": str(project.path)},
        )
    else:
        click.echo(f"Skill {spec.id!r} is already configured in {project.path}")


def _write_spec(obj: Any, project: AgentProject, spec: SkillSpec) -> None:
    changed = project.add_skill(spec)
    changed_files = [project.write()] if changed else []
    _emit_change(obj, project, spec, changed_files)


def _add_spec(obj: Any, source: pathlib.Path, spec: SkillSpec) -> None:
    _write_spec(obj, AgentProject.load(source), spec)


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Mason agent project containing agent.toml.",
    )(function)


def _contained_skill_dir(project: AgentProject, path: pathlib.Path) -> tuple[pathlib.Path, str]:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = project.root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(project.root)
    except (OSError, ValueError) as exc:
        raise AgentCliError(
            f"Path must resolve within the Mason project: {path} (project: {project.root})."
        ) from exc
    if not resolved.is_dir():
        raise AgentCliError(f"Custom skill path {path} must be a directory containing SKILL.md.")
    return resolved, relative.as_posix()


def _frontmatter(skill_md: pathlib.Path) -> dict[str, Any]:
    try:
        content = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AgentCliError(f"Could not read custom skill manifest {skill_md}: {exc}.") from exc
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AgentCliError(f"Custom skill {skill_md} must begin with YAML frontmatter.")
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None
    )
    if closing is None:
        raise AgentCliError(f"Custom skill {skill_md} has unterminated YAML frontmatter.")
    try:
        metadata = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise AgentCliError(
            f"Custom skill {skill_md} has invalid YAML frontmatter: {exc}."
        ) from exc
    if not isinstance(metadata, dict):
        raise AgentCliError(f"Custom skill {skill_md} frontmatter must be a YAML mapping.")
    return metadata


def _custom_skill_metadata(project: AgentProject, skill_dir: pathlib.Path) -> tuple[str, str]:
    skill_md = skill_dir / "SKILL.md"
    try:
        resolved_skill_md = skill_md.resolve(strict=True)
        resolved_skill_md.relative_to(project.root)
    except (OSError, ValueError) as exc:
        raise AgentCliError(f"SKILL.md must resolve within the Mason project: {skill_md}.") from exc
    if not resolved_skill_md.is_file():
        raise AgentCliError(f"Custom skill manifest {skill_md} must be a file.")
    metadata = _frontmatter(resolved_skill_md)
    name = metadata.get("name")
    if (
        not isinstance(name, str)
        or len(name) > _BUNDLE_NAME_MAX_LENGTH
        or _BUNDLE_NAME.fullmatch(name) is None
    ):
        raise AgentCliError(
            "SKILL.md frontmatter name must be at most 64 characters and contain only "
            "lowercase alphanumeric segments separated by single hyphens."
        )
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise AgentCliError("SKILL.md frontmatter description must be a nonblank string.")
    if skill_dir.name != name:
        raise AgentCliError(
            f"Custom skill directory name {skill_dir.name!r} must match frontmatter name {name!r}."
        )
    return name, description.strip()


def _validate_custom_skill(project: AgentProject, skill_dir: pathlib.Path) -> str:
    return _custom_skill_metadata(project, skill_dir)[0]


@click.group()
def skills() -> None:
    """Inspect and attach project-local agent skills."""


@skills.command("list")
@_source_option
@click.pass_obj
def list_skills(obj: Any, source: pathlib.Path) -> None:
    """List the custom skills configured in agent.toml."""
    project = AgentProject.load(source)
    records: list[dict[str, str]] = []
    for spec in project.skills:
        skill_dir, relative = _contained_skill_dir(project, pathlib.Path(_skill_source(spec)))
        name, description = _custom_skill_metadata(project, skill_dir)
        if name != spec.id:
            raise AgentCliError(
                f"Configured skill id {spec.id!r} must match SKILL.md name {name!r}."
            )
        records.append(
            {"id": spec.id, "kind": "local", "source": relative, "description": description}
        )
    if getattr(obj, "output", "text") == "json":
        render.emit_json({"schema_version": 1, "skills": records})
        return
    render.resource_table(
        "Configured agent skills",
        [("Skill", "left"), ("Path", "left"), ("Description", "left")],
        [
            (
                record["id"],
                record["source"],
                record["description"],
            )
            for record in records
        ],
        subtitle=f"Configured in {project.path}",
    )


@skills.group("add")
def add() -> None:
    """Attach an exact skill binding to agent.toml."""


@add.command("custom")
@click.argument("path", type=click.Path(path_type=pathlib.Path))
@_source_option
@click.pass_obj
def add_custom(obj: Any, path: pathlib.Path, source: pathlib.Path) -> None:
    """Attach a project-local Agent Skills directory containing SKILL.md."""
    project = AgentProject.load(source)
    skill_dir, relative = _contained_skill_dir(project, path)
    skill_id = _validate_custom_skill(project, skill_dir)
    _write_spec(obj, project, SkillSpec.local(skill_id, path=relative))
