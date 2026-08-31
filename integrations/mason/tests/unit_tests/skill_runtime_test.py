"""Behavior tests for the generated-project skill manifest reader."""

from __future__ import annotations

import asyncio
import importlib
import pathlib
import shutil
import sys
import types

import pytest

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib

from databricks_mason.templates.skill_manifest_runtime import (
    SkillRecord,
    load_skills,
)

UC_RECORD = """[[skills]]
id = "quarter-close"
source = { kind = "uc", name = "main.finance.quarter-close" }
"""
LOCAL_RECORD = """[[skills]]
id = "project-review"
source = { kind = "local", path = ".claude/skills/project-review" }
"""


def _runtime_project(tmp_path: pathlib.Path, *skills: str) -> pathlib.Path:
    project = tmp_path / "langgraph"
    mason = project / "agent" / "mason"
    mason.mkdir(parents=True)
    (project / "agent" / "__init__.py").write_text("", encoding="utf-8")
    (mason / "__init__.py").write_text("", encoding="utf-8")
    _write_manifest(project, *skills)
    template_mason = (
        pathlib.Path(__file__).parents[2] / "templates" / "agent-langgraph" / "agent" / "mason"
    )
    shutil.copyfile(template_mason / "skill_manifest.py", mason / "skill_manifest.py")
    shutil.copyfile(template_mason / "skill_runtime.py", mason / "skill_runtime.py")
    shutil.copyfile(template_mason / "workspace.py", mason / "workspace.py")
    return project


def _write_local_skill(project: pathlib.Path) -> pathlib.Path:
    skill = project / ".claude" / "skills" / "project-review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: project-review
description: Review pull requests safely.
---
LOCAL_SECRET_BODY_MARKER
""",
        encoding="utf-8",
    )
    (skill / "guide.md").write_text("LOCAL_SECRET_FILE_MARKER", encoding="utf-8")
    return skill


def _write_named_local_skill(project: pathlib.Path, name: str, description: str) -> pathlib.Path:
    skill = project / ".claude" / "skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{name} instructions\n",
        encoding="utf-8",
    )
    return skill


def _clear_agent_modules() -> None:
    for name in tuple(sys.modules):
        if name == "agent" or name.startswith("agent."):
            del sys.modules[name]


class _FakeStructuredTool:
    def __init__(self, *, name, description, coroutine):
        self.name = name
        self.description = description
        self.coroutine = coroutine

    @classmethod
    def from_function(cls, *, coroutine, name, description):
        return cls(name=name, description=description, coroutine=coroutine)

    async def ainvoke(self, arguments):
        return await self.coroutine(**arguments)


def _load_runtime(project: pathlib.Path, monkeypatch):
    _clear_agent_modules()
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(project))

    langchain_core = types.ModuleType("langchain_core")
    langchain_tools = types.ModuleType("langchain_core.tools")
    langchain_tools.__dict__["BaseTool"] = object
    langchain_tools.__dict__["StructuredTool"] = _FakeStructuredTool
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", langchain_tools)

    return importlib.import_module("agent.mason.skill_runtime")


def _write_manifest(root: pathlib.Path, *skills: str, framework: str = "langgraph") -> None:
    skill_tables = "\n".join(skills)
    (root / "agent.toml").write_text(
        f'''schema_version = 1

[agent]
framework = "{framework}"

{skill_tables}''',
        encoding="utf-8",
    )


def test_load_skills_preserves_declaration_order(monkeypatch, tmp_path: pathlib.Path):
    second_local = """[[skills]]
id = "release-check"
source = { kind = "local", path = ".claude/skills/release-check" }
"""
    _write_manifest(tmp_path, LOCAL_RECORD, second_local)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    assert load_skills("langgraph") == (
        SkillRecord("project-review", "local", ".claude/skills/project-review"),
        SkillRecord("release-check", "local", ".claude/skills/release-check"),
    )


def test_load_skills_rejects_uc_source(monkeypatch, tmp_path: pathlib.Path):
    _write_manifest(tmp_path, UC_RECORD)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="Unsupported agent.toml skill kind: 'uc'"):
        load_skills("langgraph")


def test_load_skills_rejects_duplicate_ids(monkeypatch, tmp_path: pathlib.Path):
    _write_manifest(tmp_path, LOCAL_RECORD, LOCAL_RECORD)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="skill ids must be unique"):
        load_skills("langgraph")


def test_load_skills_rejects_unsupported_framework(monkeypatch, tmp_path: pathlib.Path):
    _write_manifest(tmp_path, framework="openai")
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="framework"):
        load_skills("langgraph")


def test_load_skills_rejects_equal_unknown_framework(monkeypatch, tmp_path: pathlib.Path):
    _write_manifest(tmp_path, framework="crewai")
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="Unsupported.*framework"):
        load_skills("crewai")


@pytest.mark.parametrize(
    "skill_entry",
    [
        "[[skills]]\nid = 'review'",
        "[[skills]]\nid = 'review'\nsource = 'local'",
        "[[skills]]\nid = 'review'\nsource = { kind = 'unknown' }",
        "[[skills]]\nid = 'review'\nsource = { kind = 'local' }",
        "[[skills]]\nid = 'review'\nsource = { kind = 'local', path = 42 }",
        "[[skills]]\nid = 'review'\nsource = { kind = 'local', path = 'skills/review', name = 'main.ai.review' }",
    ],
)
def test_load_skills_rejects_malformed_sources(
    monkeypatch, tmp_path: pathlib.Path, skill_entry: str
):
    _write_manifest(tmp_path, skill_entry)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="[Ss]kill|source"):
        load_skills("langgraph")


@pytest.mark.parametrize(
    "skill_entry",
    [
        "[[skills]]\nid = 'bad id'\nsource = { kind = 'local', path = 'skills/review' }",
        "[[skills]]\nid = 'review'\nsource = { kind = 'local', path = '../review' }",
        "[[skills]]\nid = 'review'\nsource = { kind = 'local', path = 'skills//review' }",
        "[[skills]]\nid = 'review'\nsource = { kind = 'local', path = 'C:\\\\skills\\\\review' }",
    ],
)
def test_load_skills_rejects_invalid_record_values(
    monkeypatch, tmp_path: pathlib.Path, skill_entry: str
):
    _write_manifest(tmp_path, skill_entry)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="[Ss]kill|path|name"):
        load_skills("langgraph")


def test_load_skills_accepts_sixty_records_and_rejects_sixty_one(
    monkeypatch, tmp_path: pathlib.Path
):
    records = tuple(
        f"[[skills]]\nid = 'skill-{index}'\n"
        f"source = {{ kind = 'local', path = 'skills/skill-{index}' }}"
        for index in range(61)
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    _write_manifest(tmp_path, *records[:60])
    assert len(load_skills("langgraph")) == 60

    _write_manifest(tmp_path, *records)
    with pytest.raises(RuntimeError, match="at most 60 skills"):
        load_skills("langgraph")


def test_packaged_and_generated_skill_runtimes_are_identical():
    package_templates = pathlib.Path(__file__).parents[2] / "src" / "databricks_mason" / "templates"
    generated = (
        pathlib.Path(__file__).parents[2] / "templates" / "agent-langgraph" / "agent" / "mason"
    )

    assert (generated / "skill_manifest.py").read_bytes() == (
        package_templates / "skill_manifest_runtime.py"
    ).read_bytes()
    assert (generated / "skill_runtime.py").read_bytes() == (
        package_templates / "skill_runtime_langgraph.py"
    ).read_bytes()


def test_template_declares_pyyaml_as_a_direct_locked_dependency():
    template = pathlib.Path(__file__).parents[2] / "templates" / "agent-langgraph"
    with (template / "pyproject.toml").open("rb") as input_file:
        project = tomllib.load(input_file)["project"]
    with (template / "uv.lock").open("rb") as input_file:
        lock = tomllib.load(input_file)

    assert "PyYAML>=6.0" in project["dependencies"]
    package = next(item for item in lock["package"] if item["name"] == "agent-langgraph")
    assert {item["name"] for item in package["dependencies"]} >= {"pyyaml"}
    assert {item["name"] for item in package["metadata"]["requires-dist"]} >= {"pyyaml"}


def test_local_skill_injects_metadata_and_loads_body_lazily(tmp_path: pathlib.Path, monkeypatch):
    project = _runtime_project(tmp_path, LOCAL_RECORD)
    _write_local_skill(project)
    runtime = _load_runtime(project, monkeypatch)

    context, tools = asyncio.run(runtime.build_skill_context())

    assert context == (
        "Available skills:\n"
        "- [project-review] (local:.claude/skills/project-review) "
        "Review pull requests safely.\n\n"
        "Call load_skill with an ID when a task matches. "
        "Read referenced files only with read_skill_file."
    )
    assert "LOCAL_SECRET_BODY_MARKER" not in context
    assert "LOCAL_SECRET_FILE_MARKER" not in context
    assert [tool.name for tool in tools] == ["load_skill", "read_skill_file"]
    assert not hasattr(runtime, "_uc_registry")
    assert not hasattr(runtime, "workspace_client")
    load = next(tool for tool in tools if tool.name == "load_skill")
    assert asyncio.run(load.ainvoke({"skill_id": "project-review"})) == (
        "LOCAL_SECRET_BODY_MARKER\n"
    )


def test_local_skill_prompt_preserves_manifest_order(tmp_path: pathlib.Path, monkeypatch):
    zeta = """[[skills]]
id = "zeta-review"
source = { kind = "local", path = ".claude/skills/zeta-review" }
"""
    alpha = """[[skills]]
id = "alpha-review"
source = { kind = "local", path = ".claude/skills/alpha-review" }
"""
    project = _runtime_project(tmp_path, zeta, alpha)
    _write_named_local_skill(project, "zeta-review", "Review last.")
    _write_named_local_skill(project, "alpha-review", "Review first.")
    runtime = _load_runtime(project, monkeypatch)

    context, _ = asyncio.run(runtime.build_skill_context())

    assert context.index("[zeta-review]") < context.index("[alpha-review]")


def test_local_skill_id_must_match_frontmatter_name(tmp_path: pathlib.Path, monkeypatch):
    mismatched_id = """[[skills]]
id = "declared-review"
source = { kind = "local", path = ".claude/skills/project-review" }
"""
    project = _runtime_project(tmp_path, mismatched_id)
    _write_local_skill(project)
    runtime = _load_runtime(project, monkeypatch)

    with pytest.raises(RuntimeError, match="skill id.*must match frontmatter name"):
        asyncio.run(runtime.build_skill_context())


def test_local_skill_reads_only_declared_contained_relative_files(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _runtime_project(tmp_path, LOCAL_RECORD)
    skill = _write_local_skill(project)
    outside = tmp_path / "outside.md"
    outside.write_text("OUTSIDE_SECRET", encoding="utf-8")
    (skill / "escape.md").symlink_to(outside)
    runtime = _load_runtime(project, monkeypatch)
    _, tools = asyncio.run(runtime.build_skill_context())
    read = next(tool for tool in tools if tool.name == "read_skill_file")

    assert (
        asyncio.run(read.ainvoke({"skill_id": "project-review", "path": "guide.md"}))
        == "LOCAL_SECRET_FILE_MARKER"
    )
    for skill_id, path in (
        ("unknown", "guide.md"),
        ("project-review", "../secret"),
        ("project-review", "/tmp/secret"),
        ("project-review", "escape.md"),
    ):
        with pytest.raises(RuntimeError, match="declared skill ID|contained relative path"):
            asyncio.run(read.ainvoke({"skill_id": skill_id, "path": path}))


def test_local_skill_rejects_symlinked_root_and_invalid_frontmatter(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _runtime_project(tmp_path, LOCAL_RECORD)
    outside = tmp_path / "project-review"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: project-review\ndescription: Outside.\n---\nsecret\n", encoding="utf-8"
    )
    link = project / ".claude" / "skills" / "project-review"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    runtime = _load_runtime(project, monkeypatch)
    with pytest.raises(RuntimeError, match="within the project"):
        asyncio.run(runtime.build_skill_context())

    link.unlink()
    link.mkdir()
    (link / "SKILL.md").write_text("not frontmatter", encoding="utf-8")
    with pytest.raises(RuntimeError, match="YAML frontmatter"):
        asyncio.run(runtime.build_skill_context())


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("Project_Review", "standard lowercase-hyphenated name"),
        ("a" * 65, "at most 64"),
        ("another-skill", "directory name.*must match"),
    ],
    ids=["malformed", "overlong", "leaf-mismatch"],
)
def test_local_runtime_revalidates_frontmatter_name_contract(
    tmp_path: pathlib.Path, monkeypatch, name: str, message: str
):
    project = _runtime_project(tmp_path, LOCAL_RECORD)
    skill = _write_local_skill(project)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review safely.\n---\nbody\n", encoding="utf-8"
    )
    runtime = _load_runtime(project, monkeypatch)

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(runtime.build_skill_context())


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"x" * (1024 * 1024 + 1), "1 MiB"),
        (b"\xff", "UTF-8"),
    ],
    ids=["oversize", "non-utf8"],
)
def test_local_skill_file_reads_are_bounded_utf8(
    tmp_path: pathlib.Path, monkeypatch, contents: bytes, message: str
):
    project = _runtime_project(tmp_path, LOCAL_RECORD)
    skill = _write_local_skill(project)
    (skill / "bad.bin").write_bytes(contents)
    runtime = _load_runtime(project, monkeypatch)
    _, tools = asyncio.run(runtime.build_skill_context())
    read = next(tool for tool in tools if tool.name == "read_skill_file")

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(read.ainvoke({"skill_id": "project-review", "path": "bad.bin"}))


def test_local_skill_metadata_does_not_eagerly_read_invalid_body(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _runtime_project(tmp_path, LOCAL_RECORD)
    skill = _write_local_skill(project)
    (skill / "SKILL.md").write_bytes(
        b"---\nname: project-review\ndescription: Review safely.\n---\n\xff"
    )
    runtime = _load_runtime(project, monkeypatch)

    context, tools = asyncio.run(runtime.build_skill_context())

    assert "Review safely." in context
    with pytest.raises(RuntimeError, match="UTF-8"):
        asyncio.run(tools[0].ainvoke({"skill_id": "project-review"}))
