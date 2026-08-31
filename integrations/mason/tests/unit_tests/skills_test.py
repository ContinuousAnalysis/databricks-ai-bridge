"""Unit tests for project-local skill inspection and exact bindings."""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject, SkillSpec
from databricks_mason.skills import skills


class _Ctx:
    def __init__(self, client=None, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "agent"
    AgentProject.create(project, framework="langgraph").write()
    return project


def _write_skill(
    project: pathlib.Path,
    relative: str = ".claude/skills/project-review",
    *,
    frontmatter: str = "name: project-review\ndescription: Review pull requests safely.",
) -> pathlib.Path:
    path = project / relative
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\n# Instructions\n",
        encoding="utf-8",
    )
    return path


def test_add_custom_validates_frontmatter_and_is_idempotent(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    _write_skill(project)
    args = [
        "add",
        "custom",
        ".claude/skills/project-review",
        "--source",
        str(project),
    ]
    runner = CliRunner()

    first = runner.invoke(skills, args, obj=_Ctx(output="json"))
    second = runner.invoke(skills, args, obj=_Ctx(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output)["skill"] == {
        "id": "project-review",
        "kind": "local",
        "source": ".claude/skills/project-review",
    }
    assert json.loads(second.output)["changed"] is False
    loaded = AgentProject.load(project)
    assert len(loaded.skills) == 1
    assert loaded.skills[0].source.path == ".claude/skills/project-review"


def test_list_custom_skills_from_manifest_without_client(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    _write_skill(project)
    loaded = AgentProject.load(project)
    loaded.add_skill(SkillSpec.local("project-review", path=".claude/skills/project-review"))
    loaded.write()

    result = CliRunner().invoke(
        skills,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "schema_version": 1,
        "skills": [
            {
                "id": "project-review",
                "kind": "local",
                "source": ".claude/skills/project-review",
                "description": "Review pull requests safely.",
            }
        ],
    }


def test_custom_skills_help_has_no_uc_surface():
    runner = CliRunner()

    list_help = runner.invoke(skills, ["list", "--help"])
    add_help = runner.invoke(skills, ["add", "--help"])

    assert list_help.exit_code == 0, list_help.output
    assert "--source" in list_help.output
    assert "--schema" not in list_help.output
    assert add_help.exit_code == 0, add_help.output
    assert "custom" in add_help.output
    assert "uc" not in add_help.output.split("Commands:", 1)[-1].split()


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("description: Review pull requests safely.", "name"),
        ("name: Project_Review\ndescription: Review pull requests safely.", "name"),
        (f"name: {'a' * 65}\ndescription: Too long.", "name"),
        ("name: 42\ndescription: Review pull requests safely.", "name"),
        ("name: project-review", "description"),
        ("name: project-review\ndescription: '   '", "description"),
        ("name: project-review\ndescription: 42", "description"),
        ("- not\n- a\n- mapping", "frontmatter"),
        ("name: [unterminated", "frontmatter"),
        ("name: !!python/object/apply:builtins.str [unsafe]", "frontmatter"),
    ],
)
def test_add_custom_rejects_invalid_frontmatter(
    tmp_path: pathlib.Path, frontmatter: str, message: str
):
    project = _project(tmp_path)
    _write_skill(project, frontmatter=frontmatter)

    result = CliRunner().invoke(
        skills,
        [
            "add",
            "custom",
            ".claude/skills/project-review",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert message in result.output.lower()
    assert AgentProject.load(project).skills == []


def test_add_custom_rejects_missing_skill_frontmatter(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    skill = project / "skills" / "project-review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# No frontmatter\n", encoding="utf-8")

    result = CliRunner().invoke(
        skills,
        ["add", "custom", "skills/project-review", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "frontmatter" in result.output.lower()
    assert AgentProject.load(project).skills == []


def test_add_custom_accepts_absolute_contained_path_but_stores_relative_path(
    tmp_path: pathlib.Path,
):
    project = _project(tmp_path)
    skill = _write_skill(project, relative="skills/project-review")

    result = CliRunner().invoke(
        skills,
        ["add", "custom", str(skill), "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).skills[0].source.path == "skills/project-review"


def test_add_custom_requires_directory_leaf_to_match_frontmatter_name(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    _write_skill(
        project,
        relative="skills/review-folder",
        frontmatter="name: project-review\ndescription: Review pull requests safely.",
    )

    result = CliRunner().invoke(
        skills,
        ["add", "custom", "skills/review-folder", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "directory name" in result.output.lower()
    assert "project-review" in result.output
    assert AgentProject.load(project).skills == []


@pytest.mark.parametrize("absolute", [False, True])
def test_add_custom_rejects_path_outside_source(tmp_path: pathlib.Path, absolute: bool):
    project = _project(tmp_path)
    outside = tmp_path / "outside-skill"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: outside-skill\ndescription: Outside.\n---\n",
        encoding="utf-8",
    )
    path = str(outside) if absolute else "../outside-skill"

    result = CliRunner().invoke(
        skills,
        ["add", "custom", path, "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "within the Mason project" in result.output
    assert AgentProject.load(project).skills == []


def test_add_custom_rejects_symlink_that_escapes_source(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    outside = tmp_path / "outside-skill"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: linked-skill\ndescription: Outside.\n---\n",
        encoding="utf-8",
    )
    skill_link = project / "skills" / "linked-skill"
    skill_link.parent.mkdir()
    skill_link.symlink_to(outside, target_is_directory=True)

    result = CliRunner().invoke(
        skills,
        ["add", "custom", "skills/linked-skill", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "within the Mason project" in result.output
    assert AgentProject.load(project).skills == []


def test_add_custom_rejects_skill_md_symlink_that_escapes_source(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    skill = project / "skills" / "project-review"
    skill.mkdir(parents=True)
    outside = tmp_path / "outside-SKILL.md"
    outside.write_text(
        "---\nname: project-review\ndescription: Outside.\n---\n",
        encoding="utf-8",
    )
    (skill / "SKILL.md").symlink_to(outside)

    result = CliRunner().invoke(
        skills,
        ["add", "custom", "skills/project-review", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "SKILL.md must resolve within the Mason project" in result.output
    assert AgentProject.load(project).skills == []
