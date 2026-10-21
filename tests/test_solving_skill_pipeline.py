from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import subprocess
import tarfile
import pytest
from harness_dist.skills import bundle_path, write_bundle, inspect_bundle, copy_bundle
from harness_evolution.harnesses.opencode.build import register_built_binary
from harness_evolution.loop.two_sided import _persist_parent_binary
from polar.agent.models import AgentSpec
from terminal_bench_opensandbox.opencode_compat import OpenCodeCompatHarness


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def init_repo(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    git(repo,"init","-q");git(repo,"config","user.email","test@example.invalid");git(repo,"config","user.name","Skill Test")
    (repo/"README").write_text("fixture")
    git(repo,"add",".");git(repo,"commit","-qm","initial")
    return repo


def skill(repo, name, text):
    p=repo/".opencode/skills"/name/"SKILL.md";p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(f"---\nname: {name}\ndescription: A testing skill\n---\n{text}\n")
    return p


def commit(repo):
    git(repo,"add","-A");git(repo,"commit","-qm","skills")
    return git(repo,"rev-parse","HEAD")


def binary(tmp_path, name):
    p=tmp_path/name/"opencode";p.parent.mkdir(parents=True);p.write_text("binary");return p


def test_added_edited_deleted_skills_follow_commits_and_next_worktree(tmp_path):
    repo=init_repo(tmp_path);a=skill(repo,"first","version one");c1=commit(repo)
    b1=binary(tmp_path,"v1");write_bundle(repo,b1,c1)
    a.write_text(a.read_text().replace("version one","version two"));skill(repo,"second","new skill");c2=commit(repo)
    a.write_text("UNCOMMITTED: must not leak")
    b2=binary(tmp_path,"v2");archive=write_bundle(repo,b2,c2)
    with tarfile.open(archive) as tar:
        assert b"version two" in tar.extractfile("skills/first/SKILL.md").read()
        assert "skills/second/SKILL.md" in tar.getnames()
    assert inspect_bundle(archive,commit=c2)["skill_files"]==["skills/first/SKILL.md","skills/second/SKILL.md"]
    git(repo,"checkout","--",".opencode/skills/first/SKILL.md")
    git(repo,"rm","-r",".opencode/skills/first");c3=commit(repo)
    b3=binary(tmp_path,"v3");ar3=write_bundle(repo,b3,c3)
    assert inspect_bundle(ar3)["skill_files"]==["skills/second/SKILL.md"]
    next_tree=tmp_path/"next-candidate";git(repo,"worktree","add","--detach",str(next_tree),c3)
    assert not (next_tree/".opencode/skills/first/SKILL.md").exists()
    assert (next_tree/".opencode/skills/second/SKILL.md").exists()
    git(repo,"rm","-r",".opencode/skills/second");c4=commit(repo)
    b4=binary(tmp_path,"v4");assert inspect_bundle(write_bundle(repo,b4,c4))["skill_files"]==[]
    assert len(inspect_bundle(bundle_path(b1))["skill_files"])==1


def test_registry_and_promoted_parent_keep_exact_skill_bundle(tmp_path):
    repo=init_repo(tmp_path);skill(repo,"first","retained");c=commit(repo)
    src=binary(tmp_path,"dist");config=SimpleNamespace(build={},workspace_root=tmp_path/"work")
    registered=register_built_binary(config,repo,src)
    promoted=Path(_persist_parent_binary(tmp_path/"run",c,str(registered)))
    assert bundle_path(promoted).read_bytes()==bundle_path(registered).read_bytes()
    assert Path(_persist_parent_binary(tmp_path/"run",c,str(registered)))==promoted
    with pytest.raises(ValueError,match="commit mismatch"):
        inspect_bundle(bundle_path(promoted),commit="wrong")


class FakeRuntime:
    def __init__(self, session):self.session_dir=session;self.commands=[];self.uploads=[]
    async def upload_file(self,local,remote):self.uploads.append((local,remote))
    async def exec(self,command,**kwargs):
        self.commands.append(command)
        return SimpleNamespace(return_code=0,stdout="",stderr="")


def test_actual_harness_setup_uploads_bundle_and_configures_discovery(tmp_path,monkeypatch):
    monkeypatch.setenv("HARNESS_TRACE_PROXY_CAPTURE","0")
    repo=init_repo(tmp_path);skill(repo,"first","read me");c=commit(repo);b=binary(tmp_path,"built");archive=write_bundle(repo,b,c)
    h=OpenCodeCompatHarness(AgentSpec(harness="opencode",model_name="openai-compatible/fixture",settings={"opencode_skills_bundle":str(archive),"opencode_skills_commit":c}))
    runtime=FakeRuntime(tmp_path/"session");asyncio.run(h.setup(runtime))
    assert runtime.uploads==[(str(archive),"/tmp/polar-harness-skills.tar.gz")]
    config_command=next(x for x in runtime.commands if "POLARCFG" in x)
    config=json.loads(config_command.split("\n",1)[1].rsplit("\nPOLARCFG",1)[0])
    assert config["skills"]["paths"]==["/tmp/polar-harness-skills/skills"]
    assert any("tar -xzf" in x for x in runtime.commands)
    assert json.loads((runtime.session_dir/"agent/problem-solving-skills.json").read_text())["commit"]==c


def test_missing_required_sidecar_fails_instead_of_silent_skill_loss(tmp_path,monkeypatch):
    monkeypatch.setenv("HARNESS_TRACE_PROXY_CAPTURE","0")
    h=OpenCodeCompatHarness(AgentSpec(harness="opencode",settings={"opencode_skills_bundle":str(tmp_path/"missing"),"opencode_require_skills_bundle":True}))
    with pytest.raises(RuntimeError,match="bundle is missing"):
        asyncio.run(h.setup(FakeRuntime(tmp_path/"session")))


def test_skill_symlinks_rejected(tmp_path):
    repo=init_repo(tmp_path);p=repo/".opencode/skills/link";p.parent.mkdir(parents=True);p.symlink_to("/outside")
    c=commit(repo)
    with pytest.raises(ValueError,match="symlinks"):
        write_bundle(repo,binary(tmp_path,"bad"),c)


def test_skill_changes_are_collected_and_committed_by_real_evolver_paths(tmp_path):
    from harness_evolution.proposer.evolvers.opencode_bun_dev import OpenCodeBunDevEvolver
    from harness_evolution.core.git import GitHarnessManager
    repo=init_repo(tmp_path);a=skill(repo,"change","old");d=skill(repo,"remove","old");base=commit(repo)
    a.write_text(a.read_text().replace("old","updated"));d.unlink();skill(repo,"added","new")
    evolved=object.__new__(OpenCodeBunDevEvolver)
    paths=evolved._list_git_changed_files(repo)
    assert set(paths)=={".opencode/skills/change/SKILL.md",".opencode/skills/remove/SKILL.md",".opencode/skills/added/SKILL.md"}
    manager=GitHarnessManager(workspace_root=str(tmp_path/"workspaces"))
    head=manager.commit_paths(repo,paths,"skill-only candidate")
    assert git(repo,"rev-parse","HEAD")==head and head!=base
    binary_path=binary(tmp_path,"candidate")
    result=inspect_bundle(write_bundle(repo,binary_path,head))
    assert result["skill_files"]==["skills/added/SKILL.md","skills/change/SKILL.md"]


def test_real_gate_agent_spec_carries_registered_bundle_identity(tmp_path):
    from harness_evolution.evaluation.backends.opensandbox import _agent_spec
    repo=init_repo(tmp_path);skill(repo,"first","registered");c=commit(repo)
    config=SimpleNamespace(build={},workspace_root=tmp_path/"work")
    registered=register_built_binary(config,repo,binary(tmp_path,"dist"))
    settings={"agent_model_name":"fixture","context_limit":4096,"output_limit":1024,"agent_gateway_url":"http://fixture"}
    trial=SimpleNamespace(binary=str(registered),commit=c,trial_id="trial-a0")
    spec=_agent_spec(settings,trial)
    assert spec.settings["opencode_skills_bundle"]==str(bundle_path(registered))
    assert spec.settings["opencode_skills_commit"]==c
    assert spec.settings["opencode_require_skills_bundle"] is True
    promoted=_persist_parent_binary(tmp_path/"run",c,str(registered))
    trial.binary=promoted
    parent_spec=_agent_spec(settings,trial)
    assert parent_spec.settings["opencode_require_skills_bundle"] is True
    assert Path(parent_spec.settings["opencode_skills_bundle"]).exists()
