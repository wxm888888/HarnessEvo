import copy
import json
import os
from pathlib import Path
import pytest
import agent_artifacts as a
from session_exports import validate_exports, message_errors, copy_exports


def call(cid="call_1", name="Read", args='{"path":"a"}'):
    return {"id":cid,"type":"function","function":{"name":name,"arguments":args}}


def assistant(content=None, calls=None):
    m={"role":"assistant","content":content}
    if calls:m["tool_calls"]=calls
    return m


def row(n,sid,history,response,parent=None,tools=None):
    return {"n":n,"wire_n":n,"meta":{"session_id":sid,"parent_session_id":parent,"status":200},
        "request":{"model":"test","messages":history,"tools":tools or [{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"stream":False},
        "response":{"choices":[{"message":response,"finish_reason":"stop"}]}}


def export(tmp_path,rows,monkeypatch):
    monkeypatch.setenv("HARNESSRL_REQUIRE_SESSION_EXPORT","1")
    context=tmp_path/"llm-context.jsonl"
    context.write_text("".join(json.dumps(r)+"\n" for r in rows))
    a.write_openai_messages(context,tmp_path/"openai-messages.jsonl",tmp_path/"openai-messages.meta.json")
    a.write_openai_tools(context,tmp_path/"openai-tools.json")
    a.write_openai_request_meta(context,tmp_path/"openai-request.meta.json")
    return context


def test_interleaved_parent_children_and_lossless_requests(tmp_path,monkeypatch):
    base=[{"role":"system","content":"main"},{"role":"user","content":"fix"}]
    task=assistant(calls=[call("delegate","task",'{"prompt":"explore"}')])
    child=[{"role":"system","content":"explore"},{"role":"user","content":"find files"}]
    normalized=assistant(calls=[call("delegate","task",'{ "prompt": "explore" }')])
    rows=[row(1,"main",base,task),row(2,"child",child,assistant("findings"),"main"),
          row(3,"main",base+[normalized,{"role":"tool","tool_call_id":"delegate","content":"findings"}],assistant("done"))]
    export(tmp_path,rows,monkeypatch)
    assert validate_exports(tmp_path)
    main=[json.loads(l) for l in (tmp_path/"openai-messages.jsonl").read_text().splitlines()]
    assert len([t for m in main for t in m.get("tool_calls",[])])==1
    assert main[-1]["content"]=="done"
    nodes={v["id"]:v for v in map(json.loads,(tmp_path/"openai-nodes.jsonl").read_text().splitlines())}
    graph=list(map(json.loads,(tmp_path/"openai-requests.jsonl").read_text().splitlines()))
    for original,request in zip(rows,graph):
        reconstructed={**request["settings"],"messages":[nodes[k]["value"] for k in request["message_refs"]],"tools":nodes[request["tools_ref"]]["value"]}
        assert reconstructed==original["request"]
        assert nodes[request["response_ref"]]["value"]==original["response"]
    meta=json.loads((tmp_path/"openai-sessions.json").read_text())
    assert {s["session_id"]:s["parent_session_id"] for s in meta["sessions"]}=={"main":None,"child":"main"}
    target=tmp_path/"copied";target.mkdir();copy_exports(tmp_path,target)
    assert (target/"openai-sessions.json").exists()
    assert len(list((target/"openai-sessions").glob("*/openai-messages.jsonl")))==2


def test_same_id_different_sessions_allowed_and_case_change_not_duplicated(tmp_path,monkeypatch):
    base=[{"role":"user","content":"go"}]
    rows=[row(1,"root",base,assistant(calls=[call()])),
          row(2,"child",base,assistant(calls=[call()]),"root"),
          row(3,"root",base+[assistant(calls=[call(name="read",args='{ "path": "a" }')]),{"role":"tool","tool_call_id":"call_1","content":"data"}],assistant("done"))]
    export(tmp_path,rows,monkeypatch)
    assert validate_exports(tmp_path)
    assert not message_errors(list(map(json.loads,(tmp_path/"openai-messages.jsonl").read_text().splitlines())))[0]


def test_compaction_and_retries_preserved_in_graph(tmp_path,monkeypatch):
    rows=[row(1,"root",[{"role":"user","content":"old"}],assistant("old result")),
          row(2,"root",[{"role":"system","content":"summary"},{"role":"user","content":"next"}],assistant("new"))]
    export(tmp_path,rows,monkeypatch)
    assert validate_exports(tmp_path)
    text=(tmp_path/"openai-nodes.jsonl").read_text()
    assert "old result" in text and "summary" in text
    main=(tmp_path/"openai-messages.jsonl").read_text()
    assert "summary" in main and "old result" not in main


@pytest.mark.parametrize("problem",["missing_id","mixed_id","missing_parent","cycle","duplicate_call","orphan_result"])
def test_invalid_input_rejected(tmp_path,monkeypatch,problem):
    rows=[row(1,"root",[{"role":"user","content":"x"}],assistant("ok"))]
    if problem=="missing_id":rows[0]["meta"]["session_id"]=""
    if problem=="mixed_id":rows.append(row(2,"",[],assistant("x")))
    if problem=="missing_parent":rows[0]["meta"]["parent_session_id"]="absent"
    if problem=="cycle":rows[0]["meta"]["parent_session_id"]="root"
    if problem=="duplicate_call":rows[0]["response"]["choices"][0]["message"]=assistant(calls=[call(),call()])
    if problem=="orphan_result":rows[0]["request"]["messages"].append({"role":"tool","tool_call_id":"absent","content":"x"})
    with pytest.raises(ValueError):export(tmp_path,rows,monkeypatch)
    assert not (tmp_path/"openai-messages.jsonl").exists()


def test_tampered_child_or_node_detected(tmp_path,monkeypatch):
    export(tmp_path,[row(1,"root",[{"role":"user","content":"x"}],assistant("ok"))],monkeypatch)
    assert validate_exports(tmp_path)
    manifest=json.loads((tmp_path/"openai-sessions.json").read_text())
    tools=tmp_path/manifest["sessions"][0]["tools_path"]
    original=tools.read_text();tools.write_text("[]")
    assert not validate_exports(tmp_path)
    tools.write_text(original)
    with (tmp_path/"openai-nodes.jsonl").open("a") as output:output.write('{"id":"bad","kind":"message","value":{}}\n')
    assert not validate_exports(tmp_path)


def test_legacy_compatibility_is_explicit(tmp_path,monkeypatch):
    monkeypatch.setenv("HARNESSRL_REQUIRE_SESSION_EXPORT","0")
    source=tmp_path/"context.jsonl";r=row(1,"root",[{"role":"user","content":"x"}],assistant("ok"));r["meta"]={}
    source.write_text(json.dumps(r)+"\n")
    assert a.write_openai_messages(source,tmp_path/"openai-messages.jsonl",tmp_path/"openai-messages.meta.json")==2
    assert json.loads((tmp_path/"openai-messages.meta.json").read_text())["schema_version"]=="openai-messages-v1"


def test_materialize_validates_before_cleaning_and_reuses(tmp_path,monkeypatch):
    from harness_evolution.traces.materialize import materialize
    agent=tmp_path/"agent";agent.mkdir()
    export(agent,[row(1,"root",[{"role":"user","content":"x"}],assistant("ok"))],monkeypatch)
    (agent/"llm-reqs").mkdir();(agent/"llm-reqs/001-resp.sse").write_text("raw")
    assert materialize(tmp_path,reward="0",status="ok")["complete"]
    assert not (agent/"llm-context.jsonl").exists() and not (agent/"llm-reqs").exists()
    assert materialize(tmp_path,reward="0",status="ok")["complete"]


def test_invalid_graph_keeps_raw(tmp_path,monkeypatch):
    from harness_evolution.traces.materialize import materialize
    agent=tmp_path/"agent";agent.mkdir()
    export(agent,[row(1,"root",[{"role":"user","content":"x"}],assistant("ok"))],monkeypatch)
    (agent/"llm-context.jsonl").unlink()
    (agent/"openai-nodes.jsonl").write_text("corrupt\n")
    raw=agent/"llm-reqs";raw.mkdir();(raw/"bad.sse").write_text("raw")
    assert not materialize(tmp_path,reward="0",status="ok")["complete"]
    assert raw.is_dir()
