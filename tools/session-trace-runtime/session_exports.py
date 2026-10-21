"""Session-separated exports plus a lossless, deduplicated request graph."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def _rows(path):
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip():
                value=json.loads(line)
                if isinstance(value,dict):
                    yield value


def message_errors(messages):
    errors=[]; seen=set(); pending=set()
    for index,message in enumerate(messages):
        role=message.get("role")
        if role not in {"system","developer","user","assistant","tool"}:
            errors.append(f"message {index}: invalid role")
        if pending and role!="tool":
            errors.append(f"message {index}: missing tool results before next message")
        if role=="tool":
            call=message.get("tool_call_id")
            if call not in pending:
                errors.append(f"message {index}: orphan or repeated tool result")
            pending.discard(call)
        for call in message.get("tool_calls") or []:
            cid=call.get("id"); fn=call.get("function") or {}
            if not isinstance(cid,str) or not cid or not fn.get("name"):
                errors.append(f"message {index}: invalid tool call")
                continue
            if cid in seen:
                errors.append(f"message {index}: duplicate tool call id")
            seen.add(cid);pending.add(cid)
    return errors,sorted(pending)


def export_messages(context_path,dest,meta_dest,assistant_message,legacy):
    context_path=Path(context_path);dest=Path(dest)
    if not context_path.is_file():
        return 0
    ids=[bool((row.get("meta") or {}).get("session_id")) for row in _rows(context_path)]
    strict=os.environ.get("HARNESSRL_REQUIRE_SESSION_EXPORT", "0")=="1"
    if not any(ids):
        if strict:
            raise ValueError("session IDs are required; preserve raw capture")
        return legacy(context_path,dest,meta_dest)
    if not all(ids):
        raise ValueError("mixed identified and unidentified sessions; preserve raw capture")
    sessions={};nodes={};requests=[]
    def node(kind,value):
        encoded=json.dumps([kind,value],ensure_ascii=False,sort_keys=True,separators=(",",":"))
        key=hashlib.sha256(encoded.encode()).hexdigest()
        if key not in nodes:
            nodes[key]={"id":key,"kind":kind,"value":value}
        return key
    for row in _rows(context_path):
        meta=row.get("meta") or {};sid=meta["session_id"];parent=meta.get("parent_session_id") or None
        request=row.get("request") or {};history=request.get("messages") or []
        if not isinstance(history,list) or not all(isinstance(m,dict) for m in history):
            raise ValueError("invalid request messages")
        session=sessions.setdefault(sid,{"session_id":sid,"parent_session_id":parent,"request_count":0,"first_request_n":row.get("n")})
        if session["parent_session_id"]!=parent:
            raise ValueError("conflicting parent session IDs")
        response=row.get("response");assistant=assistant_message(response)
        record={"request_n":row.get("n"),"wire_n":row.get("wire_n"),"session_id":sid,"parent_session_id":parent,
                "message_refs":[node("message",m) for m in history],"tools_ref":node("tools",request.get("tools") or []),
                "settings":{k:v for k,v in request.items() if k not in {"messages","tools"}},
                "response_ref":node("response",response),"response_message_ref":node("message",assistant) if assistant is not None else None,
                "http_status":meta.get("status")}
        requests.append(record);session["request_count"]+=1
        if int(meta.get("status") or 200)<400 or "canonical" not in session:
            session["canonical"]=list(history)+([assistant] if assistant is not None else [])
            session["tools"]=request.get("tools") or []
            session["canonical_request_n"]=row.get("n")
    for sid,session in sessions.items():
        visited={sid};parent=session["parent_session_id"]
        while parent:
            if parent not in sessions:
                raise ValueError("parent session not captured")
            if parent in visited:
                raise ValueError("cyclic session ancestry")
            visited.add(parent);parent=sessions[parent]["parent_session_id"]
    roots=[s for s in sessions.values() if s["parent_session_id"] is None]
    if not roots:
        raise ValueError("no root session captured")
    # Prefer the first root which uses tools over auxiliary title requests.
    roots.sort(key=lambda s:(not bool(s.get("tools")),s["first_request_n"]))
    main=roots[0];main_id=main["session_id"]
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".session-export-",dir=dest.parent) as temporary:
        work=Path(temporary);entries=[]
        for sid,session in sessions.items():
            relative=Path("openai-sessions")/hashlib.sha256(sid.encode()).hexdigest()
            folder=work/relative;folder.mkdir(parents=True)
            messages=session["canonical"]
            errors,pending=message_errors(messages)
            if errors:
                raise ValueError(f"invalid canonical session {sid}: {errors[:3]}")
            (folder/"openai-messages.jsonl").write_text("".join(json.dumps(m,ensure_ascii=False)+"\n" for m in messages),encoding="utf-8")
            _json(folder/"openai-tools.json",session["tools"])
            entries.append({k:v for k,v in session.items() if k not in {"canonical","tools"}}|{
                "messages_path":str(relative/"openai-messages.jsonl"),"tools_path":str(relative/"openai-tools.json"),
                "message_count":len(messages),"pending_tool_call_ids":pending})
        (work/"openai-nodes.jsonl").write_text("".join(json.dumps(n,ensure_ascii=False)+"\n" for n in nodes.values()),encoding="utf-8")
        (work/"openai-requests.jsonl").write_text("".join(json.dumps(v,ensure_ascii=False)+"\n" for v in requests),encoding="utf-8")
        manifest={"schema_version":"openai-sessions-v2","main_session_id":main_id,"main_selection":"first_root_with_tools",
            "request_count":len(requests),"node_count":len(nodes),"nodes_path":"openai-nodes.jsonl","requests_path":"openai-requests.jsonl","sessions":entries}
        main_entry=next(e for e in entries if e["session_id"]==main_id)
        shutil.copy2(work/main_entry["messages_path"],work/dest.name)
        metadata={"schema_version":"openai-messages-v2","layout":"canonical-main-session","source_context":str(context_path),
            "message_count":len(main["canonical"]),"request_count":len(requests),"session_count":len(sessions),"main_session_id":main_id,
            "session_manifest":"openai-sessions.json","requests":[{k:r.get(k) for k in ["request_n","wire_n","session_id","parent_session_id"]} for r in requests]}
        _json(work/"openai-sessions.json",manifest)
        if meta_dest is not None:
            _json(work/Path(meta_dest).name,metadata)
        for file in work.iterdir():
            target=dest.parent/file.name
            if file.is_dir():
                shutil.copytree(file,target,dirs_exist_ok=True)
            else:
                os.replace(file,target)
    return len(main["canonical"])


def validate_exports(agent_dir):
    """Reject incomplete session graphs before intermediate capture is deleted."""
    base=Path(agent_dir).resolve()
    try:
        def safe(relative):
            path=base/relative
            if not path.resolve().is_relative_to(base) or path.is_symlink():
                raise ValueError("unsafe session export path")
            return path
        manifest=json.loads((base/"openai-sessions.json").read_text())
        if manifest["schema_version"]!="openai-sessions-v2":return False
        nodes={}
        for row in _rows(safe(manifest["nodes_path"])):
            key=row["id"]
            encoded=json.dumps([row["kind"],row["value"]],ensure_ascii=False,sort_keys=True,separators=(",",":"))
            if key in nodes or hashlib.sha256(encoded.encode()).hexdigest()!=key:return False
            nodes[key]=row
        sessions={row["session_id"]:row for row in manifest["sessions"]}
        if len(sessions)!=len(manifest["sessions"]):return False
        request_counts={sid:0 for sid in sessions};canonical={};canonical_tools={};count=0;seen_requests=set()
        for request in _rows(safe(manifest["requests_path"])):
            sid=request["session_id"]
            if request["request_n"] in seen_requests:return False
            seen_requests.add(request["request_n"])
            if sid not in sessions or request["parent_session_id"]!=sessions[sid]["parent_session_id"]:return False
            refs=request["message_refs"]
            if any(nodes[ref]["kind"]!="message" for ref in refs):return False
            if nodes[request["tools_ref"]]["kind"]!="tools":return False
            if nodes[request["response_ref"]]["kind"]!="response":return False
            aid=request["response_message_ref"]
            if aid is not None and nodes[aid]["kind"]!="message":return False
            if request["request_n"]==sessions[sid]["canonical_request_n"]:
                canonical[sid]=[nodes[ref]["value"] for ref in refs]+([nodes[aid]["value"]] if aid else [])
                canonical_tools[sid]=nodes[request["tools_ref"]]["value"]
            request_counts[sid]+=1;count+=1
        if count!=manifest["request_count"] or len(nodes)!=manifest["node_count"]:return False
        for sid,session in sessions.items():
            messages=list(_rows(safe(session["messages_path"])))
            if not messages or messages!=canonical[sid] or message_errors(messages)[0]:return False
            if len(messages)!=session["message_count"] or request_counts[sid]!=session["request_count"]:return False
            if json.loads(safe(session["tools_path"]).read_text())!=canonical_tools[sid]:return False
            visited={sid};parent=session["parent_session_id"]
            while parent:
                if parent in visited or parent not in sessions:return False
                visited.add(parent);parent=sessions[parent]["parent_session_id"]
        main=sessions[manifest["main_session_id"]]
        if list(_rows(base/"openai-messages.jsonl"))!=canonical[manifest["main_session_id"]]:return False
        return True
    except (OSError,ValueError,TypeError,KeyError):
        return False


def copy_exports(source,target):
    source=Path(source);target=Path(target)
    for name in ("openai-sessions.json","openai-nodes.jsonl","openai-requests.jsonl"):
        path=source/name
        if path.is_file():shutil.copy2(path,target/name)
    folder=source/"openai-sessions"
    if folder.is_dir():shutil.copytree(folder,target/folder.name,dirs_exist_ok=True)
