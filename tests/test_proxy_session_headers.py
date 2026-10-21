import http.server
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
import threading
import urllib.request

import pytest

import agent_artifacts as a
from session_exports import validate_exports


@pytest.fixture(scope="module")
def trace_proxy_binary(tmp_path_factory):
    configured = os.environ.get("HARNESS_TRACE_PROXY_BIN")
    if configured:
        binary = Path(configured)
        assert binary.is_file(), "HARNESS_TRACE_PROXY_BIN must name an existing binary"
        return binary
    compiler = shutil.which("cc") or shutil.which("gcc")
    if sys.platform != "linux" or compiler is None:
        pytest.skip("the C trace-proxy integration test requires Linux and a C compiler")
    binary = tmp_path_factory.mktemp("trace-proxy") / "trace-proxy"
    source = Path(__file__).resolve().parents[1] / "tools/trace-proxy/trace_proxy.c"
    subprocess.run([compiler, "-O2", "-o", str(binary), str(source)], check=True)
    return binary


def test_real_c_proxy_records_only_safe_session_headers(tmp_path,monkeypatch,trace_proxy_binary):
    seen=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(dict(self.headers))
            payload=json.dumps({"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}).encode()
            self.send_response(200);self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env={**os.environ,"TRACE_PROXY_UPSTREAM":f"http://127.0.0.1:{server.server_port}","TRACE_PROXY_OUTDIR":str(tmp_path/"wire"),"TRACE_PROXY_PORT":"0"}
    binary=trace_proxy_binary
    process=subprocess.Popen([str(binary)],env=env,stderr=subprocess.PIPE,text=True)
    try:
        ready=process.stderr.readline();port=int(re.search(r"port=(\d+)",ready).group(1))
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for sid,parent in [("ses_root",None),("ses_child","ses_root"),("ses_root",None)]:
            headers={"Content-Type":"application/json","X-Session-Id":sid,"Authorization":"Bearer PRIVATE-TEST-TOKEN"}
            if parent:headers["x-parent-session-id"]=parent
            payload={"model":"test","stream":False,"messages":[{"role":"user","content":"hello"}],"tools":[]}
            request=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(payload).encode(),headers=headers)
            with opener.open(request,timeout=5) as response:assert response.status==200
        process.terminate();process.wait(timeout=5)
        wire=tmp_path/"wire";metas=[json.loads(f.read_text()) for f in sorted(wire.glob("*-meta.json"))]
        assert [m["session_id"] for m in metas]==["ses_root","ses_child","ses_root"]
        assert [m["parent_session_id"] for m in metas]==["","ses_root",""]
        assert all("PRIVATE-TEST-TOKEN" not in f.read_text() for f in wire.glob("*-meta.json"))
        assert all(next(v for k,v in headers.items() if k.lower()=="authorization")=="Bearer PRIVATE-TEST-TOKEN" for headers in seen)
        context=tmp_path/"llm-context.jsonl";a.write_llm_context(wire,context)
        monkeypatch.setenv("HARNESSRL_REQUIRE_SESSION_EXPORT","1")
        a.write_openai_messages(context,tmp_path/"openai-messages.jsonl",tmp_path/"openai-messages.meta.json")
        a.write_openai_tools(context,tmp_path/"openai-tools.json")
        a.write_openai_request_meta(context,tmp_path/"openai-request.meta.json")
        assert validate_exports(tmp_path)
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=5)
        server.shutdown();server.server_close()
