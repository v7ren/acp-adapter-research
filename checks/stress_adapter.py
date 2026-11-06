"""stress_adapter.py: stability stress test for adapters/aider_adapter.py, local mock only.

Usage: python stress_adapter.py [--only NAME] [--quick]

Reuses check_adapter_client.Adapter (stdio client) and the process helpers by import. The model is a
localhost scripted mock with its own responder (copied from check_adapter_mock.py, extended with
stall, drop, error, big and unicode replies). Every process is started here and stopped by PID.
Writes stress-results.txt next to this file. Exit 1 on any FAIL.
"""
import argparse
import csv
import http.server
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_adapter_client import Adapter, NO_WINDOW, descendants, kill_pid, process_table  # noqa: E402
from check_adapter_mock import PACKS  # noqa: E402

PY = sys.executable  # override with --python; that Python needs Aider installed
ADAPTER = os.path.abspath(os.path.join(HERE, os.pardir, "adapters", "aider_adapter.py"))
RESULTS = os.path.join(HERE, "stress-results.txt")
FAKE_KEY = "fake-key-stress"
FORBIDDEN_PORTS = {17321, 17322, 9333}


# ---------------------------------------------------------------- mock model

def last_user(msgs):
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def respond(text, pack=None):
    pack = pack or PACKS["aider"]
    """Scripted reply for one request. Markers are S-* tokens in the last user message."""
    d = {"pieces": ["ok"], "pause": 0.0, "status": 200}
    m = re.search(r"S-CREATE:(\S+)", text)
    if "S-500" in text:
        d["status"] = 500
    elif "S-429" in text:
        d["status"] = 429
    elif "S-GARBAGE" in text:
        d.update(pieces=[], raw_stream="data: {not json\n\ndata: [DONE]\n\n", raw_plain="not json {")
    elif "S-EMPTY" in text:
        d["pieces"] = []
    elif "S-DROP" in text:
        d.update(pieces=["drop "] * 20, pause=0.05, drop_after=3)
    elif "S-STALL" in text:
        d.update(pieces=["late"], first_delay=60)
    elif "S-LONG" in text:
        d.update(pieces=["word "] * 400, pause=0.05)
    elif "S-BIG" in text:
        d["pieces"] = ["x" * 65536] * 16
    elif "S-LINE" in text:
        d["pieces"] = ["y" * 200000]
    elif "S-UNI" in text:
        d["pieces"] = ["中文 🙂 مرحبا"]
    elif "S-SURR" in text:
        d["pieces"] = ["a\ud800b"]
    elif "S-SHELL-SLEEP" in text:
        d["pieces"] = [pack["shell"]("ping -n 121 127.0.0.1 > nul")]
    elif "S-PY-SLEEP" in text:
        d["pieces"] = ["```python\nimport subprocess, time\n"
                       "subprocess.Popen(['ping', '-n', '121', '127.0.0.1'], creationflags=0x08000000)\n"
                       "time.sleep(120)\n```\n"]
    elif m:
        d["pieces"] = [pack["create"](m.group(1))]
    return d


class Mock:
    def __init__(self, replies="aider"):
        self.pack = PACKS[replies]
        self.requests = []
        self._lock = threading.Lock()
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                size = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    body = json.loads(self.rfile.read(size) or b"{}")
                except ValueError:
                    body = {}
                with mock._lock:
                    mock.requests.append(body)
                try:
                    mock._serve(self, body, respond(last_user(body.get("messages", [])), mock.pack))
                except OSError:
                    pass  # the adapter went away (cancel, kill or stop)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self.server.server_address[1]
        assert port not in FORBIDDEN_PORTS, port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{port}/v1"

    def _serve(self, h, body, d):
        if d.get("first_delay"):
            time.sleep(d["first_delay"])
        if d["status"] != 200:
            err = json.dumps({"error": {"message": "stress fake error", "type": "stress",
                                        "code": str(d["status"])}}).encode("utf-8")
            h.send_response(d["status"])
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(err)))
            h.end_headers()
            h.wfile.write(err)
            return
        if body.get("stream"):
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.send_header("Cache-Control", "no-cache")
            h.end_headers()
            if d.get("raw_stream"):
                h.wfile.write(d["raw_stream"].encode("utf-8"))
                h.wfile.flush()
                return
            for i, piece in enumerate(d["pieces"]):
                if d.get("drop_after") == i:
                    return  # close the connection mid-reply
                delta = {"role": "assistant", "content": piece} if i == 0 else {"content": piece}
                obj = {"id": "chatcmpl-stress", "object": "chat.completion.chunk", "created": 0,
                       "model": body.get("model"),
                       "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                h.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
                h.wfile.flush()
                time.sleep(d.get("pause", 0.0))
            last = {"id": "chatcmpl-stress", "object": "chat.completion.chunk", "created": 0,
                    "model": body.get("model"),
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            h.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode("utf-8"))
            h.wfile.flush()
            return
        if d.get("raw_plain") is not None:
            data = d["raw_plain"].encode("utf-8")
        else:
            data = json.dumps({"id": "chatcmpl-stress", "object": "chat.completion", "created": 0,
                               "model": body.get("model"),
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant",
                                                        "content": "".join(d["pieces"])}}]}).encode("utf-8")
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def count(self):
        with self._lock:
            return len(self.requests)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------- client

class StressClient(Adapter):
    """Adapter client with (a) per-session text, (b) optional holding of permission asks.

    Overrides _read_out so a permission ask can be left unanswered (hold) and answered later.
    """

    def __init__(self, cmd, cwd, env):
        self.hold = False
        self.held = []
        self.texts = {}
        self.known = set()
        self.stopped = False
        self.py_pid = None
        self.work = cwd
        super().__init__(cmd, cwd, env)

    def reset(self):
        with self.cv:
            self.updates = []
            self.texts = {}

    def text_of(self, sid):
        with self.cv:
            return self.texts.get(sid, "")

    def _read_out(self):
        for line in self.p.stdout:
            now = time.monotonic()
            with self.cv:
                self.raw.append(line.rstrip("\r\n"))
            try:
                msg = json.loads(line)
            except ValueError:
                msg = None
            if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
                with self.cv:
                    self.bad_lines.append(line.rstrip()[:200])
                continue
            with self.cv:
                method = msg.get("method")
                params = msg.get("params") or {}
                if method == "session/update":
                    upd = params.get("update") or {}
                    self.updates.append((upd, now))
                    c = upd.get("content")
                    if upd.get("sessionUpdate") == "agent_message_chunk" and isinstance(c, dict):
                        sid = params.get("sessionId")
                        self.texts[sid] = self.texts.get(sid, "") + str(c.get("text", ""))
                elif method == "session/request_permission":
                    self.perms.append(params)
                    if self.hold:
                        self.held.append((msg.get("id"), params))
                    else:
                        self._answer(msg.get("id"), params, self.answer)
                elif "id" in msg and method is None:
                    self.responses[msg["id"]] = msg
                    self.resp_time[msg["id"]] = now
                self.cv.notify_all()

    def _answer(self, rid, params, answer):
        saved = self.answer
        self.answer = answer
        try:
            outcome, chosen = self._choose(params)
        finally:
            self.answer = saved
        self.perm_answers.append(chosen)
        self._send({"jsonrpc": "2.0", "id": rid, "result": {"outcome": outcome}})

    def release(self, answer):
        """Answer every held permission ask now ("allow", "reject" or None for cancelled)."""
        with self.cv:
            held, self.held = self.held, []
            for rid, params in held:
                self._answer(rid, params, answer)

    def send_raw(self, text):
        with self.wlock:
            self.p.stdin.write(text + "\n")
            self.p.stdin.flush()


# ---------------------------------------------------------------- helpers

def ps(script):
    return subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True,
                          text=True, creationflags=NO_WINDOW, timeout=60).stdout


def sample(pid):
    """(working set MB, thread count) for one PID, or None if it is gone."""
    out = ps("$p = Get-Process -Id %d -ErrorAction SilentlyContinue; "
             "if ($p) { $p.WorkingSet64; $p.Threads.Count }" % pid).split()
    if len(out) < 2:
        return None
    return int(out[0]) / 1048576.0, int(out[1])


def alive_names(pids):
    """{pid: image name} for those of these PIDs that are alive. One tasklist call."""
    if not pids:
        return {}
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                         creationflags=NO_WINDOW, timeout=60).stdout
    seen = {}
    for row in csv.reader(io.StringIO(out)):
        if len(row) > 1 and row[1].isdigit():
            seen[int(row[1])] = row[0]
    return {p: seen[p] for p in pids if p in seen}


def describe_pids(pids):
    """'pid:image parent=ppid(image)' for each PID, from one process table and one tasklist call."""
    if not pids:
        return []
    table = process_table()
    parent = {kid: par for par, kids in table.items() for kid in kids}
    names = alive_names(sorted(set(pids) | {parent[p] for p in pids if p in parent}))
    out = []
    for pid in pids:
        par = parent.get(pid)
        out.append(f"{pid}:{names.get(pid, '?')} parent={par}({names.get(par, '?')})")
    return out


def alive_set(pids):
    return set(alive_names(list(pids)))


def stderr_text(ad, n=40):
    with ad.cv:
        return "\n".join(list(ad.err)[-n:])


def wait_until(fn, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.1)
    return fn()


def collected(ad, sid):
    return ad.text_of(sid)


def prompt(ad, sid, text, timeout):
    ad.reset()
    t0 = time.monotonic()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": text}]})
    try:
        r = ad.wait(rid, timeout)
    except TimeoutError:
        # Recover: cancel the stuck turn so it cannot block the next prompt in this session.
        ad.notify("session/cancel", {"sessionId": sid})
        try:
            ad.wait(rid, 10)
            recovered = "cancel ended it"
        except TimeoutError:
            recovered = "cancel did NOT end it"
        return {"ok": False, "why": f"no reply in {timeout}s ({recovered})",
                "lat": time.monotonic() - t0, "text": collected(ad, sid), "rid": rid}
    lat = ad.resp_time.get(rid, time.monotonic()) - t0
    if "error" in r:
        return {"ok": False, "err": r["error"], "lat": lat, "text": collected(ad, sid), "rid": rid}
    return {"ok": True, "stop": (r.get("result") or {}).get("stopReason"), "lat": lat,
            "text": collected(ad, sid), "rid": rid}


def good_turn(r):
    return r["ok"] and r.get("stop") == "end_turn" and "ok" in r["text"].lower()


def describe(r):
    if not r["ok"]:
        if "err" in r:
            return "error " + str(r["err"].get("message", ""))[:90]
        return r.get("why", "failed")
    return f"stop={r.get('stop')} text={r['text'][:40]!r}"


class Lab:
    """Per-scenario bookkeeping: started adapters, PIDs seen, orphans found."""

    def __init__(self, mock, env):
        self.mock = mock
        self.env = env
        self.clients = []
        self.orphans = []
        self.work_dirs = []

    def new_work(self):
        d = tempfile.mkdtemp(prefix="stress_adapter_")
        self.work_dirs.append(d)
        return d

    def start(self, cwd=None):
        cwd = cwd or self.new_work()
        ad = StressClient(f'"{PY}" "{ADAPTER}"', cwd, self.env)
        self.clients.append(ad)
        self.track(ad)
        for _ in range(60):
            kids = process_table().get(ad.p.pid, [])
            if kids:
                ad.py_pid = kids[0]
                break
            time.sleep(0.25)
        self.track(ad)
        rid = ad.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        ad.wait(rid, 120)  # TimeoutError = adapter never answered initialize
        return ad

    def track(self, ad):
        kids = process_table()
        pids = [ad.p.pid] + descendants(ad.p.pid, kids)
        if ad.py_pid:
            pids += [ad.py_pid] + descendants(ad.py_pid, kids)
        ad.known.update(pids)

    def session(self, ad, cwd):
        rid = ad.request("session/new", {"cwd": cwd, "mcpServers": []})
        return ad.wait(rid, 60)["result"]["sessionId"]

    def stop(self, ad):
        if ad.stopped:
            return []
        self.track(ad)
        try:
            ad.p.stdin.close()
        except Exception:
            pass
        end = time.monotonic() + 10
        while ad.p.poll() is None and time.monotonic() < end:
            time.sleep(0.2)
        ad.stopped = True
        left = sorted(alive_set(list(ad.known)))
        notes = describe_pids(left)  # image and parent of each PID still alive when stop began
        for pid in left:
            kill_pid(pid)
        if ad.p.poll() is None:
            kill_pid(ad.p.pid)
        self.orphans += notes
        return notes

    def sweep(self):
        for ad in self.clients:
            self.stop(ad)
        return self.orphans


def exited_within(ad, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if ad.p.poll() is not None:
            return True
        time.sleep(0.25)
    return ad.p.poll() is not None


# ---------------------------------------------------------------- scenarios
# Each returns (status, detail). status is "PASS" or "FAIL".

def sc_soak(lab, quick):
    n = 20 if quick else 200
    pts = [1, n // 2, n] if quick else [1, 100, 200]
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    lat, mem, bad = [], {}, []
    for i in range(1, n + 1):
        r = prompt(ad, sid, f"S-SOAK turn {i}", 60)
        lat.append(r["lat"])
        if not good_turn(r):
            bad.append(f"turn {i}: {describe(r)}")
            break
        if i in pts:
            mem[i] = sample(ad.py_pid)
    lab.stop(ad)
    if bad:
        return "FAIL", "; ".join(bad)
    if any(mem.get(p) is None for p in pts):
        return "FAIL", "could not sample adapter memory/threads"
    m1, t1 = mem[pts[0]]
    mn, tn = mem[pts[-1]]
    ratio = mn / m1 if m1 else 0
    f10 = statistics_mean(lat[:10])
    l10 = statistics_mean(lat[-10:])
    detail = (f"{n} turns, 0 errors; mem MB t1={m1:.0f} t{pts[1]}={mem[pts[1]][0]:.0f} "
              f"t{n}={mn:.0f} (x{ratio:.2f}); threads t1={t1} t{n}={tn}; "
              f"latency first10={f10:.2f}s last10={l10:.2f}s max={max(lat):.2f}s")
    if ratio > 3:
        return "FAIL", "memory grew over 3x: " + detail
    if tn - t1 > 5:
        return "FAIL", "threads grew over 5: " + detail
    return "PASS", detail


def statistics_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sc_cancel_storm(lab, quick):
    n = 10 if quick else 50
    rng = random.Random(7)
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    worst, problems, cancelled = 0.0, [], 0
    for i in range(n):
        ad.reset()
        rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [
            {"type": "text", "text": f"S-LONG storm {i}"}]})
        if not ad.wait_chunk(20):
            problems.append(f"#{i} no stream")
            break
        time.sleep(rng.uniform(0.1, 1.5))
        t_c = time.monotonic()
        ad.notify("session/cancel", {"sessionId": sid})
        try:
            r = ad.wait(rid, 5)
        except TimeoutError:
            problems.append(f"#{i} cancel not answered in 5s")
            break
        lat = ad.resp_time[rid] - t_c
        worst = max(worst, lat)
        if (r.get("result") or {}).get("stopReason") == "cancelled":
            cancelled += 1
        else:
            problems.append(f"#{i} stopReason={(r.get('result') or {}).get('stopReason')}")
        if lat > 5:
            problems.append(f"#{i} cancel took {lat:.1f}s")
        nxt = prompt(ad, sid, f"S-SHORT after storm {i}", 30)
        if not good_turn(nxt):
            problems.append(f"#{i} next prompt: {describe(nxt)}")
            break
    lab.stop(ad)
    detail = f"{cancelled}/{n} cancelled, worst cancel answer {worst:.2f}s"
    if problems:
        return "FAIL", detail + "; " + "; ".join(problems[:3])
    return "PASS", detail


def sc_cancel_stalled(lab, quick):
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "S-STALL wait"}]})
    time.sleep(1.0)
    t_c = time.monotonic()
    ad.notify("session/cancel", {"sessionId": sid})
    try:
        ad.wait(rid, 15)
    except TimeoutError:
        lab.stop(ad)
        return "FAIL", ("cancel ignored: turn still running 15s after cancel while model had not yet "
                        "streamed a token (drain() checks cancel only between pieces, aider_adapter.py:371-380)")
    lat = ad.resp_time[rid] - t_c
    lab.stop(ad)
    if lat <= 5:
        return "PASS", f"cancel ended stalled turn in {lat:.2f}s"
    return "FAIL", f"cancel ended stalled turn only after {lat:.1f}s (limit 5s)"


def sc_cancel_in_permission(lab, quick):
    ad = lab.start()
    ad.hold = True
    sid = lab.session(ad, ad.work)
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [
        {"type": "text", "text": "S-CREATE:perm_cancel.txt make it"}]})
    if not wait_until(lambda: len(ad.perms) >= 1, 60):
        lab.stop(ad)
        return "FAIL", "no permission request arrived"
    t_c = time.monotonic()
    ad.notify("session/cancel", {"sessionId": sid})
    try:
        r = ad.wait(rid, 5)
    except TimeoutError:
        lab.stop(ad)
        return "FAIL", "turn did not end within 5s of cancel while permission was pending"
    lat = ad.resp_time[rid] - t_c
    stop = (r.get("result") or {}).get("stopReason")
    path = os.path.join(ad.work, "perm_cancel.txt")
    created_now = os.path.exists(path)
    ad.hold = False
    ad.release("reject")  # answer the stale ask late
    time.sleep(1.0)
    created_later = os.path.exists(path)
    nxt = prompt(ad, sid, "S-SHORT after permission cancel", 60)
    lab.stop(ad)
    detail = f"stop={stop} in {lat:.2f}s, file created={created_now or created_later}, next={describe(nxt)}"
    if stop == "cancelled" and not created_now and not created_later and good_turn(nxt):
        return "PASS", detail
    return "FAIL", detail


def sc_permission_never_answered(lab, quick):
    hold_s = 10 if quick else 60
    ad = lab.start()
    ad.hold = True
    sid = lab.session(ad, ad.work)
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [
        {"type": "text", "text": "S-CREATE:held_file.txt make it"}]})
    if not wait_until(lambda: len(ad.perms) >= 1, 60):
        lab.stop(ad)
        return "FAIL", "no permission request arrived"
    t0 = time.monotonic()
    while time.monotonic() - t0 < hold_s:
        if ad.p.poll() is not None:
            lab.stop(ad)
            return "FAIL", f"adapter exited after {time.monotonic() - t0:.0f}s with the ask unanswered"
        time.sleep(0.5)
    ad.hold = False
    ad.release("allow")
    try:
        r = ad.wait(rid, 60)
        stop = (r.get("result") or {}).get("stopReason", str(r.get("error")))
    except TimeoutError:
        lab.stop(ad)
        return "FAIL", f"turn did not finish within 60s after the late answer ({hold_s}s hold)"
    nxt = prompt(ad, sid, "S-SHORT after late answer", 60)
    lab.stop(ad)
    detail = f"held {hold_s}s, adapter alive, late answer -> stop={stop}, next={describe(nxt)}"
    return ("PASS" if good_turn(nxt) else "FAIL"), detail


def sc_shell_hang(lab, quick):
    """Allowed shell command sleeps ~120 s. Cancel must end the turn in 10 s and kill the command."""
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    base = set(descendants(ad.p.pid, process_table()))  # the adapter's own processes, before any command
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [
        {"type": "text", "text": "S-SHELL-SLEEP run the command"}]})
    if not wait_until(lambda: any("shell" in json.dumps(p).lower() for p in ad.perms), 30):
        lab.stop(ad)
        return "FAIL", "no shell permission request arrived"
    # Sample the adapter's whole process tree every 0.5 s during the turn; keep every PID seen.
    seen = {}
    t_end = time.monotonic() + 8
    while time.monotonic() < t_end:
        # Root at the shell wrapper that started the adapter. ad.py_pid is only the first child seen at
        # start-up (a launcher stub), so a tree rooted there can miss the command's chain.
        tree = [p for p in descendants(ad.p.pid, process_table()) if p not in base]
        if tree:
            names = alive_names(tree)
            for pid in tree:
                seen.setdefault(pid, names.get(pid, "?"))
        time.sleep(0.5)
    tree_note = ", ".join(f"{pid}:{nm}" for pid, nm in sorted(seen.items())) or "none"
    if not seen:
        err = stderr_text(ad, 15).replace("\n", " | ")[-400:]
        lab.stop(ad)
        answers = list(ad.perm_answers)
        names = alive_names(sorted(ad.known))
        return "FAIL", (f"permission answers={answers}; no child process of the adapter seen in 8s "
                        f"(cause not established); adapter-known pids alive={names}; stderr tail: {err}")
    t_c = time.monotonic()
    ad.notify("session/cancel", {"sessionId": sid})
    try:
        r = ad.wait(rid, 10)
    except TimeoutError:
        left = alive_names(sorted(seen))
        lab.stop(ad)
        return "FAIL", (f"turn did not end within 10s of cancel while sleep ran; tree seen: {tree_note}; "
                        f"still alive: {left}")
    lat = ad.resp_time[rid] - t_c
    time.sleep(1)
    still = alive_names(sorted(seen))
    stop = (r.get("result") or {}).get("stopReason")
    lab.stop(ad)
    detail = f"stop={stop} in {lat:.2f}s; tree seen: {tree_note}; still alive after cancel: {still}"
    if still:
        return "FAIL", detail + " (command not killed)"
    return "PASS", detail


def sc_client_gone(lab, quick):
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    ad.reset()
    ad.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "S-LONG gone"}]})
    if not ad.wait_chunk(20):
        lab.stop(ad)
        return "FAIL", "no stream started"
    lab.track(ad)
    t0 = time.monotonic()
    try:
        ad.p.stdin.close()
    except Exception:
        pass
    def gone():
        py_alive = bool(ad.py_pid) and ad.py_pid in alive_set([ad.py_pid])
        return ad.p.poll() is not None and not py_alive

    end = time.monotonic() + 10
    while time.monotonic() < end and not gone():
        time.sleep(0.5)
    took = time.monotonic() - t0
    exited = gone()
    left = lab.stop(ad)
    detail = f"exited={exited} in {took:.1f}s, orphan pids={left}"
    if not exited:
        return "FAIL", "adapter still running 10s after client closed stdin; " + detail
    return ("FAIL" if left else "PASS"), detail


def abrupt(lab, perm):
    ad = lab.start()
    ad.hold = perm
    sid = lab.session(ad, ad.work)
    ad.reset()
    ad.request("session/prompt", {"sessionId": sid, "prompt": [
        {"type": "text", "text": "S-CREATE:abrupt.txt" if perm else "S-LONG abrupt"}]})
    if perm:
        if not wait_until(lambda: len(ad.perms) >= 1, 60):
            lab.stop(ad)
            return "FAIL", "no permission request arrived"
    elif not ad.wait_chunk(20):
        lab.stop(ad)
        return "FAIL", "no stream started"
    lab.track(ad)
    t0 = time.monotonic()
    stdout_closed = [False]

    def close_stdout():
        try:
            ad.p.stdout.close()
            stdout_closed[0] = True
        except Exception:
            pass

    for f in (ad.p.stdin, ad.p.stderr):
        try:
            f.close()
        except Exception:
            pass
    th = threading.Thread(target=close_stdout, daemon=True)
    th.start()
    th.join(3)
    exited = exited_within(ad, 10)
    took = time.monotonic() - t0
    left = lab.stop(ad)
    return exited, took, left, stdout_closed[0]


def sc_client_killed_abruptly(lab, quick):
    notes, bad = [], []
    for perm in (False, True):
        lab2 = Lab(lab.mock, lab.env)
        try:
            exited, took, left, closed = abrupt(lab2, perm)
        except TimeoutError:
            exited, took, left, closed = False, 0, [], False
        label = "while permission waits" if perm else "mid-stream"
        notes.append(f"{label}: exited={exited} in {took:.1f}s, stdout-close-returned={closed}, orphans={left}")
        if not exited:
            bad.append(f"{label}: still running after 10s")
        if left:
            bad.append(f"{label}: orphans {left}")
        lab.orphans += lab2.orphans
    detail = "; ".join(notes)
    return ("FAIL" if bad else "PASS"), detail


def sc_model_stream_drop(lab, quick):
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    r = prompt(ad, sid, "S-DROP stream", 60)
    nxt = prompt(ad, sid, "S-SHORT after drop", 60)
    lab.stop(ad)
    if not r["ok"] and "no reply" in r.get("why", ""):
        return "FAIL", "drop turn hung: " + describe(r)
    detail = f"drop turn: {describe(r)} in {r['lat']:.1f}s; next: {describe(nxt)}"
    return ("PASS" if good_turn(nxt) else "FAIL"), detail


def sc_model_errors(lab, quick):
    """Each error case on a fresh adapter: time it, check cancel during the wait, check next prompt."""
    notes, bad = [], []
    for marker in ("S-500", "S-429", "S-GARBAGE", "S-EMPTY"):
        ad = lab.start()
        sid = lab.session(ad, ad.work)
        limit = 180 if marker in ("S-500", "S-429") else 60
        ad.reset()
        t0 = time.monotonic()
        rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [
            {"type": "text", "text": f"{marker} case"}]})
        cancel_note = ""
        if marker in ("S-500", "S-429"):
            time.sleep(5)
            tc = time.monotonic()
            ad.notify("session/cancel", {"sessionId": sid})
            try:
                got_resp = ad.wait(rid, limit - 5)
                cancel_note = f"; cancel at 5s ended turn after {time.monotonic() - tc:.1f}s"
            except TimeoutError:
                cancel_note = f"; cancel at 5s did NOT end turn within {limit - 5}s"
                bad.append(f"{marker}: cancel ignored, turn hung {limit}s")
                took = time.monotonic() - t0
                notes.append(f"{marker}: no reply in {limit}s took {took:.1f}s{cancel_note}")
                lab.stop(ad)
                continue
        else:
            try:
                got_resp = ad.wait(rid, limit)
            except TimeoutError:
                took = time.monotonic() - t0
                bad.append(f"{marker}: no reply in {limit}s")
                notes.append(f"{marker}: no reply in {limit}s")
                lab.stop(ad)
                continue
        took = ad.resp_time.get(rid, time.monotonic()) - t0
        resp = got_resp
        r = {"ok": "error" not in resp, "err": resp.get("error"), "lat": took, "text": "",
             "stop": (resp.get("result") or {}).get("stopReason")}
        nxt = prompt(ad, sid, "S-SHORT after error", 60)
        notes.append(f"{marker}: {describe(r)[:70]} took {took:.1f}s{cancel_note}; "
                     f"next={'ok' if good_turn(nxt) else describe(nxt)[:50]}")
        if not good_turn(nxt):
            bad.append(f"{marker}: next prompt failed")
        lab.stop(ad)
    detail = " | ".join(notes)
    return ("FAIL" if bad else "PASS"), (detail + ("; " + "; ".join(bad) if bad else ""))


def sc_big_output(lab, quick):
    """1 MB reply and 200 KB single line. Each on a fresh adapter so a slow turn cannot spill over."""
    bad, notes = [], []
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    big = prompt(ad, sid, "S-BIG reply", 300)
    if not big["ok"] or big.get("stop") != "end_turn":
        bad.append(f"1 MB reply: {describe(big)}")
    elif len(big["text"]) < 1_000_000:
        bad.append(f"1 MB reply truncated to {len(big['text'])} chars")
    notes.append(f"1 MB: {len(big['text'])} chars in {big['lat']:.1f}s")
    lab.stop(ad)
    ad2 = lab.start()
    sid2 = lab.session(ad2, ad2.work)
    line = prompt(ad2, sid2, "S-LINE reply", 300)
    if not line["ok"] or line.get("stop") != "end_turn":
        bad.append(f"200 KB line: {describe(line)}")
    elif len(line["text"]) < 200_000:
        bad.append(f"200 KB line truncated to {len(line['text'])} chars")
    notes.append(f"200 KB line: {len(line['text'])} chars in {line['lat']:.1f}s")
    if ad.bad_lines or ad2.bad_lines:
        bad.append(f"non-JSON-RPC stdout lines: {len(ad.bad_lines) + len(ad2.bad_lines)}")
    lab.stop(ad2)
    detail = "; ".join(notes)
    return ("FAIL" if bad else "PASS"), (detail + ("; " + "; ".join(bad) if bad else ""))


def sc_unicode(lab, quick):
    """Unicode split into separate cases, each on a fresh adapter. Stderr is captured as evidence."""
    results = {}

    def case(label, fn):
        ad = lab.start()
        sid = lab.session(ad, ad.work)
        try:
            ok, detail = fn(ad, sid)
        except TimeoutError as err:
            ok, detail = False, f"timeout {err}"
        results[label] = (ok, detail, stderr_text(ad, 60))
        lab.stop(ad)

    def chinese_emoji_arabic(ad, sid):
        r = prompt(ad, sid, "S-UNI 你好 🙂 مرحبا", 60)
        return r["ok"] and "中文 🙂 مرحبا" in r["text"], describe(r)[:80]

    def unicode_filename(ad, sid):
        name = "测试🙂.txt"
        r = prompt(ad, sid, f"S-CREATE:{name} please create it", 60)
        made = os.path.exists(os.path.join(ad.work, name))
        return made and r.get("stop") == "end_turn", f"file made={made}, {describe(r)[:60]}"

    def surrogate_in_reply(ad, sid):
        r = prompt(ad, sid, "S-SURR reply please", 60)
        return r["ok"] and "err" not in r, describe(r)[:100]

    def surrogate_in_prompt(ad, sid):
        r = prompt(ad, sid, "S-SHORT lone \ud800 surrogate in prompt", 60)
        return r["ok"] and good_turn(r), describe(r)[:100]

    case("chinese_emoji_arabic", chinese_emoji_arabic)
    case("unicode_filename", unicode_filename)
    case("lone_surrogate_in_reply", surrogate_in_reply)
    case("lone_surrogate_in_prompt", surrogate_in_prompt)

    bad, notes = [], []
    for label, (ok, detail, _err) in results.items():
        notes.append(f"{label}: {'PASS' if ok else 'FAIL'} ({detail})")
        if not ok:
            bad.append(label)
    err = results["lone_surrogate_in_reply"][2]
    quote = [ln for ln in err.splitlines() if "Error" in ln][-2:]
    detail = " | ".join(notes)
    if quote:
        detail += " | stderr: " + " / ".join(q.strip()[:200] for q in quote)
    return ("FAIL" if bad else "PASS"), detail


def sc_big_prompt(lab, quick):
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    big = "S-ECHO " + ("abc def " * 125000)  # about 1 MB
    r = prompt(ad, sid, big, 150)
    size = max((len(json.dumps(b)) for b in lab.mock.requests[-3:]), default=0)
    lab.stop(ad)
    detail = f"prompt {len(big)} chars: {describe(r)[:80]} in {r['lat']:.1f}s; mock saw {size} bytes"
    if good_turn(r) and size > 1_000_000:
        return "PASS", detail
    return "FAIL", detail


def sc_bad_input(lab, quick):
    ad = lab.start()
    expect = {"pre1": "err", "u1": "err", "m1": "err", "m2": "err", "huge": "ok", "w1": "err", "n1": "err"}
    ad.send_raw("{not json")
    ad.send_raw('{"jsonrpc":"2.0","id":"pre1","method":"session/prompt","params":{"sessionId":"x","prompt":[]}}')
    ad.send_raw('{"jsonrpc":"2.0","id":"u1","method":"no/such","params":{}}')
    ad.send_raw('{"jsonrpc":"2.0","id":"m1","method":"session/prompt"}')
    ad.send_raw('{"jsonrpc":"2.0","id":"m2","method":"session/set_config_option","params":{}}')
    ad.send_raw('{"jsonrpc":"2.0","id":"' + "h" * 100000 + '","method":"initialize","params":{}}')
    ad.send_raw("[1,2,3]")
    ad.send_raw('{"jsonrpc":"2.0","id":"w1","method":"session/prompt","params":{"sessionId":"wrong","prompt":[{"type":"text","text":"hi"}]}}')
    ad.send_raw('{"jsonrpc":"2.0","id":"n1","method":"session/prompt","params":{"sessionId":null,"prompt":"notalist"}}')
    expect_ids = [k for k in expect if k != "huge"] + ["h" * 100000]
    with ad.cv:
        ad.cv.wait_for(lambda: all(i in ad.responses for i in expect_ids) and None in ad.responses, 20)
        got = dict(ad.responses)
    missing = [k[:12] for k in expect_ids if k not in got]
    if None not in got:
        missing.append("(parse error, id null)")
    wrong = [k[:12] for k in expect_ids if k in got and k != "h" * 100000 and "error" not in got[k]]
    alive = ad.p.poll() is None
    sid = lab.session(ad, ad.work)
    nxt = prompt(ad, sid, "S-SHORT after bad input", 60)
    lab.stop(ad)
    detail = (f"{len(got)} replies; missing={missing}; should-be-error-but-not={wrong}; "
              f"alive={alive}; next={'ok' if good_turn(nxt) else describe(nxt)}")
    bad = missing or wrong or not alive or not good_turn(nxt)
    return ("FAIL" if bad else "PASS"), detail


def sc_two_sessions(lab, quick):
    ad = lab.start()
    a, b = lab.new_work(), lab.new_work()
    sa = lab.session(ad, a)
    sb = lab.session(ad, b)
    bad = []
    ra = prompt(ad, sa, "S-CREATE:fa.txt in A", 60)
    rb = prompt(ad, sb, "S-CREATE:fb.txt in B", 60)
    for label, r in (("A", ra), ("B", rb)):
        if not (r["ok"] and r.get("stop") == "end_turn"):
            bad.append(f"{label} create turn: {describe(r)}")
    if not (os.path.exists(os.path.join(a, "fa.txt")) and not os.path.exists(os.path.join(a, "fb.txt"))):
        bad.append("folder A has wrong files")
    if not (os.path.exists(os.path.join(b, "fb.txt")) and not os.path.exists(os.path.join(b, "fa.txt"))):
        bad.append("folder B has wrong files")
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sa, "prompt": [{"type": "text", "text": "S-LONG in A"}]})
    if not ad.wait_chunk(20):
        bad.append("A never streamed")
    during = prompt(ad, sb, "S-SHORT hello from B during A", 60)
    ad.notify("session/cancel", {"sessionId": sa})
    try:
        ra_stop = (ad.wait(rid, 20).get("result") or {}).get("stopReason")
    except TimeoutError:
        ra_stop = "no reply"
        bad.append("A did not end after cancel")
    if not during["ok"]:
        note = "clear error: " + describe(during)[:80]
    elif good_turn(during):
        note = "B answered ok during A's turn"
    else:
        note = "B reply odd: " + describe(during)
        bad.append("B reply during A not clean")
    if during["ok"] and "word" in during["text"]:
        bad.append("cross-talk: A's stream text reached B")
    lab.stop(ad)
    detail = (f"A create turn stop={ra.get('stop')}, B create turn stop={rb.get('stop')}; during A's turn: {note}; "
              f"A stop={ra_stop}")
    return ("FAIL" if bad else "PASS"), detail + ("; " + "; ".join(bad) if bad else "")


def sc_rapid_restart(lab, quick):
    n = 4 if quick else 20
    work = lab.new_work()
    bad = []
    for i in range(n):
        ad = lab.start(cwd=work)
        sid = lab.session(ad, work)
        r = prompt(ad, sid, f"S-SHORT cycle {i}", 60)
        if not good_turn(r):
            bad.append(f"cycle {i}: {describe(r)}")
        left = lab.stop(ad)
        if left:
            bad.append(f"cycle {i}: orphans {left}")
        if bad:
            break
    leftover = sorted(f for f in os.listdir(work) if not f.startswith(".aider"))
    if leftover:
        bad.append(f"leftover files in work folder: {leftover}")
    detail = f"{n} start/stop cycles, leftover non-Aider files={leftover}"
    return ("FAIL" if bad else "PASS"), detail + ("; " + "; ".join(bad) if bad else "")


def sc_cancel_python(lab, quick):
    """Allowed Python block runs about 120 s and starts a child process. Cancel must end the turn
    within 10 s, leave no process of the block alive, and the next prompt must work."""
    ad = lab.start()
    sid = lab.session(ad, ad.work)
    base = set(descendants(ad.p.pid, process_table()))
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [
        {"type": "text", "text": "S-PY-SLEEP run the code"}]})
    if not wait_until(lambda: any("python" in json.dumps(p).lower() for p in ad.perms), 60):
        lab.stop(ad)
        return "FAIL", "no python permission request arrived"
    seen = {}
    t_end = time.monotonic() + 8
    while time.monotonic() < t_end:
        tree = [p for p in descendants(ad.p.pid, process_table()) if p not in base]
        if tree:
            names = alive_names(tree)
            for pid in tree:
                seen.setdefault(pid, names.get(pid, "?"))
        time.sleep(0.5)
    tree_note = ", ".join(f"{pid}:{nm}" for pid, nm in sorted(seen.items())) or "none"
    if not seen:
        lab.stop(ad)
        return "FAIL", "no child process of the adapter seen in 8s while the python block ran"
    t_c = time.monotonic()
    ad.notify("session/cancel", {"sessionId": sid})
    try:
        r = ad.wait(rid, 10)
    except TimeoutError:
        left = alive_names(sorted(seen))
        lab.stop(ad)
        return "FAIL", f"turn did not end within 10s of cancel; tree seen: {tree_note}; still alive: {left}"
    lat = ad.resp_time[rid] - t_c
    time.sleep(1)
    still = alive_names(sorted(seen))
    stop = (r.get("result") or {}).get("stopReason")
    nxt = prompt(ad, sid, "S-SHORT after python cancel", 60)
    lab.stop(ad)
    detail = (f"stop={stop} in {lat:.2f}s; tree seen: {tree_note}; still alive after cancel: {still}; "
              f"next={describe(nxt)}")
    ok = stop == "cancelled" and lat <= 10 and not still and good_turn(nxt)
    return ("PASS" if ok else "FAIL"), detail


SCENARIOS = [
    ("soak", sc_soak, 600, 240),
    ("cancel_storm", sc_cancel_storm, 420, 180),
    ("cancel_stalled", sc_cancel_stalled, 60, 60),
    ("cancel_in_permission", sc_cancel_in_permission, 120, 120),
    ("permission_never_answered", sc_permission_never_answered, 180, 120),
    ("shell_hang", sc_shell_hang, 120, 120),
    ("client_gone", sc_client_gone, 90, 90),
    ("client_killed_abruptly", sc_client_killed_abruptly, 150, 150),
    ("model_stream_drop", sc_model_stream_drop, 120, 120),
    ("model_errors", sc_model_errors, 600, 420),
    ("big_output", sc_big_output, 700, 420),
    ("unicode", sc_unicode, 150, 150),
    ("big_prompt", sc_big_prompt, 200, 200),
    ("bad_input", sc_bad_input, 120, 120),
    ("two_sessions", sc_two_sessions, 150, 150),
    ("rapid_restart", sc_rapid_restart, 300, 120),
    ("cancel_python", sc_cancel_python, 120, 120),
]


def run_scenario(name, fn, limit, mock, quick):
    lab = Lab(mock, make_env(mock))
    result = {}

    def body():
        try:
            result["v"] = fn(lab, quick)
        except Exception as err:  # a scenario bug is a FAIL with the reason, not a crash
            result["v"] = ("FAIL", f"exception {type(err).__name__}: {str(err)[:150]}")

    t0 = time.monotonic()
    th = threading.Thread(target=body, daemon=True)
    th.start()
    th.join(limit)
    if th.is_alive():
        status, detail = "FAIL", f"hang: no finish within {limit}s"
    else:
        status, detail = result["v"]
    left = lab.sweep()  # stop every adapter this scenario started, kill strays by PID
    if left:
        status = "FAIL"
        detail += f"; orphan processes after scenario: {left}"
    for d in lab.work_dirs:
        shutil.rmtree(d, ignore_errors=True)  # our own temp folders only
    return status, detail, time.monotonic() - t0


def make_env(mock):
    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k and "AUTH_TOKEN" not in k}
    env.pop("TERM", None)
    env.update(OPENAI_API_BASE=mock.base, OPENAI_BASE_URL=mock.base, OPENAI_API_KEY=FAKE_KEY,
               AIDER_MODEL="openai/mock", AIDER_MODEL_2="openai/mock-b",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run one scenario by name")
    ap.add_argument("--quick", action="store_true", help="smaller counts and shorter holds")
    ap.add_argument("--replies", default="aider", choices=sorted(PACKS),
                    help="reply pack: how the scripted model writes edits, creates and shell commands")
    ap.add_argument("--adapter", help="adapter script to run (default adapters/aider_adapter.py)")
    ap.add_argument("--python", help="python that runs the adapter (default: this python; needs Aider)")
    args = ap.parse_args()
    global ADAPTER, PY, RESULTS
    if args.adapter:
        ADAPTER = os.path.abspath(args.adapter)
    if args.python:
        PY = os.path.abspath(args.python)
    if args.replies != "aider":  # keep stress-results.txt (Aider's record) untouched
        RESULTS = os.path.join(HERE, f"stress-results.{args.replies}.txt")
    sys.stdout.reconfigure(errors="backslashreplace")  # console may not be UTF-8 (cp950)
    names = [s[0] for s in SCENARIOS]
    if args.only and args.only not in names:
        print(f"unknown scenario {args.only!r}; choose from: {', '.join(names)}")
        return 2
    mock = Mock(args.replies)
    lines, fails = [], 0
    print(f"mock at {mock.base}; fake key only; replies {args.replies}; adapter {ADAPTER}; quick={args.quick}")
    t_all = time.monotonic()
    try:
        for name, fn, limit, qlimit in SCENARIOS:
            if args.only and name != args.only:
                continue
            status, detail, took = run_scenario(name, fn, qlimit if args.quick else limit, mock, args.quick)
            if status != "PASS":
                fails += 1
            line = f"{status} {name} [{detail}] ({took:.0f}s)"
            print(line, flush=True)
            lines.append(line)
    finally:
        mock.close()
    total = f"TOTAL {len(lines) - fails} PASS, {fails} FAIL, wall {time.monotonic() - t_all:.0f}s"
    print(total)
    lines.append(total)
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
