"""run_adapter_real.py: probes adapters/aider_adapter.py (ACP stdio adapter for Aider).

  python run_adapter_real.py          asks for an OpenRouter key (getpass), runs a real model,
                                      writes adapter-real-results.txt
  python run_adapter_real.py --mock   no key, scripted local server from check_adapter_mock,
                                      writes adapter-mock-results.txt (never the real file)

The key goes only into the adapter's environment. It is never printed, logged or saved, and
every output line is scrubbed of it. Each probe starts its own adapter in a fresh temp folder,
deleted afterwards. Adapters are stopped by PID only (check_adapter_client.kill_tree).
The error probe always uses the local 401 stub, so a fake key never leaves this machine.
"""
import argparse
import getpass
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_adapter_client import Adapter, pid_alive  # noqa: E402
from check_adapter_mock import Mock  # noqa: E402

VENV_PY = sys.executable  # override with --python; that Python needs Aider installed
ADAPTER = os.path.abspath(os.path.join(HERE, os.pardir, "adapters", "aider_adapter.py"))
REAL_RESULTS = os.path.join(HERE, "adapter-real-results.txt")
MOCK_RESULTS = os.path.join(HERE, "adapter-mock-results.txt")
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4.1-flash-1017"
SWITCH_MODEL = "openrouter/deepseek/deepseek-chat"
MOCK_MODEL, MOCK_MODEL_2 = "openai/mock", "openai/mock-b"
WORDS = ["lantern", "walnut", "comet", "saddle", "orchid", "pebble", "thimble", "glacier"]
TURN_WAIT = 90  # seconds per turn; no reply in time = hang = FAIL
MAX_ATTEMPTS = 3  # the first try plus 2 retries

MOCK = False
MOCK_SRV = None
SECRET = [None]
OUT, RESULTS = [], []
TURNS = [0]
BAD_LINES, RAW_COUNT = [], [0]


def scrub(text):
    s = str(text)
    if SECRET[0]:
        s = s.replace(SECRET[0], "***")
    return s


def emit(text):
    text = scrub(text)
    print(text, flush=True)
    OUT.append(text)


def record(name, status, reason):
    RESULTS.append(status)
    emit(f"{status} {name}  [{reason}]")


def tag(text, probe_tag):
    """Mock only: the scripted server picks its reply from a PROBE-* token in the prompt."""
    return f"{text} {probe_tag}" if MOCK else text


def base_env():
    env = dict(os.environ)
    for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENROUTER_API_BASE",
              "AIDER_MODEL", "AIDER_MODEL_2", "ADAPTER_MODEL", "ADAPTER_MODELS", "TERM"):
        env.pop(k, None)  # TERM: Git Bash TERM makes aider print a prompt_toolkit warning
    return env


def run_adapter(env_extra):
    """Start the adapter with the chosen python in a fresh temp folder (absolute paths)."""
    folder = tempfile.mkdtemp(prefix="adapter-probe-")
    env = base_env()
    env.update(env_extra)
    cmd = subprocess.list2cmdline([VENV_PY, ADAPTER])
    return folder, Adapter(cmd, folder, env)


def stop_adapter(folder, a):
    try:
        a.close()
    finally:
        BAD_LINES.extend(a.bad_lines)
        RAW_COUNT[0] += len(a.raw)
        shutil.rmtree(folder, ignore_errors=True)


def start_session(a, folder):
    a.wait(a.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}}), 60)
    res = a.wait(a.request("session/new", {"cwd": folder, "mcpServers": []}), 60)
    result = res.get("result") or {}
    if "sessionId" not in result:
        raise RuntimeError(f"session/new failed: {res}")
    return result["sessionId"], {o["id"]: o for o in result.get("configOptions", [])}


def turn(a, sid, text, timeout=TURN_WAIT):
    """One prompt. Clears the update buffer first, so snapshot() holds only this turn."""
    TURNS[0] += 1
    a.reset()
    rid = a.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": text}]})
    return rid, a.wait(rid, timeout)


def reply_text(a):
    return "".join((u.get("content") or {}).get("text", "") for u, _ in a.snapshot()
                   if u.get("sessionUpdate") == "agent_message_chunk")


def stop_of(resp):
    return (resp.get("result") or {}).get("stopReason")


def chunks_before(a, rid):
    with a.cv:
        t_end = a.resp_time.get(rid, float("inf"))
        return sum(1 for u, t in a.updates
                   if u.get("sessionUpdate") == "agent_message_chunk" and t <= t_end)


def read_text(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main_env():
    if MOCK:
        return {"OPENAI_API_BASE": MOCK_SRV.base, "OPENAI_API_KEY": "dummy-key-local",
                "AIDER_MODEL": MOCK_MODEL, "AIDER_MODEL_2": MOCK_MODEL_2}
    return {"OPENROUTER_API_KEY": SECRET[0], "AIDER_MODEL": DEFAULT_MODEL,
            "AIDER_MODEL_2": SWITCH_MODEL}


# ---- attempts: each returns (status, reason); status PASS, FAIL, NOASK ----
def att_stream():
    d, a = run_adapter(main_env())
    try:
        sid, _ = start_session(a, d)
        rid, resp = turn(a, sid, tag("Count from 1 to 30.", "PROBE-COUNT"))
        if "error" in resp:
            return "FAIL", f"error reply: {resp['error']}"
        n, stop = chunks_before(a, rid), stop_of(resp)
        ok = n >= 3 and stop == "end_turn"
        return ("PASS" if ok else "FAIL"), f"{n} chunks before the response, stopReason={stop}"
    finally:
        stop_adapter(d, a)


def att_edit():
    d, a = run_adapter(main_env())
    try:
        hello = os.path.join(d, "hello.txt")
        marker = None
        if MOCK:
            # The mock answers with the edit only if the model request holds this marker, so the
            # edit proves the adapter sent the real file content.
            marker = "MARK-" + secrets.token_hex(8)
            MOCK_SRV.marker = marker
            write_text(hello, f"hello\n{marker}\n")
        else:
            write_text(hello, "hello")
        a.answer = "allow"
        sid, _ = start_session(a, d)
        n0 = len(a.perms)
        rid, resp = turn(a, sid, tag("In hello.txt, change the word hello to hi.", "PROBE-EDIT"))
        if "error" in resp:
            return "FAIL", f"error reply: {resp['error']}"
        body = read_text(hello)
        ups = [u for u, _ in a.snapshot() if u.get("sessionUpdate") in ("tool_call", "tool_call_update")]
        located = any(u.get("locations") or any(isinstance(c, dict) and c.get("type") == "diff"
                                                for c in (u.get("content") or []))
                      for u in ups)
        asked = len(a.perms) - n0
        expect = f"hi\n{marker}" if MOCK else "hi"
        ok = body == expect and located
        return ("PASS" if ok else "FAIL"), (f"hello.txt={body!r}, tool_call with location/diff seen={located}, "
                                             f"permission asks={asked} (allowed)")
    finally:
        stop_adapter(d, a)


def att_create(answer):
    d, a = run_adapter(main_env())
    try:
        a.answer = answer
        sid, _ = start_session(a, d)
        n0 = len(a.perms)
        rid, resp = turn(a, sid, tag("Create new.txt containing the single word hi.", "PROBE-CREATE"))
        if "error" in resp:
            return "FAIL", f"error reply: {resp['error']}"
        asked = len(a.perms) - n0
        exists = os.path.exists(os.path.join(d, "new.txt"))
        if asked == 0:
            return "FAIL", "no permission ask, so the edit was not gated"
        ok = (not exists) if answer == "reject" else exists
        return ("PASS" if ok else "FAIL"), f"asked={asked}, answered {answer}, new.txt exists={exists}"
    finally:
        stop_adapter(d, a)


def att_shell(answer):
    d, a = run_adapter(main_env())
    try:
        a.answer = answer
        sid, _ = start_session(a, d)
        n0 = len(a.perms)
        rid, resp = turn(a, sid, tag("Suggest a shell command that writes marker.txt. Do not run it yourself.",
                                     "PROBE-SHELL"))
        if "error" in resp:
            return "FAIL", f"error reply: {resp['error']}"
        shell_asks = [p for p in a.perms[n0:] if (p.get("toolCall") or {}).get("kind") == "execute"]
        if not shell_asks:
            return "NOASK", "no shell permission ask"
        exists = os.path.exists(os.path.join(d, "marker.txt"))
        ok = exists if answer == "allow" else not exists
        return ("PASS" if ok else "FAIL"), f"shell asks={len(shell_asks)}, answered {answer}, marker.txt exists={exists}"
    finally:
        stop_adapter(d, a)


def att_cancel():
    d, a = run_adapter(main_env())
    try:
        sid, _ = start_session(a, d)
        a.reset()
        TURNS[0] += 1
        rid = a.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text":
                        tag("Write a 500 word story.", "PROBE-COUNT")}]})
        with a.cv:
            got = a.cv.wait_for(lambda: sum(1 for u, _ in a.updates
                                            if u.get("sessionUpdate") == "agent_message_chunk") >= 3, TURN_WAIT)
        if not got:
            return "FAIL", "fewer than 3 chunks arrived before the turn ended, nothing to cancel"
        t0 = time.monotonic()
        a.notify("session/cancel", {"sessionId": sid})
        resp = a.wait(rid, 10)  # TimeoutError past 10 s is a FAIL
        dt = time.monotonic() - t0
        stop = stop_of(resp)
        if stop != "cancelled":
            return "FAIL", f"stopReason={stop} after cancel"
        rid2, resp2 = turn(a, sid, "Reply with one word: ok")
        text2 = reply_text(a).strip()
        ok2 = stop_of(resp2) == "end_turn" and bool(text2)
        return ("PASS" if ok2 else "FAIL"), (f"cancelled in {dt:.1f}s; next prompt stopReason={stop_of(resp2)}, "
                                              f"reply {len(text2)} chars")
    finally:
        stop_adapter(d, a)


def att_memory():
    d, a = run_adapter(main_env())
    try:
        word = "pineapple" if MOCK else random.choice(WORDS)
        sid, _ = start_session(a, d)
        pid1 = a.adapter_pid()
        _, r1 = turn(a, sid, tag(f"The code word is {word}. Just acknowledge it.", "PROBE-TEXT"))
        if "error" in r1:
            return "FAIL", f"turn 1 error: {r1['error']}"
        q = "What word did you just reply with?" if MOCK else "What is the code word I told you? Reply with only the word."
        _, r2 = turn(a, sid, q)
        if "error" in r2:
            return "FAIL", f"turn 2 error: {r2['error']}"
        text2 = reply_text(a)
        pid2 = a.adapter_pid()
        ok = word in text2.lower() and pid1 == pid2 and pid_alive(pid2)
        return ("PASS" if ok else "FAIL"), (f"word in reply={word in text2.lower()}, "
                                             f"same adapter PID={pid1 == pid2} ({pid2})")
    finally:
        stop_adapter(d, a)


def att_model_switch():
    d, a = run_adapter(main_env())
    try:
        sid, opts = start_session(a, d)
        target = MOCK_MODEL_2 if MOCK else SWITCH_MODEL
        opt = next((o for o in opts["model"].get("options", []) if o.get("name") == target), None)
        if opt is None:
            return "SKIP", f"the option list does not offer {target}"
        rid = a.request("session/set_config_option", {"sessionId": sid, "configId": "model", "value": opt["value"]})
        resp = a.wait(rid, 30)
        if "error" in resp:
            return "FAIL", f"set model refused: {resp['error']}"
        _, r2 = turn(a, sid, "Reply with one word: ok")
        text = reply_text(a).strip()
        ok = stop_of(r2) == "end_turn" and bool(text)
        return ("PASS" if ok else "FAIL"), (f"set {target}: accepted and next turn worked "
                                             f"(stopReason={stop_of(r2)}); this does not prove which model answered")
    finally:
        stop_adapter(d, a)


def att_effort():
    d, a = run_adapter(main_env())
    try:
        sid, _ = start_session(a, d)
        rid = a.request("session/set_config_option", {"sessionId": sid, "configId": "effort", "value": "high"})
        resp = a.wait(rid, 30)
        if "error" in resp:
            return "FAIL", f"set effort refused: {resp['error']}"
        cur = {o["id"]: o.get("currentValue") for o in (resp.get("result") or {}).get("configOptions", [])}
        _, r2 = turn(a, sid, "Reply with one word: ok")
        text = reply_text(a).strip()
        ok = cur.get("effort") == "high" and stop_of(r2) == "end_turn" and bool(text)
        return ("PASS" if ok else "FAIL"), (f"effort high accepted, next turn worked (stopReason={stop_of(r2)}); "
                                             "not proven to reach the model")
    finally:
        stop_adapter(d, a)


def att_error():
    err_env = {"OPENAI_API_KEY": "bad-key-for-test", "OPENAI_API_BASE": MOCK_SRV.base,
               "AIDER_MODEL": MOCK_MODEL}
    d, a = run_adapter(err_env)
    try:
        sid, _ = start_session(a, d)
        _, resp = turn(a, sid, "Say hi. PROBE-FAIL")
        err = resp.get("error")
        if err and "result" not in resp:
            return "PASS", f"JSON-RPC error {err.get('code')}: {str(err.get('message'))[:120]}"
        return "FAIL", f"prompt ended without an error: {str(resp)[:160]}"
    finally:
        stop_adapter(d, a)


def run_probe(name, attempt, *args, retries=True):
    """Runs the attempt, retrying on FAIL when retries is True. Every try is printed, none hidden."""
    noask = 0
    status, reason, tries = "FAIL", "", 0
    limit = MAX_ATTEMPTS if retries else 1
    for i in range(limit):
        tries = i + 1
        try:
            status, reason = attempt(*args)
        except TimeoutError as e:
            status, reason = "FAIL", f"hang: {e}"
        except Exception as e:  # a harness problem is a FAIL for this probe, not a crash
            status, reason = "FAIL", f"harness error: {type(e).__name__}: {e}"
        emit(f"    try {tries}/{limit}: {status} {reason}")
        if status == "NOASK":
            noask += 1
            if noask >= 2:
                status, reason = "SKIP", "no shell permission ask after 2 tries (model did not suggest one)"
                break
            continue
        if status in ("PASS", "SKIP"):
            break
    retried = tries - 1
    if retried and status != "SKIP":
        reason += f" (after {retried} retr{'y' if retried == 1 else 'ies'})"
    record(name, status if status != "NOASK" else "FAIL", reason)
    return status


def probe_stdout_clean():
    bad = list(BAD_LINES)
    if not bad and RAW_COUNT[0] == 0:
        # Only the probes that start adapters fill these counters: a partial run cannot judge them.
        return record("stdout_clean", "SKIP", "no adapter output yet; run the full set to judge stdout")
    ok = not bad and RAW_COUNT[0] > 0
    detail = f"{RAW_COUNT[0]} stdout lines, all JSON-RPC 2.0" if ok else \
        f"{len(bad)} non-JSON-RPC stdout line(s), first: {bad[0] if bad else 'no stdout seen'}"
    record("stdout_clean", "PASS" if ok else "FAIL", detail)


def check_key():
    key = getpass.getpass("OpenRouter key: ").strip()
    SECRET[0] = key or None
    print(f"Key read: {len(key)} characters, starts with sk-or-: {key.startswith('sk-or-')}")
    if not key.startswith("sk-or-"):
        sys.exit("This is not an OpenRouter key. They start with sk-or-. Nothing was sent.")
    try:
        urllib.request.urlopen(urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": "Bearer " + key}), timeout=20).close()
        print("OpenRouter accepts this key.")
    except urllib.error.HTTPError as e:
        sys.exit(f"OpenRouter refused this key (HTTP {e.code}). Make a new key and try again.")


def plan():
    """(name, runner). A runner takes retries=True or False and returns the probe's status."""
    return [
        ("stream", lambda r: run_probe("stream", att_stream, retries=r)),
        ("edit", lambda r: run_probe("edit", att_edit, retries=r)),
        ("perm_deny", lambda r: run_probe("perm_deny", att_create, "reject", retries=r)),
        ("perm_allow", lambda r: run_probe("perm_allow", att_create, "allow", retries=r)),
        ("shell_deny", lambda r: run_probe("shell_deny", att_shell, "reject", retries=r)),
        ("shell_allow", lambda r: run_probe("shell_allow", att_shell, "allow", retries=r)),
        ("cancel", lambda r: run_probe("cancel", att_cancel, retries=r)),
        ("memory", lambda r: run_probe("memory", att_memory, retries=r)),
        ("model_switch", lambda r: run_probe("model_switch", att_model_switch, retries=r)),
        ("effort", lambda r: run_probe("effort", att_effort, retries=r)),
        ("error", lambda r: run_probe("error", att_error, retries=r)),
        ("stdout_clean", lambda r: probe_stdout_clean()),
    ]


ARGS = argparse.Namespace(only=None, repeat=1)


def main():
    global MOCK, MOCK_SRV, ARGS, VENV_PY, ADAPTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", metavar="NAME", help="run only this probe, e.g. edit")
    ap.add_argument("--repeat", type=int, default=1, metavar="N", help="run the --only probe N times")
    ap.add_argument("--adapter", metavar="PATH", help="adapter script (default adapters/aider_adapter.py)")
    ap.add_argument("--python", metavar="PATH", help="python that runs the adapter (default: this python; needs Aider)")
    args = ap.parse_args()
    if args.adapter:
        ADAPTER = os.path.abspath(args.adapter)
    if args.python:
        VENV_PY = os.path.abspath(args.python)
    names = [n for n, _ in plan()]
    if args.only and args.only not in names:
        ap.error(f"--only: unknown probe {args.only!r}; choose from {', '.join(names)}")
    if args.repeat < 1:
        ap.error("--repeat must be 1 or more")
    if args.repeat > 1 and not args.only:
        ap.error("--repeat needs --only NAME")
    ARGS = args
    MOCK = args.mock
    results_path = MOCK_RESULTS if MOCK else REAL_RESULTS
    MOCK_SRV = Mock()  # localhost only; scripted replies
    try:
        if MOCK:
            emit(f"mode: mock ({MOCK_SRV.base})")
        else:
            check_key()
            emit(f"mode: real, model {DEFAULT_MODEL}, switch to {SWITCH_MODEL}")

        for name, runner in plan():
            if ARGS.only and name != ARGS.only:
                continue
            if ARGS.repeat > 1 and name != "stdout_clean":
                passes = 0
                for k in range(ARGS.repeat):
                    emit(f"  run {k + 1}/{ARGS.repeat}")
                    if runner(False) == "PASS":
                        passes += 1
                emit(f"{name} {passes}/{ARGS.repeat} PASS")
            else:
                runner(True)

        p, f, s = RESULTS.count("PASS"), RESULTS.count("FAIL"), RESULTS.count("SKIP")
        emit(f"TOTAL PASS {p}  FAIL {f}  SKIP {s}")
        emit(f"turns used: {TURNS[0]}")
        emit("cost unknown, check OpenRouter (the adapter does not report cost)" if not MOCK
             else "cost: none (mock)")
    finally:
        MOCK_SRV.close()
        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(OUT) + "\n")
    sys.exit(0 if RESULTS and "FAIL" not in RESULTS else 1)


if __name__ == "__main__":
    main()
