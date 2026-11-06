"""check_adapter.py: conformance checker for a stdio ACP adapter given as a command line.

Usage: python check_adapter.py --cmd "<command>" [--json out.json] [--replies NAME]

--replies picks the reply pack the scripted model speaks (see PACKS in check_adapter_mock.py):
aider (default). The pass rules are the same for every pack.

Each probe starts the adapter in a fresh temp work folder, with the model base URL pointed at a
local scripted mock and a fake key. Probes judge real effects (files on disk, mock requests,
process ids, timing), not the adapter's claims. Prints one line per probe: PASS, FAIL or SKIP.
Exit 1 if any probe FAILs. Every process started is killed by PID at the end of its probe.
"""
import argparse
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_adapter_client import Adapter, Fail, Skip, pid_alive  # noqa: E402
from check_adapter_mock import PACKS, Mock  # noqa: E402

TURN_WAIT = 60        # seconds for one prompt to finish; longer is a hang
PROBE_LIMIT = 240     # seconds for one whole probe; longer is a hang
NOISE = [re.compile(p, re.I | re.M) for p in (
    r"aider v\d", r"\btokens?\s*[:=]", r"\btokens?\s+(sent|received)", r"\bcost\b", r"\$\s?\d",
    r"\bmodel:\s", r"git repo", r"^warning:", r"\x1b\[")]


class Ctx:
    def __init__(self, cmd, env, mock):
        self.cmd, self.env, self.mock = cmd, env, mock
        self.work = tempfile.mkdtemp(prefix="chk_adapter_")
        self.adapters = []


def call(ad, method, params, timeout, what):
    rid = ad.request(method, params)
    try:
        return ad.wait(rid, timeout)
    except TimeoutError:
        raise Fail(f"HANG: no reply to {what} within {timeout} s; last stdout {ad.last_line()!r}; "
                   f"stderr {ad.stderr_tail()!r}")


def open_session(ctx):
    ad = Adapter(ctx.cmd, ctx.work, ctx.env)
    ctx.adapters.append(ad)
    init = call(ad, "initialize", {"protocolVersion": 1, "clientCapabilities": {}}, 60, "initialize")
    if "error" in init or not isinstance(init.get("result"), dict):
        raise Fail(f"initialize failed: {str(init)[:160]}")
    new = call(ad, "session/new", {"cwd": ctx.work, "mcpServers": []}, 60, "session/new")
    res = new.get("result") or {}
    if "error" in new or not isinstance(res.get("sessionId"), str):
        raise Fail(f"session/new failed: {str(new)[:160]}")
    return ad, res["sessionId"], res


def prompt(ad, sid, text):
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": text}]})
    try:
        resp = ad.wait(rid, TURN_WAIT)
    except TimeoutError:
        raise Fail(f"HANG: no reply to session/prompt {text[:30]!r} within {TURN_WAIT} s; "
                   f"last stdout {ad.last_line()!r}")
    ups = ad.snapshot()
    chunks = [(u, t) for u, t in ups if u.get("sessionUpdate") == "agent_message_chunk"]
    return {"stop": (resp.get("result") or {}).get("stopReason"), "error": resp.get("error"),
            "text": "".join((u.get("content") or {}).get("text", "") for u, _ in chunks
                            if isinstance(u.get("content"), dict)),
            "chunk_times": [t for _, t in chunks], "resp_time": ad.resp_time.get(rid),
            "updates": [u for u, _ in ups]}


def options(res):
    return {o.get("id"): o for o in res.get("configOptions", []) if isinstance(o, dict)}


def has_value(obj, val):
    if isinstance(obj, dict):
        return any(has_value(v, val) for v in obj.values())
    if isinstance(obj, list):
        return any(has_value(v, val) for v in obj)
    return obj == val


def read_text(path):
    return open(path, encoding="utf-8").read().strip() if os.path.isfile(path) else None


# ---- probes: each returns a detail string on PASS, raises Fail or Skip otherwise ----
def probe_handshake(ctx):
    ad, sid, _ = open_session(ctx)
    return f"initialize and session/new ok, sessionId={sid}"


def probe_stream(ctx):
    ad, sid, _ = open_session(ctx)
    t = prompt(ad, sid, "PROBE-STREAM: tell me the story")
    if t["stop"] != "end_turn":
        raise Fail(f"stopReason={t['stop']!r}, expected end_turn (err={t['error']})")
    before = [x for x in t["chunk_times"] if x < t["resp_time"]]
    if len(before) < 3:
        raise Fail(f"{len(before)} text chunks arrived before the prompt response, need 3 "
                   f"(total {len(t['chunk_times'])})")
    span = before[-1] - before[0]
    if span < 0.3:
        raise Fail(f"chunks arrived in one burst, spread {span:.2f} s")
    return f"{len(before)} chunks before the response, spread {span:.2f} s"


def probe_no_noise(ctx):
    ad, sid, _ = open_session(ctx)
    t = prompt(ad, sid, "PROBE-STREAM: tell me the story")
    missing = [w for w in ("quick brown fox", "lazy dog") if w not in t["text"]]
    if missing:
        raise Fail(f"model words missing from the chunks: {missing}; text starts {t['text'][:100]!r}")
    bad = [p.pattern for p in NOISE if p.search(t["text"])]
    if bad:
        raise Fail(f"noise in the chunks, matched {bad}; text {t['text'][:160]!r}")
    return "chunks hold the model words, no banner, token or cost lines"


def probe_edit_tool(ctx):
    ad, sid, _ = open_session(ctx)
    # A random marker in the file. The mock answers with the edit only if the model request holds
    # it, so the edit proves the adapter sent the real file content, not just a guessed answer.
    marker = "MARK-" + secrets.token_hex(8)
    with open(os.path.join(ctx.work, "hello.txt"), "w", encoding="utf-8") as f:
        f.write(f"hello\n{marker}\n")
    ctx.mock.marker = marker
    t = prompt(ad, sid, "PROBE-EDIT: change the word in hello.txt")
    body = read_text(os.path.join(ctx.work, "hello.txt"))
    if body != f"hi\n{marker}":
        raise Fail(f"hello.txt on disk is {body!r}, expected 'hi' plus the marker (stop={t['stop']}, "
                   f"err={t['error']}); the model never saw the file, so it could not edit it")
    tc = [u for u in t["updates"] if u.get("sessionUpdate") in ("tool_call", "tool_call_update")]
    if not tc:
        raise Fail("the file changed but no tool_call update was sent")
    statuses = [u.get("status") for u in tc if u.get("status")]
    if len(set(statuses)) < 2 or "completed" not in statuses:
        raise Fail(f"tool status did not change to completed: {statuses}")
    has_diff = any(isinstance(c, dict) and c.get("type") == "diff"
                   for u in tc for c in (u.get("content") or []) if isinstance(c, dict))
    has_loc = any(u.get("locations") for u in tc)
    if not (has_diff or has_loc):
        raise Fail("tool_call has neither a diff content item nor a file location")
    return f"file now 'hi'; status {' -> '.join(dict.fromkeys(statuses))}; diff={has_diff} location={has_loc}"


def probe_mention_deny(ctx):
    """The prompt names hello.txt; the client denies the add. The file must not reach the model and
    the turn must still end cleanly."""
    ad, sid, _ = open_session(ctx)
    ad.answer = "reject"
    marker = "MARK-" + secrets.token_hex(8)
    path = os.path.join(ctx.work, "hello.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"hello\n{marker}\n")
    ctx.mock.marker = marker
    i = ctx.mock.count()
    t = prompt(ad, sid, "PROBE-EDIT: change the word in hello.txt")
    if not ad.perms:
        raise Fail("the prompt names hello.txt but the client was never asked to add it")
    if any(marker in json.dumps(r) for r in ctx.mock.since(i)):
        raise Fail("denied, but the file content still reached the model")
    if read_text(path) != f"hello\n{marker}":
        raise Fail("denied, but hello.txt was changed")
    if t["stop"] != "end_turn" or t["error"]:
        raise Fail(f"turn did not end cleanly after the deny: stop={t['stop']} err={t['error']}")
    return f"asked {len(ad.perms)}x, answered 'reject', file not sent, file unchanged, end_turn"


def probe_no_escape(ctx):
    """A secret file just outside the session folder. Mentioned by relative path (..\\name) and by
    absolute path. The client would allow any add, and the file must still never reach the model."""
    ad, sid, _ = open_session(ctx)
    ad.answer = "allow"
    name = f"secret_{secrets.token_hex(6)}.txt"
    secret = "SECRET-" + secrets.token_hex(12)
    outside = os.path.join(os.path.dirname(ctx.work), name)
    with open(outside, "w", encoding="utf-8") as f:
        f.write(secret + "\n")
    try:
        i = ctx.mock.count()
        t = prompt(ad, sid, f"PROBE-TEXT: look at ..\\{name} and also {outside}")
        sent = [json.dumps(r) for r in ctx.mock.since(i)]
        if any(secret in s for s in sent):
            raise Fail("the file outside the session folder reached the model")
        if t["stop"] != "end_turn" or t["error"]:
            raise Fail(f"turn did not end cleanly: stop={t['stop']} err={t['error']}")
        if not os.path.isfile(outside) or read_text(outside) != secret:
            raise Fail("the outside file was changed")
    finally:
        try:
            os.remove(outside)  # the probe's own temp file
        except OSError:
            pass
    return f"relative ..\\ and absolute paths named; outside file never sent, end_turn"


def probe_perm_allow(ctx):
    ad, sid, _ = open_session(ctx)
    ad.answer = "allow"
    t = prompt(ad, sid, "PROBE-CREATE: create new.txt")
    if not ad.perms:
        raise Fail("creating new.txt sent no session/request_permission")
    body = read_text(os.path.join(ctx.work, "new.txt"))
    if body != "hi":
        raise Fail(f"allowed, but new.txt is {body!r} (stop={t['stop']}, err={t['error']})")
    return f"asked {len(ad.perms)}x, answered {ad.perm_answers[-1]!r}, new.txt has 'hi'"


def probe_perm_deny(ctx):
    ad, sid, _ = open_session(ctx)
    ad.answer = "reject"
    t = prompt(ad, sid, "PROBE-CREATE: create new.txt")
    if not ad.perms:
        raise Fail("creating new.txt sent no session/request_permission")
    if os.path.exists(os.path.join(ctx.work, "new.txt")):
        raise Fail("rejected, but new.txt was created anyway")
    if t["stop"] != "end_turn" or t["error"]:
        raise Fail(f"turn did not end cleanly after reject: stop={t['stop']} err={t['error']}")
    return f"asked {len(ad.perms)}x, answered {ad.perm_answers[-1]!r}, no file, end_turn"


def probe_shell_deny(ctx):
    ad, sid, _ = open_session(ctx)
    ad.answer = "reject"
    t = prompt(ad, sid, "PROBE-SHELL: run marker")
    marker = os.path.join(ctx.work, "marker.txt")
    if not ad.perms:
        raise Fail("the shell command ran or was dropped without a session/request_permission")
    if os.path.exists(marker):
        raise Fail("rejected, but marker.txt was written by the shell command")
    if t["stop"] != "end_turn" or t["error"]:
        raise Fail(f"turn did not end cleanly after reject: stop={t['stop']} err={t['error']}")
    return f"asked {len(ad.perms)}x, answered {ad.perm_answers[-1]!r}, no marker, end_turn"


def probe_shell_allow(ctx):
    ad, sid, _ = open_session(ctx)
    ad.answer = "allow"
    t = prompt(ad, sid, "PROBE-SHELL: run marker")
    marker = os.path.join(ctx.work, "marker.txt")
    if not ad.perms:
        raise Fail("the shell command sent no session/request_permission")
    if not os.path.isfile(marker):
        raise Fail(f"allowed, but marker.txt was not written (stop={t['stop']}, err={t['error']})")
    return f"asked {len(ad.perms)}x, answered {ad.perm_answers[-1]!r}, marker.txt written"


def probe_cancel(ctx):
    ad, sid, _ = open_session(ctx)
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text",
                     "text": "PROBE-COUNT: count from 1 to 300"}]})
    if not ad.wait_chunk(30):
        raise Fail("no text chunk within 30 s, so there was nothing to cancel")
    t0 = time.monotonic()
    ad.notify("session/cancel", {"sessionId": sid})
    try:
        resp = ad.wait(rid, 5)
    except TimeoutError:
        raise Fail("no reply to session/prompt within 5 s after session/cancel")
    took = time.monotonic() - t0
    stop = (resp.get("result") or {}).get("stopReason")
    if stop != "cancelled":
        raise Fail(f"stopReason={stop!r} after cancel, expected 'cancelled' (err={resp.get('error')})")
    t = prompt(ad, sid, "PROBE-TEXT: say the word")
    if t["stop"] != "end_turn" or "pineapple" not in t["text"].lower():
        raise Fail(f"the next prompt after cancel failed: stop={t['stop']} text={t['text'][:60]!r}")
    return f"stopReason cancelled in {took:.2f} s; next prompt in the same session worked"


def probe_memory(ctx):
    ad, sid, _ = open_session(ctx)
    prompt(ad, sid, "PROBE-TEXT: reply with the word")
    pid1 = ad.adapter_pid()
    i = ctx.mock.count()
    t2 = prompt(ad, sid, "What word did you just reply with? One word.")
    pid2 = ad.adapter_pid()
    reqs = ctx.mock.since(i)
    if not reqs:
        raise Fail("turn 2 sent no request to the model")
    msgs = reqs[-1].get("messages", [])
    if not any(m.get("role") == "assistant" and "pineapple" in str(m.get("content", "")).lower() for m in msgs):
        raise Fail("turn 2 request to the model did not contain turn 1's reply")
    if t2["stop"] != "end_turn" or "pineapple" not in t2["text"].lower():
        raise Fail(f"turn 2 reply wrong: stop={t2['stop']} text={t2['text'][:60]!r}")
    if pid1 != pid2 or not pid_alive(pid2):
        raise Fail(f"adapter process changed between turns: pid {pid1} -> {pid2}")
    return f"turn 2 request carried turn 1's reply; adapter pid {pid1} unchanged"


def probe_model_switch(ctx):
    ad, sid, res = open_session(ctx)
    opt = options(res).get("model")
    if not opt:
        raise Skip("adapter lists no model option")
    cur = opt.get("currentValue")
    vals = [o.get("value") for o in opt.get("options", []) if isinstance(o, dict)]
    new = next((v for v in vals if v != cur), None)
    if new is None:
        raise Skip("adapter lists only one model value")
    r = call(ad, "session/set_config_option", {"sessionId": sid, "configId": "model", "value": new}, 30, "set model")
    if "error" in r:
        raise Fail(f"setting model {new!r} was refused: {r['error']}")
    ad.reset()
    rid = ad.request("session/prompt", {"sessionId": sid, "prompt": [{"type": "text",
                     "text": "PROBE-COUNT: count from 1 to 300"}]})
    if not ad.wait_chunk(30):
        raise Fail("no chunk within 30 s for the during-turn test")
    busy = call(ad, "session/set_config_option", {"sessionId": sid, "configId": "model", "value": cur}, 15,
                "set model during a turn")
    ad.notify("session/cancel", {"sessionId": sid})
    try:
        ad.wait(rid, 30)
    except TimeoutError:
        raise Fail("the turn did not stop after session/cancel")
    if "error" not in busy:
        raise Fail("a model change during a running turn was accepted, expected an error")
    i = ctx.mock.count()
    t = prompt(ad, sid, "PROBE-TEXT: say the word")
    reqs = ctx.mock.since(i)
    model = str(reqs[-1].get("model")) if reqs else ""
    if new not in model:
        raise Fail(f"next request model={model!r}, expected it to carry {new!r}")
    return f"model {cur!r} -> {new!r}; next request carries it; change refused during a turn"


def probe_effort(ctx):
    ad, sid, res = open_session(ctx)
    opt = options(res).get("effort")
    if not opt:
        raise Skip("adapter lists no effort option")
    cur = opt.get("currentValue")
    vals = [o.get("value") for o in opt.get("options", []) if isinstance(o, dict)]
    new = next((v for v in vals if v != cur), None)
    if new is None:
        raise Skip("adapter lists only one effort value")
    r = call(ad, "session/set_config_option", {"sessionId": sid, "configId": "effort", "value": new}, 30, "set effort")
    if "error" in r:
        raise Fail(f"setting effort {new!r} was refused: {r['error']}")
    i = ctx.mock.count()
    prompt(ad, sid, "PROBE-TEXT: say the word")
    reqs = ctx.mock.since(i)
    body = {k: v for k, v in (reqs[-1] if reqs else {}).items() if k != "messages"}
    if not has_value(body, new):
        raise Fail(f"next request {json.dumps(body)[:200]} does not carry effort {new!r}")
    return f"effort {cur!r} -> {new!r}; next request carries it"


def probe_error(ctx):
    ad, sid, _ = open_session(ctx)
    t = prompt(ad, sid, "PROBE-FAIL: hello")
    if t["error"] is None or t["stop"] is not None:
        raise Fail(f"a 401 from the model ended as stop={t['stop']!r} err={t['error']}, expected a JSON-RPC error")
    msg = (t["error"] or {}).get("message", "") if isinstance(t["error"], dict) else str(t["error"])
    return f"JSON-RPC error {(t['error'] or {}).get('code')}: {str(msg)[:80]!r}"


def probe_stdout_clean(ctx):
    ad, sid, _ = open_session(ctx)
    prompt(ad, sid, "PROBE-TEXT: say the word")
    time.sleep(0.5)  # let any trailing stdout arrive
    lines = ad.raw_lines()
    if not lines:
        raise Fail("the adapter wrote nothing to stdout")
    if ad.bad_lines:
        raise Fail(f"{len(ad.bad_lines)} stdout line(s) are not JSON-RPC 2.0, first: {ad.bad_lines[0]!r}")
    return f"{len(lines)} stdout lines, all JSON-RPC 2.0"


def thought_text(t):
    return "".join((u.get("content") or {}).get("text", "") for u in t["updates"]
                   if u.get("sessionUpdate") == "agent_thought_chunk" and isinstance(u.get("content"), dict))


def tool_text(updates):
    """Text carried by the content items of tool_call / tool_call_update notifications."""
    out = []
    for u in updates:
        if u.get("sessionUpdate") not in ("tool_call", "tool_call_update"):
            continue
        for c in u.get("content") or []:
            inner = c.get("content") if isinstance(c, dict) else None
            if isinstance(inner, dict):
                out.append(str(inner.get("text", "")))
    return "\n".join(out)


def probe_thought_split(ctx):
    """Fake reasoning arrives as reasoning deltas, cut into 3-character pieces. It must reach the client
    as agent_thought_chunk, and the answer alone as agent_message_chunk, with no header lines."""
    ad, sid, _ = open_session(ctx)
    t = prompt(ad, sid, "PROBE-THINK: what word is the answer?")
    if t["stop"] != "end_turn":
        raise Fail(f"stopReason={t['stop']!r}, err={t['error']}")
    thought, msg = thought_text(t), t["text"]
    if "Weighing the riddle" not in thought:
        raise Fail(f"reasoning did not arrive as agent_thought_chunk; thought={thought[:80]!r}")
    if "Weighing" in msg:
        raise Fail(f"reasoning text also arrived in agent_message_chunk: {msg[:100]!r}")
    if "pineapple" not in msg:
        raise Fail(f"the answer is missing from agent_message_chunk: {msg[:100]!r}")
    if re.search(r"THINKING|ANSWER|►", msg):
        raise Fail(f"a reasoning header line is in the message text: {msg[:120]!r}")
    return "reasoning in thought chunks only; answer alone in message chunks; no header lines"


def probe_thought_inline(ctx):
    """The model writes its reasoning inline as think tags, cut into 3-character pieces, so each tag is
    split across chunks. The tags must not reach the message text and the reasoning must be a thought."""
    ad, sid, _ = open_session(ctx)
    t = prompt(ad, sid, "PROBE-THINK-INLINE: what word is the answer?")
    if t["stop"] != "end_turn":
        raise Fail(f"stopReason={t['stop']!r}, err={t['error']}")
    thought, msg = thought_text(t), t["text"]
    if "Weighing the riddle" not in thought:
        raise Fail(f"inline reasoning did not arrive as agent_thought_chunk; thought={thought[:80]!r}")
    if re.search(r"</?think|Weighing", msg):
        raise Fail(f"think tags or reasoning text in the message chunks: {msg[:120]!r}")
    if "pineapple" not in msg:
        raise Fail(f"the answer is missing from agent_message_chunk: {msg[:100]!r}")
    return "inline think tags split across chunks; reasoning as thought, answer as message"


def probe_edit_hidden(ctx):
    """Prose before and after a SEARCH/REPLACE block streams; the block itself never appears as text.
    The edit must still be made on disk and reported as a tool_call of kind edit with a diff."""
    ad, sid, _ = open_session(ctx)
    marker = "MARK-" + secrets.token_hex(8)
    with open(os.path.join(ctx.work, "hello.txt"), "w", encoding="utf-8") as f:
        f.write(f"hello\n{marker}\n")
    ctx.mock.marker = marker
    t = prompt(ad, sid, "PROBE-EDIT-PROSE: change the word in hello.txt")
    body = read_text(os.path.join(ctx.work, "hello.txt"))
    if body != f"hi\n{marker}":
        raise Fail(f"hello.txt on disk is {body!r}, expected 'hi' plus the marker (stop={t['stop']})")
    msg = t["text"]
    found = [m for m in ("<<<<<<<", "=======", ">>>>>>>", "SEARCH", "REPLACE") if m in msg]
    if found:
        raise Fail(f"edit block text in agent_message_chunk: {found}; text {msg[:160]!r}")
    for prose in ("Sure, I will change the word.", "Done, hello.txt is updated."):
        if prose not in msg:
            raise Fail(f"text around the block did not stream: {prose!r} missing from {msg[:160]!r}")
    if not any(u.get("sessionUpdate") == "tool_call" and u.get("kind") == "edit" for u in t["updates"]):
        raise Fail("no tool_call with kind 'edit'")
    diffs = [c for u in t["updates"] if u.get("sessionUpdate") == "tool_call_update"
             for c in (u.get("content") or []) if isinstance(c, dict) and c.get("type") == "diff"]
    if not any(str(d.get("path", "")).replace("\\", "/").endswith("/hello.txt") and marker in (d.get("newText") or "")
               for d in diffs):
        raise Fail(f"no diff with the real path and the new text; diffs={str(diffs[:1])[:160]}")
    return "block hidden from text; prose streamed; tool_call kind edit with diff at hello.txt"


def probe_shell_output(ctx):
    """An allowed shell command: one permission card (kind execute), a tool_call of kind execute, and
    its output in a completed tool_call_update. Aider's second question 'Add command output' is never
    shown to the client."""
    ad, sid, _ = open_session(ctx)
    ad.answer = "allow"
    tok = "SHOUT-" + secrets.token_hex(6)
    t = prompt(ad, sid, f"PROBE-SHELL-OUT: run echo {tok}")
    run_perms = [p for p in ad.perms if (p.get("toolCall") or {}).get("kind") == "execute"]
    if len(ad.perms) != 1 or len(run_perms) != 1:
        titles = [(p.get("toolCall") or {}).get("title", "")[:50] for p in ad.perms]
        raise Fail(f"expected exactly one permission request for the shell command, got {len(ad.perms)}: {titles}")
    if any("add command output" in json.dumps(p).lower() for p in ad.perms):
        raise Fail("the client was asked 'Add command output to the chat?' as well")
    calls = [u for u in t["updates"] if u.get("sessionUpdate") == "tool_call" and u.get("kind") == "execute"]
    if not calls:
        raise Fail("no tool_call with kind 'execute' for the allowed command")
    done = [u for u in t["updates"] if u.get("sessionUpdate") == "tool_call_update"
            and u.get("status") in ("completed", "failed") and tok in tool_text([u])]
    if not done:
        raise Fail(f"the command output {tok} is not in any tool_call_update; updates "
                   f"{[u.get('status') for u in t['updates'] if u.get('sessionUpdate') == 'tool_call_update']}")
    if not any(u.get("status") == "completed" for u in done):
        raise Fail(f"output reported but status is not completed: {[u.get('status') for u in done]}")
    return f"one permission (execute); tool_call execute; output {tok} in a completed update"


def probe_usage(ctx):
    """Context use: when the adapter sends usage_update, used and size must be real positive numbers
    with used <= size. The mock model has no known context window, so none is sent (SKIP)."""
    ad, sid, _ = open_session(ctx)
    t = prompt(ad, sid, "PROBE-TEXT: say the word")
    ups = [u for u in t["updates"] if u.get("sessionUpdate") == "usage_update"]
    if not ups:
        raise Skip("no usage_update: the mock model has no known context window")
    u = ups[-1]
    used, size = u.get("used"), u.get("size")
    if not (isinstance(used, int) and isinstance(size, int) and 0 < used <= size):
        raise Fail(f"usage_update has bad numbers: used={used!r} size={size!r}")
    return f"usage_update used={used} size={size}"


PROBES = [("handshake", probe_handshake), ("stream", probe_stream), ("no_noise", probe_no_noise),
          ("edit_tool", probe_edit_tool), ("mention_deny", probe_mention_deny),
          ("no_escape", probe_no_escape), ("perm_allow", probe_perm_allow), ("perm_deny", probe_perm_deny),
          ("shell_deny", probe_shell_deny), ("shell_allow", probe_shell_allow), ("cancel", probe_cancel),
          ("memory", probe_memory), ("model_switch", probe_model_switch), ("effort", probe_effort),
          ("error", probe_error), ("stdout_clean", probe_stdout_clean),
          ("thought_split", probe_thought_split), ("thought_inline", probe_thought_inline),
          ("edit_hidden", probe_edit_hidden), ("shell_output", probe_shell_output), ("usage", probe_usage)]


def run_probe(name, fn, ctx):
    box = {}

    def body():
        try:
            box["detail"] = fn(ctx)
            box["status"] = "PASS"
        except Skip as e:
            box["status"], box["detail"] = "SKIP", str(e)
        except Fail as e:
            box["status"], box["detail"] = "FAIL", str(e)
        except Exception as e:  # a checker bug or a crash is a FAIL, never a silent pass
            box["status"], box["detail"] = "FAIL", f"checker error: {e!r}"

    th = threading.Thread(target=body, daemon=True)
    th.start()
    th.join(PROBE_LIMIT)
    if th.is_alive():
        return "FAIL", f"HANG: probe did not finish within {PROBE_LIMIT} s"
    return box["status"], box["detail"]


def main():
    ap = argparse.ArgumentParser(description="conformance checker for a stdio ACP adapter")
    ap.add_argument("--cmd", required=True, help='the adapter command line, e.g. "python my_adapter.py"')
    ap.add_argument("--json", help="write the results to this file")
    ap.add_argument("--replies", default="aider", choices=sorted(PACKS),
                    help="reply pack: how the scripted model writes edits, creates and shell commands")
    args = ap.parse_args()

    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k and "AUTH_TOKEN" not in k}
    env.pop("TERM", None)
    mock = Mock(args.replies)
    env.update(OPENAI_API_BASE=mock.base, OPENAI_BASE_URL=mock.base, OPENAI_API_KEY="fake-key-checker",
               AIDER_MODEL="openai/mock", AIDER_MODEL_2="openai/mock-b",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    print(f"mock model at {mock.base}; fake key only; replies: {args.replies}; adapter: {args.cmd}")

    results = []
    try:
        for name, fn in PROBES:
            ctx = Ctx(args.cmd, env, mock)
            try:
                status, detail = run_probe(name, fn, ctx)
            finally:
                for ad in ctx.adapters:
                    ad.close()  # stops the adapter's process tree by PID
                shutil.rmtree(ctx.work, ignore_errors=True)
            detail = " ".join(str(detail).split())[:260]
            print(f"{status:<5} {name:<13} {detail}", flush=True)
            results.append({"probe": name, "status": status, "detail": detail})
    finally:
        mock.close()

    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\nPASS {counts['PASS']}  FAIL {counts['FAIL']}  SKIP {counts['SKIP']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"cmd": args.cmd, "counts": counts, "probes": results}, f, indent=2)
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
