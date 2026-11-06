#!/usr/bin/env python3
"""ACP (Agent Client Protocol) stdio adapter for Aider.

One long-lived process. Speaks JSON-RPC 2.0 on stdin/stdout, one object per line, and runs
Aider in-process (Coder.send_message), not as the aider CLI. Anything Aider prints goes to stderr.
Stdlib plus Aider only (and psutil and tiktoken, which Aider already installs).

Run with: python adapters/aider_adapter.py (Aider 0.86.2 installed in that Python).
Environment: AIDER_MODEL (e.g. openai/gpt-4o), optional AIDER_MODEL_2 (second choice),
OPENAI_API_BASE / OPENAI_API_KEY as litellm reads them.

Turns: each prompt runs on its own worker thread and owns a Turn. A cancel (or the worker
finishing) answers the prompt exactly once. A cancelled turn is stale: its updates are dropped,
its edits and shell commands are refused, and its model requests stop. The next prompt continues
on a fresh coder seeded with the chat history when the stale worker still holds the old one.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid

# The protocol owns the real stdout. Aider and litellm print to sys.stdout, so route them to stderr.
_OUT = sys.stdout.buffer
sys.stdout = sys.stderr

import psutil  # noqa: E402  (Aider already depends on it)
import tiktoken.core as _tiktoken_core  # noqa: E402
from aider import models  # noqa: E402
from aider.coders import base_coder  # noqa: E402
from aider.coders.editblock_coder import EditBlockCoder  # noqa: E402
from aider.io import InputOutput  # noqa: E402
from aider.reasoning_tags import REASONING_TAG  # noqa: E402

PROTOCOL_VERSION = 1
EFFORTS = ["low", "medium", "high"]
DEFAULT_EFFORT = "medium"
DEFAULT_MODEL = "openai/gpt-4o-mini"
NO_REPLY = object()
MAX_LIVE_TURNS = 3           # turns (cancelled ones included) whose worker may still run
RETRY_GIVE_UP_SECONDS = 4    # Aider's backoff waits 0.25, 0.5, 1, 2, 4 s, then gives up: about 8 s in all
TOKEN_CHUNK = 4096
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
SHELL_SHOW_CHARS = 20000     # command output sent to the client as a tool call; longer output is cut
SEARCH_LINE = re.compile(r"<{5,9} SEARCH\s*$")
REPLACE_LINE = re.compile(r">{5,9} REPLACE\s*$")
SEARCH_PREFIX = re.compile(r"<{0,9}( ?S?E?A?R?C?H?)?")   # a line that may still become a SEARCH line
MAX_CANDIDATES = 2           # single-word prose lines held back: one may be a block's filename

# Aider module globals changed at runtime. No Aider file is edited.
# RETRY_TIMEOUT: base_coder compares its backoff with this name, so a 60 s give-up becomes ~8 s.
base_coder.RETRY_TIMEOUT = RETRY_GIVE_UP_SECONDS
# run_cmd: base_coder.handle_shell_commands calls this name; run_cmd() below replaces it.

_orig_encode = _tiktoken_core.Encoding.encode


def _encode_in_chunks(self, text, *args, **kwargs):
    """tiktoken is quadratic on one long run of characters (a 200 KB line took 12 s to count).
    Counting in slices keeps the cost linear. Aider's token counts are estimates anyway."""
    if len(text) <= 2 * TOKEN_CHUNK:
        return _orig_encode(self, text, *args, **kwargs)
    tokens = []
    for start in range(0, len(text), TOKEN_CHUNK):
        tokens.extend(_orig_encode(self, text[start:start + TOKEN_CHUNK], *args, **kwargs))
    return tokens


_tiktoken_core.Encoding.encode = _encode_in_chunks

_out_lock = threading.Lock()
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
_tls = threading.local()


def clean_text(text):
    """Replace lone surrogates, which cannot be encoded as UTF-8."""
    return _LONE_SURROGATE.sub("�", text)


def clean(obj):
    """Copy of obj with every string cleaned, so the whole structure can be sent and sent to a model."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, dict):
        return {(clean_text(k) if isinstance(k, str) else k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clean(v) for v in obj)
    return obj


def send(obj):
    data = (json.dumps(clean(obj), ensure_ascii=False) + "\n").encode("utf-8", "replace")
    with _out_lock:
        _OUT.write(data)
        _OUT.flush()


def current_turn():
    return getattr(_tls, "turn", None)


def _kill_quietly(proc):
    try:
        proc.kill()
    except psutil.Error:
        pass


def kill_tree(pid):
    """Kill a process and every descendant it has, found by PID. Never by image name."""
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return
    victims = {}
    for _ in range(10):
        try:
            fresh = [p for p in root.children(recursive=True) if p.pid not in victims]
        except psutil.Error:
            break
        if not fresh:
            break
        for p in fresh:
            victims[p.pid] = p
            _kill_quietly(p)
    _kill_quietly(root)
    psutil.wait_procs(list(victims.values()) + [root], timeout=5)


def run_cmd(command, verbose=False, error_print=None, cwd=None):
    """Replaces Aider's run_cmd so a cancelled turn can stop its command and the whole tree."""
    turn = current_turn()
    if turn is None or turn.cancelled():
        return 1, "not run: the turn was cancelled"
    try:
        proc = subprocess.Popen(command, shell=True, cwd=cwd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=NO_WINDOW)
    except OSError as err:
        return 1, str(err)
    if not turn.track(proc):
        kill_tree(proc.pid)
        return 1, "not run: the turn was cancelled"
    chunks = []

    def pump():
        for piece in iter(lambda: proc.stdout.read(4096), b""):
            chunks.append(piece)

    # The command is reported to the client as an execute tool call, with its output when it ends.
    sess = turn.coder.acp
    tid = "shell-" + uuid.uuid4().hex[:12]
    if sess is not None:
        sess.emit({"sessionUpdate": "tool_call", "toolCallId": tid, "title": command, "kind": "execute",
                   "status": "in_progress", "rawInput": {"command": command}}, turn)
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        while proc.poll() is None and not turn.cancelled():
            time.sleep(0.05)
        if proc.poll() is None:
            kill_tree(proc.pid)
            proc.wait(timeout=5)
    finally:
        turn.untrack(proc)
    reader.join(5)
    output = b"".join(chunks).decode("utf-8", "replace")
    if turn.cancelled():
        output += "\n[cancelled]"
    rc = proc.returncode
    if sess is not None:
        shown = output if len(output) <= SHELL_SHOW_CHARS else output[:SHELL_SHOW_CHARS] + "\n[output cut]"
        sess.emit({"sessionUpdate": "tool_call_update", "toolCallId": tid,
                   "status": "completed" if rc == 0 else "failed",
                   "content": [{"type": "content", "content": {"type": "text",
                                "text": shown or "(no output)"}}],
                   "rawOutput": {"exitCode": rc}}, turn)
    if turn.cancelled():
        return 1, output
    return rc, output


base_coder.run_cmd = run_cmd


class BlockFilter:
    """Cuts Aider's SEARCH/REPLACE blocks out of the streamed answer text, piece by piece.

    A block is: a filename line, then a SEARCH line, the search and replace lines, a REPLACE line,
    and optionally a closing fence. Everything else is prose and is streamed as it arrives. A prose
    line with no space in it may be the filename, so it is held back until the next line shows
    whether a SEARCH line follows; a block's filename is then dropped, never shown.
    feed() returns the text to show now; finish() returns what is left at the end of the stream.
    A block cut off by a cancel leaves no text."""

    def __init__(self):
        self.mode = "prose"       # prose | edit (inside a block) | after (just past REPLACE)
        self.buf = ""            # text not yet decided, or the rest of a line being dropped
        self.committed = False   # the current line is prose and its text is already shown
        self.dropping = False    # the rest of the current line is a fence after a block: dropped
        self.candidates = []     # held-back single-word prose lines, oldest first

    def _release(self):
        out, self.candidates = self.candidates, []
        return out

    def feed(self, piece):
        out = []
        self.buf += piece
        while True:
            if self.dropping:
                nl = self.buf.find("\n")
                if nl < 0:
                    self.buf = ""
                    break
                self.buf = self.buf[nl + 1:]
                self.dropping = False
                continue
            if self.mode == "edit":
                nl = self.buf.find("\n")
                if nl < 0:
                    break
                line, self.buf = self.buf[:nl], self.buf[nl + 1:]
                if REPLACE_LINE.match(line):
                    self.mode = "after"
                continue
            if self.committed:
                nl = self.buf.find("\n")
                if nl < 0:
                    out.append(self.buf)
                    self.buf = ""
                    break
                out.append(self.buf[:nl + 1])
                self.buf = self.buf[nl + 1:]
                self.committed = False
                continue
            nl = self.buf.find("\n")
            line = self.buf if nl < 0 else self.buf[:nl]
            if self.mode == "after":
                if line.startswith("```"):
                    self.mode = "prose"
                    self.dropping = True       # the fence that closes the block
                    continue
                if nl < 0 and "```".startswith(line):
                    break                      # may still become the closing fence
                self.mode = "prose"
            if nl < 0:
                if not line or SEARCH_PREFIX.fullmatch(line):
                    break                      # could still be a SEARCH line
                if " " in line or "\t" in line:
                    out.extend(self._release())
                    self.committed = True      # clearly prose: show it as it streams
                    continue
                break                          # one word so far: could be a filename
            self.buf = self.buf[nl + 1:]
            if SEARCH_LINE.match(line):
                self.candidates = []           # the line before a SEARCH line was its filename
                self.mode = "edit"
                continue
            text = line + "\n"
            if line.strip() and not any(c.isspace() for c in line) and len(line) <= 300:
                self.candidates.append(text)
                if len(self.candidates) > MAX_CANDIDATES:
                    out.append(self.candidates.pop(0))
            else:
                out.extend(self._release())
                out.append(text)
        return [s for s in out if s]

    def finish(self):
        """End of stream: show what is still held, unless it is inside a block or a block marker."""
        out = []
        if self.mode != "edit" and not self.dropping:
            out.extend(self._release())
            if self.buf and not (self.mode == "after" or SEARCH_LINE.match(self.buf)):
                out.append(self.buf)
        self.buf, self.candidates, self.mode, self.dropping = "", [], "prose", False
        return [s for s in out if s]


class InlineThink:
    """Splits content text at think tags (the reasoning tag names), which may be cut across pieces.
    feed() and flush() return (is_reasoning, text) pairs."""

    def __init__(self, names):
        self.tags = [f"<{n}>" for n in names] + [f"</{n}>" for n in names]
        self.opens = {f"<{n}>" for n in names}
        self.buf = ""
        self.inside = False

    def feed(self, piece):
        s, self.buf = self.buf + piece, ""
        out, i = [], 0
        while True:
            hits = [(s.find(t, i), t) for t in self.tags if s.find(t, i) >= 0]
            if not hits:
                break
            j, t = min(hits, key=lambda h: (h[0], -len(h[1])))
            if j > i:
                out.append((self.inside, s[i:j]))
            i = j + len(t)
            self.inside = t in self.opens
        rest = s[i:]
        longest = max((len(t) for t in self.tags), default=1) - 1
        for k in range(min(len(rest), longest), 0, -1):
            if any(t.startswith(rest[-k:]) for t in self.tags):
                self.buf = rest[-k:]
                rest = rest[:-k]
                break
        if rest:
            out.append((self.inside, rest))
        return out

    def flush(self):
        out = [(self.inside, self.buf)] if self.buf else []
        self.buf = ""
        return out


class Turn:
    """One prompt. The client is answered once: by the worker, or by cancel, whichever is first."""

    def __init__(self, rid, coder):
        self.rid = rid
        self.coder = coder
        self.answered = False                # guarded by Session.lock
        self.shell_allowed = False           # the client allowed a shell command not yet reported
        self.finished = threading.Event()    # the worker thread has returned
        self.last_error = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._procs = []

    def cancelled(self):
        return self._cancel.is_set()

    def cancel(self):
        with self._lock:
            self._cancel.set()
            procs, self._procs = self._procs, []
        for proc in procs:
            threading.Thread(target=kill_tree, args=(proc.pid,), daemon=True).start()

    def track(self, proc):
        """Register a child process. False if the turn was already cancelled (caller must kill it)."""
        with self._lock:
            if self.cancelled():
                return False
            self._procs.append(proc)
            return True

    def untrack(self, proc):
        with self._lock:
            if proc in self._procs:
                self._procs.remove(proc)


def _model_choices():
    """Map each config value (bare model id) to the full Aider model name."""
    choices = {}
    for name in ("AIDER_MODEL", "AIDER_MODEL_2"):
        full = (os.environ.get(name) or "").strip()
        if not full:
            continue
        bare = full.split("/", 1)[-1]
        if bare in choices and choices[bare] != full:
            bare = full
        choices[bare] = full
    if not choices:
        choices[DEFAULT_MODEL.split("/", 1)[-1]] = DEFAULT_MODEL
    return choices


class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _read_file(path):
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            return f.read()
    except OSError:
        return ""


LIST_SKIP_DIRS = {"node_modules", "__pycache__", "site-packages"}
LIST_BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".7z",
                   ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyd", ".class", ".jar", ".wasm", ".bin",
                   ".mp3", ".mp4", ".mov", ".wav", ".ttf", ".otf", ".woff", ".woff2", ".sqlite", ".db"}
LIST_MAX_FILES = 500          # files offered to the model at most
LIST_MAX_VISITED = 5000       # directory entries looked at, so a huge tree stops the walk
LIST_MAX_BYTES = 256 * 1024   # bigger files are not offered


def inside_root(root, path):
    """True if path (absolute) is root or below it. Symlinks are resolved first."""
    base = os.path.normcase(os.path.realpath(root))
    full = os.path.normcase(os.path.realpath(path))
    try:
        return os.path.commonpath([base, full]) == base
    except ValueError:  # different drives
        return False


def _looks_binary(full):
    try:
        with open(full, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def list_session_files(root):
    """Relative paths of text files under root that Aider may offer. Skips hidden entries, .git,
    node_modules, venvs (a pyvenv.cfg marks one), binary-looking and oversized files, symlinks that
    leave the folder. Bounded by LIST_MAX_FILES and LIST_MAX_VISITED."""
    out, visited = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in sorted(dirnames):
            if d.startswith(".") or d in LIST_SKIP_DIRS or os.path.isfile(os.path.join(dirpath, d, "pyvenv.cfg")):
                continue
            keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            visited += 1
            if visited > LIST_MAX_VISITED or len(out) >= LIST_MAX_FILES:
                return sorted(out)
            full = os.path.join(dirpath, name)
            if name.startswith(".") or os.path.splitext(name)[1].lower() in LIST_BINARY_EXT:
                continue
            if os.path.islink(full) and not inside_root(root, full):
                continue
            try:
                if not os.path.isfile(full) or os.path.getsize(full) > LIST_MAX_BYTES:
                    continue
            except OSError:
                continue
            if _looks_binary(full):
                continue
            out.append(os.path.relpath(full, root))
    return sorted(out)


class AcpIO(InputOutput):
    """Aider's InputOutput with no console output. Permission asks go to the ACP client."""

    def __init__(self, session):
        super().__init__(pretty=False, fancy_input=False, yes=False)
        self.session = session

    def confirm_ask(self, question, default="y", subject=None, explicit_yes_required=False,
                    group=None, allow_never=False):
        turn = current_turn()
        if question == "Add command output to the chat?":
            # Aider's second shell question. Only the first question is a client permission. It is
            # answered yes here when the client allowed the command, and never asked of the client.
            return self.session.take_shell_go_ahead(turn)
        return self.session.ask_permission(question, subject, turn)

    def offer_url(self, url, prompt="Open URL for more info?", allow_never=True):
        return False  # never open a browser

    def tool_output(self, *messages, log_only=False, bold=False):
        pass

    def tool_warning(self, message="", strip=True):
        pass

    def tool_error(self, message="", strip=True):
        pass

    def assistant_output(self, message, pretty=None):
        pass

    def ai_output(self, content):
        pass

    def llm_started(self):
        pass

    def ring_bell(self):
        pass

    def print(self, message=""):
        pass


class AcpCoder(EditBlockCoder):
    """Aider's SEARCH/REPLACE coder, with ACP hooks for streaming edits and send errors."""

    acp = None        # the Session
    acp_turn = None   # the Turn this coder is serving

    def keyboard_interrupt(self):
        pass  # never exit the adapter; cancel is handled by the ACP session

    def get_all_relative_files(self):
        """Files Aider may offer to add: real files under the session folder, bounded.

        Aider has no git repo here, so its default would list only the files already in the chat,
        and a mention of hello.txt could never match. Adding a file is still never automatic:
        check_for_file_mentions asks the client first (ask_permission), like every other edit."""
        return list_session_files(self.root)

    def allowed_to_edit(self, path):
        # Edits may only touch files inside the session folder, even when the model names another path.
        if not inside_root(self.root, self.abs_root_path(path)):
            return None
        return super().allowed_to_edit(path)

    def _stale(self):
        return self.acp_turn is not None and self.acp_turn.cancelled()

    def send(self, messages, model=None, functions=None):
        turn = self.acp_turn
        if turn is not None:
            if turn.cancelled():
                raise KeyboardInterrupt()  # a stale turn makes no more model requests
            turn.last_error = None
        try:
            yield from super().send(clean(messages), model=model, functions=functions)
        except KeyboardInterrupt:
            raise
        except Exception as err:
            if turn is not None:
                turn.last_error = err
            raise

    def calculate_and_show_tokens_and_cost(self, messages, completion=None):
        """Aider's own prompt-token count for this request, sent as usage_update (context in use).
        Sent only when Aider knows the model's input window, since the update carries the size."""
        before = self.message_tokens_sent
        super().calculate_and_show_tokens_and_cost(messages, completion)
        used = self.message_tokens_sent - before
        window = (getattr(self.main_model, "info", None) or {}).get("max_input_tokens") or 0
        turn = self.acp_turn
        if self.acp is not None and turn is not None and used > 0 and window > 0:
            self.acp.emit({"sessionUpdate": "usage_update", "used": int(used), "size": int(window)}, turn)

    def show_send_output_stream(self, completion):
        """Aider's stream loop, yielding typed pieces: ("thought", text) for reasoning and
        ("text", text) for the answer, with SEARCH/REPLACE blocks cut out. partial_response_content
        still gets the raw text with Aider's reasoning tags, because Aider parses edits from it."""
        reasoning_tags = {"think", self.main_model.reasoning_tag} - {None, ""}
        inline = InlineThink(reasoning_tags)
        blocks = BlockFilter()
        received_content = False

        def typed(pairs):
            events = []
            for is_reasoning, seg in pairs:
                if is_reasoning:
                    events.append(("thought", seg))
                else:
                    events.extend(("text", s) for s in blocks.feed(seg))
            return events

        for chunk in completion:
            if len(chunk.choices) == 0:
                continue
            if getattr(chunk.choices[0], "finish_reason", None) == "length":
                raise base_coder.FinishReasonLength()
            delta = chunk.choices[0].delta
            try:
                func = delta.function_call
                for k, v in func.items():
                    if k in self.partial_response_function_call:
                        self.partial_response_function_call[k] += v
                    else:
                        self.partial_response_function_call[k] = v
                received_content = True
            except AttributeError:
                pass

            text, events = "", []
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content is None:
                reasoning_content = getattr(delta, "reasoning", None)
            if reasoning_content:
                if not self.got_reasoning_content:
                    text += f"<{REASONING_TAG}>\n\n"
                text += reasoning_content
                self.got_reasoning_content = True
                received_content = True
                events.append(("thought", reasoning_content))
            content = getattr(delta, "content", None)
            if content:
                if self.got_reasoning_content and not self.ended_reasoning_content:
                    text += f"\n\n</{self.reasoning_tag_name}>\n\n"
                    self.ended_reasoning_content = True
                text += content
                received_content = True
                events.extend(typed(inline.feed(content)))
            if received_content:
                self._stop_waiting_spinner()
            self.partial_response_content += text
            for ev in events:
                yield ev

        for ev in typed(inline.flush()):
            yield ev
        for s in blocks.finish():
            yield ("text", s)
        if not received_content:
            self.io.tool_warning("Empty response received from LLM. Check your provider account?")

    def apply_edits(self, edits, dry_run=False):
        if self._stale():
            return None  # a cancelled turn must not touch files
        edits = clean(edits)
        if dry_run or not edits or self.acp is None:
            return super().apply_edits(edits, dry_run=dry_run)
        turn = self.acp_turn
        rows = []
        for path, _original, _updated in edits:
            full = self.abs_root_path(path)
            existed = os.path.isfile(full)
            before = _read_file(full) if existed else ""
            tid = "edit-" + uuid.uuid4().hex[:12]
            self.acp.emit({"sessionUpdate": "tool_call", "toolCallId": tid, "title": f"Edit {path}",
                           "kind": "edit", "status": "pending", "locations": [{"path": full}]}, turn)
            rows.append((tid, full, existed, before))
        for tid, full, _e, _b in rows:
            self.acp.emit({"sessionUpdate": "tool_call_update", "toolCallId": tid,
                           "status": "in_progress", "locations": [{"path": full}]}, turn)
        try:
            return super().apply_edits(edits, dry_run=dry_run)
        finally:
            for tid, full, existed, before in rows:
                after_exists = os.path.isfile(full)
                after = _read_file(full) if after_exists else ""
                if after_exists != existed or after != before:
                    self.acp.emit({"sessionUpdate": "tool_call_update", "toolCallId": tid,
                                   "status": "completed", "locations": [{"path": full}],
                                   "content": [{"type": "diff", "path": full,
                                                "oldText": before if existed else None,
                                                "newText": after}]}, turn)
                else:
                    self.acp.emit({"sessionUpdate": "tool_call_update", "toolCallId": tid,
                                   "status": "failed", "locations": [{"path": full}],
                                   "content": [{"type": "content", "content": {"type": "text",
                                                "text": "No change: the SEARCH block did not match."}}]},
                                  turn)


class Session:
    def __init__(self, server, sid, cwd):
        self.server = server
        self.sid = sid
        self.cwd = cwd
        self.lock = threading.RLock()
        self.turn = None      # the prompt the client is waiting on, if any
        self.live = []        # turns whose worker thread may still run
        self.models = _model_choices()
        self.model_value = next(iter(self.models))
        self.effort = DEFAULT_EFFORT
        self.io = AcpIO(self)
        self.coder = self.new_coder(self.build_model())

    def new_coder(self, main_model, history=None):
        coder = AcpCoder(main_model=main_model, io=self.io, use_git=False, fnames=[], stream=True,
                         auto_lint=False, auto_test=False, suggest_shell_commands=True, map_tokens=0,
                         detect_urls=False)
        coder.root = self.cwd
        coder.acp = self
        if history is not None:
            coder.done_messages = list(history)
        return coder

    def build_model(self):
        m = models.Model(self.models[self.model_value])
        m.set_reasoning_effort(self.effort)
        return m

    def apply_model(self):
        self.coder.main_model = self.build_model()

    def config_options(self):
        return [
            {"id": "model", "name": "Model", "category": "model", "type": "select",
             "currentValue": self.model_value,
             "options": [{"value": k, "name": v} for k, v in self.models.items()]},
            {"id": "effort", "name": "Reasoning effort", "category": "thought_level", "type": "select",
             "currentValue": self.effort,
             "options": [{"value": e, "name": e} for e in EFFORTS]},
        ]

    def start_turn(self, rid):
        """Caller holds self.lock and has checked that no turn is waiting for the client."""
        self.live = [t for t in self.live if not t.finished.is_set()]
        if len(self.live) >= MAX_LIVE_TURNS:
            raise RpcError(-32000, "earlier cancelled turns are still stopping; retry in a few seconds")
        coder = self.coder
        if coder.acp_turn is not None and not coder.acp_turn.finished.is_set():
            # A cancelled turn's worker still holds this coder. Continue on a fresh coder with the history.
            coder = self.new_coder(coder.main_model, history=coder.done_messages)
            self.coder = coder
        turn = Turn(rid, coder)
        coder.acp_turn = turn
        self.turn = turn
        self.live.append(turn)
        return turn

    def cancel_current(self):
        with self.lock:
            turn = self.turn
            if turn is None:
                return
            self.turn = None
            turn.answered = True
            turn.cancel()
            self.server.reply(turn.rid, {"stopReason": "cancelled"})

    def emit(self, update, turn=None):
        with self.lock:  # one lock for updates and replies, so no update follows its turn's reply
            if turn is not None and turn.cancelled():
                return
            send({"jsonrpc": "2.0", "method": "session/update",
                  "params": {"sessionId": self.sid, "update": update}})

    def emit_text(self, text, turn):
        self.emit({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}, turn)

    def emit_thought(self, text, turn):
        self.emit({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": text}}, turn)

    def take_shell_go_ahead(self, turn):
        """Aider's 'Add command output' question: yes only if the client allowed the shell command
        this turn. Consumed, so it answers one command's question once."""
        if turn is None:
            return False
        allowed, turn.shell_allowed = turn.shell_allowed, False
        return allowed

    def ask_permission(self, question, subject, turn):
        """Block the turn's worker until the client answers. Deny when the turn is cancelled."""
        if turn is None or turn.cancelled():
            return False
        shell = question.lower().startswith("run shell")
        title = question if not subject else f"{question}\n{subject}"
        tid = "perm-" + uuid.uuid4().hex[:12]
        ev, box = self.server.request("session/request_permission", {
            "sessionId": self.sid,
            "toolCall": {"toolCallId": tid, "title": title, "kind": "execute" if shell else "edit",
                         "status": "pending"},
            "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "reject", "name": "Deny", "kind": "reject_once"}]})
        try:
            while not ev.wait(0.1):
                if turn.cancelled():
                    return False
        finally:
            self.server.forget(box)
        if "error" in box.get("msg", {}):
            allowed = False
        else:
            outcome = (box.get("msg", {}).get("result") or {}).get("outcome") or {}
            allowed = outcome.get("outcome") == "selected" and outcome.get("optionId") == "allow"
        if shell:
            turn.shell_allowed = allowed
        return allowed


class Server:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.pending = {}
        self.next_req = 0

    # --- agent -> client requests ---
    def request(self, method, params):
        with self.lock:
            self.next_req += 1
            rid = f"agent-{self.next_req}"
            ev = threading.Event()
            box = {"id": rid}
            self.pending[rid] = (ev, box)
        send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return ev, box

    def forget(self, box):
        with self.lock:
            self.pending.pop(box.get("id"), None)

    def on_response(self, msg):
        with self.lock:
            entry = self.pending.pop(msg.get("id"), None)
        if entry:
            ev, box = entry
            box["msg"] = msg
            ev.set()

    # --- client -> agent ---
    def reply(self, rid, result=None, error=None):
        if rid is None:
            return
        if error is not None:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": error.code, "message": error.message}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "result": result})

    def session_for(self, params):
        sess = self.sessions.get((params or {}).get("sessionId"))
        if sess is None:
            raise RpcError(-32602, "unknown sessionId")
        return sess

    def dispatch(self, method, params, rid):
        if method == "initialize":
            return {"protocolVersion": PROTOCOL_VERSION,
                    "agentCapabilities": {"loadSession": False,
                                          "promptCapabilities": {"image": False, "audio": False,
                                                                 "embeddedContext": False}},
                    "authMethods": [],
                    "agentInfo": {"name": "aider-acp", "title": "Aider", "version": "0.86.2"}}
        if method == "session/new":
            sid = uuid.uuid4().hex
            cwd = (params or {}).get("cwd") or os.getcwd()
            sess = Session(self, sid, cwd)
            self.sessions[sid] = sess
            return {"sessionId": sid, "configOptions": sess.config_options()}
        if method == "session/prompt":
            sess = self.session_for(params)
            text = "\n".join(b.get("text", "") for b in (params.get("prompt") or [])
                             if isinstance(b, dict) and b.get("type") == "text")
            text = clean_text(text)
            with sess.lock:
                if sess.turn is not None:
                    raise RpcError(-32000, "a turn is already running in this session")
                turn = sess.start_turn(rid)
            threading.Thread(target=self.run_prompt, args=(sess, turn, text), daemon=True).start()
            return NO_REPLY
        if method == "session/cancel":
            sess = self.sessions.get((params or {}).get("sessionId"))
            if sess is not None:
                sess.cancel_current()
            return {}
        if method == "session/set_config_option":
            sess = self.session_for(params)
            if sess.turn is not None:
                raise RpcError(-32000, "cannot change config while a turn is running")
            cfg, value = (params or {}).get("configId"), (params or {}).get("value")
            if cfg == "model":
                if value not in sess.models:
                    raise RpcError(-32602, f"unknown model {value!r}")
                sess.model_value = value
                sess.apply_model()
            elif cfg == "effort":
                if value not in EFFORTS:
                    raise RpcError(-32602, f"unknown effort {value!r}")
                sess.effort = value
                sess.apply_model()
            else:
                raise RpcError(-32602, f"unknown configId {cfg!r}")
            return {"configOptions": sess.config_options()}
        raise RpcError(-32601, f"method not found: {method}")

    def finish(self, sess, turn, result=None, error=None):
        with sess.lock:
            if turn.answered:
                return  # cancel already answered this prompt
            turn.answered = True
            if sess.turn is turn:
                sess.turn = None
            self.reply(turn.rid, result, error=error)

    def run_prompt(self, sess, turn, text):
        _tls.turn = turn
        try:
            stop = self.run_turn(sess, turn, text)
            self.finish(sess, turn, result={"stopReason": stop})
        except RpcError as err:
            self.finish(sess, turn, error=err)
        except Exception as err:  # keep the adapter alive on bugs
            traceback.print_exc(file=sys.stderr)
            self.finish(sess, turn, error=RpcError(-32603, f"adapter error: {err}"))
        finally:
            turn.finished.set()

    def run_turn(self, sess, turn, text):
        coder = turn.coder
        coder.init_before_message()
        message = text
        first = True
        while message:
            coder.reflected_message = None
            if first:
                # Files the prompt names (Aider's own mention flow). Each one asks the client first.
                first = False
                note = coder.check_for_file_mentions(message)
                if turn.cancelled():
                    return "cancelled"
                if note:
                    message = message + "\n\n" + note
            if self.drain(sess, turn, coder.send_message(message)):
                return "cancelled"
            if turn.last_error is not None:
                err = turn.last_error
                key = os.environ.get("OPENAI_API_KEY") or ""
                detail = str(err)
                if key:
                    detail = detail.replace(key, "***")
                raise RpcError(-32000, f"model request failed: {type(err).__name__}: {detail[:500]}")
            if turn.cancelled():
                return "cancelled"
            if not coder.reflected_message:
                break
            if coder.num_reflections >= coder.max_reflections:
                break
            coder.num_reflections += 1
            message = coder.reflected_message
        return "cancelled" if turn.cancelled() else "end_turn"

    def drain(self, sess, turn, gen):
        """Stream the generator's text as chunks. Returns True if the turn was cancelled."""
        while True:
            if turn.cancelled():
                try:
                    gen.throw(KeyboardInterrupt())  # Aider's own interrupt path records the partial reply
                except (StopIteration, KeyboardInterrupt):
                    pass
                return True
            try:
                piece = next(gen)
            except StopIteration:
                return turn.cancelled()
            kind, text = piece
            if text:
                if kind == "thought":
                    sess.emit_thought(text, turn)
                else:
                    sess.emit_text(text, turn)


def main():
    server = Server()
    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        if not isinstance(msg, dict):
            continue
        if "method" not in msg:
            if "id" in msg:
                server.on_response(msg)
            continue
        method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        try:
            result = server.dispatch(method, params, rid)
        except RpcError as err:
            server.reply(rid, error=err)
            continue
        except Exception as err:
            traceback.print_exc(file=sys.stderr)
            server.reply(rid, error=RpcError(-32603, f"adapter error: {err}"))
            continue
        if result is NO_REPLY or rid is None:
            continue
        server.reply(rid, result)


if __name__ == "__main__":
    main()
