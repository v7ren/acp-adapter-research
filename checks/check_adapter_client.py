"""check_adapter_client.py: ACP stdio client and process-tree helpers for check_adapter.py.

Stdio client and process-tree helpers, written for this checker.
Every process is started by this module and stopped by PID only, never by image name.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque

NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class Fail(Exception):
    pass


class Skip(Exception):
    pass


def process_table():
    """parent pid -> [child pids], for every process on the machine."""
    if sys.platform == "win32":
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process | ForEach-Object { $_.ProcessId.ToString() + ',' + $_.ParentProcessId.ToString() }"],
                             capture_output=True, text=True, creationflags=NO_WINDOW, timeout=60).stdout
        pairs = []
        for ln in out.splitlines():
            a, _, b = ln.strip().partition(",")
            if a.isdigit() and b.isdigit():
                pairs.append((int(a), int(b)))
    else:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
        pairs = [(int(f[0]), int(f[1])) for f in (ln.split() for ln in out.splitlines()) if len(f) == 2]
    kids = {}
    for child, parent in pairs:
        kids.setdefault(parent, []).append(child)
    return kids


def descendants(pid, kids):
    found, stack = [], [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    return found


def pid_alive(pid):
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             creationflags=NO_WINDOW, timeout=15).stdout
        return str(pid) in out.split()
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid):
    """Stop one PID this checker started. Never by image name."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=NO_WINDOW, timeout=15)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def kill_tree(pids):
    """Kill the given PIDs (a tree snapshot taken earlier) that are still alive."""
    for pid in pids:
        if pid_alive(pid):
            kill_pid(pid)


class Adapter:
    """One adapter process started through the shell, with a JSON-RPC client on its pipes."""

    def __init__(self, cmd, cwd, env):
        self.p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                  bufsize=1, cwd=cwd, env=env, creationflags=NO_WINDOW)
        self.cv = threading.Condition()
        self.wlock = threading.Lock()
        self.next_id = 0
        self.responses, self.resp_time = {}, {}
        self.updates = []        # (session/update params.update, monotonic time)
        self.perms = []          # session/request_permission params, one per ask
        self.perm_answers = []   # optionId given, or "cancelled"
        self.raw = []            # every stdout line as received
        self.bad_lines = []      # stdout lines that are not JSON-RPC 2.0 objects
        self.err = deque(maxlen=200)
        self.answer = "allow"    # "allow", "reject", or None to answer "cancelled"
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _send(self, obj):
        with self.wlock:
            self.p.stdin.write(json.dumps(obj) + "\n")
            self.p.stdin.flush()

    def _choose(self, params):
        if self.answer is None:
            return {"outcome": "cancelled"}, "cancelled"
        want = "allow" if self.answer == "allow" else "reject"
        opts = [o for o in (params.get("options") or []) if isinstance(o, dict)]
        pick = next((o for o in opts if str(o.get("kind", "")).startswith(want)), None)
        if pick is None:
            pick = next((o for o in opts if want in str(o.get("optionId", "")).lower()), None)
        if pick is None:
            return {"outcome": "cancelled"}, "cancelled"
        return {"outcome": "selected", "optionId": pick.get("optionId")}, pick.get("optionId")

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
                    self.updates.append((params.get("update") or {}, now))
                elif method == "session/request_permission":
                    self.perms.append(params)
                    outcome, chosen = self._choose(params)
                    self.perm_answers.append(chosen)
                    self._send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {"outcome": outcome}})
                elif "id" in msg and method is None:
                    self.responses[msg["id"]] = msg
                    self.resp_time[msg["id"]] = now
                self.cv.notify_all()

    def _read_err(self):
        for line in self.p.stderr:
            with self.cv:
                self.err.append(line.rstrip())

    def request(self, method, params):
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        return self.next_id

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def wait(self, rid, timeout):
        with self.cv:
            if not self.cv.wait_for(lambda: rid in self.responses, timeout):
                raise TimeoutError(f"no reply to request {rid} within {timeout} s")
            return self.responses.pop(rid)

    def wait_chunk(self, timeout):
        with self.cv:
            return self.cv.wait_for(lambda: any(u.get("sessionUpdate") == "agent_message_chunk"
                                                for u, _ in self.updates), timeout)

    def reset(self):
        with self.cv:
            self.updates = []

    def snapshot(self):
        with self.cv:
            return list(self.updates)

    def raw_lines(self):
        with self.cv:
            return list(self.raw)

    def last_line(self):
        with self.cv:
            return self.raw[-1][:160] if self.raw else "(no stdout)"

    def stderr_tail(self):
        with self.cv:
            return " | ".join(list(self.err)[-3:])[:300]

    def adapter_pid(self):
        """The adapter is the direct child of the shell that started it (or the shell if none)."""
        kids = process_table().get(self.p.pid, [])
        return kids[0] if kids else self.p.pid

    def tree_snapshot(self):
        return [self.p.pid] + descendants(self.p.pid, process_table())

    def close(self):
        pids = self.tree_snapshot()
        try:
            self.p.stdin.close()
            self.p.wait(2)
        except Exception:
            pass
        kill_tree(pids)  # every process this adapter started, by PID
