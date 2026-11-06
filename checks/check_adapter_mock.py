"""check_adapter_mock.py: scripted OpenAI-style chat server for check_adapter.py.

Localhost only, random free port. Each reply is chosen by a PROBE-* token in the last user
message (see reply_for). Every request body is recorded so probes can check what the adapter sent.
Extends a plain chat-mock handler: per-probe scripted replies instead of
the fixed set, and the full request body kept.
"""
import http.server
import json
import re
import threading
import time

# Aider SEARCH/REPLACE blocks. An empty SEARCH creates a new file.
AIDER_EDIT = "hello.txt\n<<<<<<< SEARCH\nhello\n=======\nhi\n>>>>>>> REPLACE\n"
AIDER_CREATE = "new.txt\n<<<<<<< SEARCH\n=======\nhi\n>>>>>>> REPLACE\n"
SHELL = "Here is the command:\n\n```bash\necho marked > marker.txt\n```\n"
STREAM = "The quick brown fox jumps over the lazy dog. "

# Reply packs. Each says how the mock answers in the format one agent reads from a model:
#   edit(): reply that changes hello.txt from "hello" to "hi" (the mock picks it only when the
#           marker is in the request);  create(name): reply that creates `name` with "hi";
#   shell(cmd): reply that asks the agent to run `cmd`.
# "aider" is the default and its replies are byte-identical to the ones this file always sent.
PACKS = {
    "aider": {
        "label": "Aider SEARCH/REPLACE blocks in plain text",
        "edit": AIDER_EDIT,
        "create": lambda name: f"{name}\n<<<<<<< SEARCH\n=======\nhi\n>>>>>>> REPLACE\n",
        "shell": lambda cmd: f"Here is the command:\n\n```bash\n{cmd}\n```\n",
    },
}


def _pieces(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def _last_user(msgs):
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def _all_text(msgs):
    """Every text the request carried: all roles and content parts, file bodies included."""
    parts = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        parts.append(str(c or ""))
    return "\n".join(parts)


def reply_for(msgs, marker=None, pack=None):
    """Return (pieces, seconds between pieces, http status) for the scripted reply.

    PROBE-EDIT answers with the edit only when the request contains the marker that the probe
    wrote into hello.txt. The model sees that file only if the adapter sent it, so without the
    marker the reply is a plain sentence with no edit."""
    pack = pack or PACKS["aider"]
    text = _last_user(msgs)
    if "PROBE-FAIL" in text:
        return [], 0.0, 401
    if "PROBE-STREAM" in text:
        return _pieces(STREAM, 3), 0.15, 200
    if "PROBE-COUNT" in text:
        return [f"{n}\n" for n in range(1, 301)], 0.05, 200
    # Reasoning: a ("reasoning", text) piece is sent as delta.reasoning_content, a str as delta.content.
    if "PROBE-THINK-INLINE" in text:
        return _pieces("<think>Weighing the riddle first.</think>The answer is pineapple.", 3), 0.02, 200
    if "PROBE-THINK" in text:
        thought = [("reasoning", p) for p in _pieces("Weighing the riddle first. ", 3)]
        return thought + _pieces("The answer is pineapple.", 3), 0.02, 200
    if "PROBE-SHELL-OUT" in text:
        tok = re.search(r"SHOUT-[0-9a-f]+", text)
        cmd = f"echo {tok.group(0)}" if tok else "echo none"
        return _pieces(pack["shell"](cmd), 12), 0.02, 200
    if "PROBE-EDIT-PROSE" in text:
        if marker and marker in _all_text(msgs):
            reply = "Sure, I will change the word.\n" + pack["edit"] + "Done, hello.txt is updated.\n"
            return _pieces(reply, 12), 0.02, 200
        return _pieces("I cannot see that file, so I will not change it.", 12), 0.02, 200
    if "PROBE-EDIT" in text:
        if marker and marker in _all_text(msgs):
            return _pieces(pack["edit"], 12), 0.02, 200
        return _pieces("I cannot see that file, so I will not change it.", 12), 0.02, 200
    if "PROBE-CREATE" in text:
        return _pieces(pack["create"]("new.txt"), 12), 0.02, 200
    if "PROBE-SHELL" in text:
        return _pieces(pack["shell"]("echo marked > marker.txt"), 12), 0.02, 200
    if "What word did you just reply with" in text:
        said = any(m.get("role") == "assistant" and "pineapple" in str(m.get("content", "")).lower()
                   for m in msgs)
        word = "pineapple" if said else "I do not know"
        return _pieces(word, 3), 0.05, 200
    if "PROBE-TEXT" in text:
        return _pieces("pineapple", 3), 0.05, 200
    return ["ok"], 0.0, 200


class Mock:
    def __init__(self, replies="aider"):
        self.pack = PACKS[replies]
        self.requests = []
        self.marker = None  # the probe sets this to the text it wrote into hello.txt
        self._lock = threading.Lock()
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _write(self, data):
                self.wfile.write(data.encode("utf-8"))
                self.wfile.flush()

            def do_POST(self):
                size = int(self.headers.get("Content-Length", 0) or 0)
                try:
                    body = json.loads(self.rfile.read(size) or b"{}")
                except ValueError:
                    body = {}
                with mock._lock:
                    mock.requests.append(body)
                pieces, pause, status = reply_for(body.get("messages", []), mock.marker, mock.pack)
                if status != 200:
                    err = json.dumps({"error": {"message": "Incorrect API key provided: fake-key-checker",
                                                "type": "invalid_request_error",
                                                "code": "invalid_api_key"}}).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                try:
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        for i, piece in enumerate(pieces):
                            if isinstance(piece, tuple):
                                delta = {"reasoning_content": piece[1]}
                            else:
                                delta = {"role": "assistant", "content": piece} if i == 0 else {"content": piece}
                            obj = {"id": "chatcmpl-check", "object": "chat.completion.chunk", "created": 0,
                                   "model": body.get("model"),
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                            self._write(f"data: {json.dumps(obj)}\n\n")
                            time.sleep(pause)
                        last = {"id": "chatcmpl-check", "object": "chat.completion.chunk", "created": 0,
                                "model": body.get("model"),
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                        self._write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n")
                    else:
                        data = json.dumps({"id": "chatcmpl-check", "object": "chat.completion", "created": 0,
                                           "model": body.get("model"),
                                           "choices": [{"index": 0, "finish_reason": "stop",
                                                        "message": {"role": "assistant",
                                                                    "content": "".join(p for p in pieces if isinstance(p, str))}}]})
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self._write(data)
                except OSError:
                    pass  # the adapter went away (cancel or kill)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def count(self):
        with self._lock:
            return len(self.requests)

    def since(self, index):
        with self._lock:
            return list(self.requests[index:])

    def close(self):
        self.server.shutdown()
        self.server.server_close()
