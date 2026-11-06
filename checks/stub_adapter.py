"""stub_adapter.py: a fake stdio ACP adapter that only does handshake and canned end_turn.

Used to prove check_adapter.py cannot be fooled: it must FAIL most probes. It calls no model,
edits no files, asks no permission and does not support config options.
"""
import json
import sys

SESSION = "stub-session-1"


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": SESSION}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": SESSION, "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Canned reply from the stub."}}}})
        send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
    elif rid is not None and method is not None:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}})
