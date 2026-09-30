#!/usr/bin/env python3
"""An external bevy-agent plugin, speaking only BRP (JSON-RPC over HTTP).

It offers the model a `fingerprint` tool: the first 12 hex digits of the
SHA-256 of a text. It survives agent reloads: when the agent restarts, the
watch stream ends, and the plugin reconnects to the same port and registers
again (registration is idempotent), after which unanswered calls are resent.

    python3 brp_plugin.py PORT
"""

import hashlib
import http.client
import json
import sys
import time

PORT = int(sys.argv[1])
PLUGIN = "fingerprint-plugin"
TOOL = {
    "plugin": PLUGIN,
    "name": "fingerprint",
    "description": "The first 12 hex digits of the SHA-256 of `text`.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def post(method, params):
    """Open a BRP request and return the HTTP response, unread."""
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=None)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    connection.request("POST", "/", body, {"Content-Type": "application/json"})
    return connection.getresponse()


def call(method, params):
    reply = json.load(post(method, params))
    if "error" in reply:
        raise RuntimeError(f"{method}: {reply['error']}")
    return reply["result"]


def serve():
    call("agent.register_tool", TOOL)
    log("registered `fingerprint`; watching for calls")
    stream = post("agent.tool_calls+watch", {"plugin": PLUGIN})
    while True:
        line = stream.readline()
        if not line:
            raise ConnectionError("watch stream closed")
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        for request in json.loads(line[len("data: "):]).get("result", []):
            text = request["arguments"].get("text", "")
            output = hashlib.sha256(text.encode()).hexdigest()[:12]
            call("agent.tool_result", {"call_id": request["call_id"], "output": output})
            log(f"fingerprint({text!r}) = {output}")


while True:
    try:
        serve()
    except (OSError, ConnectionError, RuntimeError, ValueError) as error:
        log(f"disconnected ({error}); reconnecting")
        time.sleep(0.5)
