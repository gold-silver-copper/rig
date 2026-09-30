#!/usr/bin/env python3
"""Extend a running bevy-agent over the Bevy Remote Protocol.

Registers a `secret_number` tool, then serves every call the model makes to
it until interrupted. Usage: brp_extension.py PORT [--prompt TEXT]
"""

import json
import sys
import urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/"


def request(method, params=None, id=1):
    body = json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).encode()
    return urllib.request.Request(url, data=body, headers={"content-type": "application/json"})


def call(method, params=None):
    with urllib.request.urlopen(request(method, params)) as response:
        return json.load(response)


print(call("agent.register_tool", {
    "name": "secret_number",
    "description": "Returns the secret number. Call it whenever the user asks for the secret number.",
    "parameters": {"type": "object", "properties": {}},
}))

if "--prompt" in sys.argv:
    print(call("agent.prompt", {"text": sys.argv[sys.argv.index("--prompt") + 1]}))

# A `+watch` method streams one server-sent event per new call.
with urllib.request.urlopen(request("agent.tool_calls+watch", {"name": "secret_number"}, id=2)) as stream:
    for raw in stream:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: "):])
        tool_call = event["result"]
        print("call:", tool_call)
        print(call("agent.tool_result", {"call_id": tool_call["call_id"], "content": "The secret number is 4217."}))
