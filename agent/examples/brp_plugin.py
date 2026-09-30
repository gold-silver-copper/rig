#!/usr/bin/env python3
"""An external rig-pi plugin over BRP: offers the model a `reverse` tool.

    python3 examples/brp_plugin.py PORT

It opens a `rig_pi.serve+watch` stream, which registers its tools and then
delivers the model's calls, and answers each with `rig_pi.tool_result`. When
the agent reloads the stream ends; the plugin reconnects to the same port
with a fresh session id and registers again.
"""

import json
import sys
import time
import urllib.request
import uuid

URL = f"http://127.0.0.1:{int(sys.argv[1])}/"
PLUGIN = "py-reverse"
TOOLS = [
    {
        "name": "reverse",
        "description": "Reverse a string.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }
]


def request(method, params):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    return urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )


def run(call):
    if call["name"] == "reverse":
        return call["arguments"]["text"][::-1]
    raise ValueError(f"unknown tool {call['name']}")


def serve():
    session = uuid.uuid4().hex
    with urllib.request.urlopen(request("rig_pi.serve+watch", {
        "plugin": PLUGIN, "session": session, "tools": TOOLS,
    })) as stream:
        for line in stream:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            message = json.loads(line[len("data: "):])
            if "error" in message:
                raise RuntimeError(message["error"])
            call = message["result"]
            if "registered" in call:
                print("registered", call["registered"], flush=True)
                continue
            print("call", call, flush=True)
            try:
                answer = {"call_id": call["call_id"], "output": run(call)}
            except Exception as error:  # the model sees the error text
                answer = {"call_id": call["call_id"], "error": str(error)}
            urllib.request.urlopen(request("rig_pi.tool_result", answer), timeout=10).read()


while True:
    try:
        serve()
        print("stream ended", flush=True)
    except Exception as error:
        print("disconnected:", error, flush=True)
    time.sleep(0.5)
