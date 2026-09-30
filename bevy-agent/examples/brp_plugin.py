#!/usr/bin/env python3
"""An external bevy-agent plugin that talks to the agent over BRP.

It serves one tool, `shout`, and reconnects whenever its stream closes, as it
does when the agent reloads: re-sending `agent.serve+watch` registers the same
tool again.

    python3 brp_plugin.py <brp-port>
"""

import json
import os
import sys
import time
import urllib.request

URL = f"http://127.0.0.1:{sys.argv[1]}/"
TOOLS = [
    {
        "name": "shout",
        "description": "Upper-case `text` and mark it with the plugin's call count.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }
]
calls = 0


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    request = urllib.request.Request(URL, body.encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(request)


def shout(arguments):
    global calls
    calls += 1
    return f"{arguments['text'].upper()}! (shout #{calls} from plugin pid {os.getpid()})"


while True:
    try:
        with rpc("agent.serve+watch", {"tools": TOOLS}) as stream:
            print("registered", flush=True)
            for line in stream:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[len(b"data: ") :])
                if "error" in event:
                    print("error:", event["error"], flush=True)
                    break
                call = event["result"]
                output = shout(call["arguments"])
                with rpc("agent.tool_result", {"call_id": call["call_id"], "output": output}) as reply:
                    answer = json.load(reply)
                print(call["tool"], call["arguments"], "->", answer.get("error", "ok"), flush=True)
        print("stream closed", flush=True)
    except OSError as error:
        print("not connected:", error, flush=True)
    time.sleep(0.5)
