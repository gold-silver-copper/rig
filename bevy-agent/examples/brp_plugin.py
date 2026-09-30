#!/usr/bin/env python3
"""An out-of-process rigpi plugin, speaking the Bevy Remote Protocol.

Registers a `py_eval` tool, then answers every call the model makes to it.
Standard library only.

    python3 examples/brp_plugin.py PORT
"""

import json
import sys
import time
import urllib.request

URL = f"http://127.0.0.1:{sys.argv[1]}/"


def brp(method, params=None):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    request = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        reply = json.load(response)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply.get("result")


SAFE = {name: __builtins__.__dict__[name] for name in
        ["abs", "all", "any", "len", "max", "min", "range", "round", "sorted", "str", "sum"]}


def py_eval(arguments):
    return repr(eval(arguments["expression"], {"__builtins__": SAFE}, {}))


brp("rigpi/register_tool", {
    "name": "py_eval",
    "description": "Evaluate a Python expression in an external plugin process and return repr() of the result.",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
})
print("registered py_eval; serving calls", flush=True)

try:
    while True:
        for call in brp("rigpi/take_calls", {"tools": ["py_eval"]}):
            try:
                output, is_error = py_eval(call["arguments"]), False
            except Exception as error:  # the model sees the failure
                output, is_error = f"{type(error).__name__}: {error}", True
            print(f"py_eval {call['arguments']} -> {output}", flush=True)
            brp("rigpi/complete_call", {"call": call["call"], "output": output, "is_error": is_error})
        time.sleep(0.2)
except KeyboardInterrupt:
    brp("rigpi/unregister_tool", {"name": "py_eval"})
