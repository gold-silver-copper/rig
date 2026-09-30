#!/usr/bin/env python3
"""A separate, dependency-free BRP tool plugin. Run from tools/sprout."""
import json
from pathlib import Path
import sys
import time
import urllib.request


def rpc(url, method, params=None):
    request = urllib.request.Request(
        url,
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        reply = json.load(response)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["result"]


def run(url):
    rpc(url, "sprout.register", {
        "name": "uppercase", "description": "Uppercase text in an external Python process.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
    })
    print("registered uppercase over BRP", flush=True)
    while True:
        for call in rpc(url, "sprout.poll", {"name": "uppercase"}):
            text = call["arguments"]["text"].upper()
            rpc(url, "sprout.result", {"id": call["id"], "text": text})
            print(f"served call {call['id']}: {text}", flush=True)
        time.sleep(0.15)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else json.loads(Path(".sprout/session.json").read_text())["url"]
    try:
        run(url)
    except KeyboardInterrupt:
        pass
