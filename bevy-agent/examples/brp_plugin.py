#!/usr/bin/env python3
"""An external bevy-agent plugin over the Bevy Remote Protocol (stdlib only).

It registers a `reverse` tool, then listens on the `agent.tool_calls+watch`
stream and answers each call with `agent.tool_result`. When the agent reloads,
the stream ends; the plugin reconnects to the same port and registers again.

usage: brp_plugin.py <port>     (the port is in $BEVY_AGENT_HOME/brp-port)
"""
import json
import os
import sys
import time
import urllib.request

URL = f"http://127.0.0.1:{sys.argv[1]}/"
PLUGIN = "py-reverse"
TOOLS = [{
    "name": "reverse",
    "description": "Reverse a string. Served by an external process over BRP.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}]


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def request(method, params, **kwargs):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    headers = {"Content-Type": "application/json"}
    return urllib.request.urlopen(urllib.request.Request(URL, body, headers), **kwargs)


def call(method, params):
    with request(method, params, timeout=10) as response:
        reply = json.load(response)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["result"]


def answer(tool_call):
    text = tool_call["arguments"].get("text", "")
    return f"{text[::-1]} (served by plugin pid {os.getpid()})"


def main():
    while True:
        try:
            log(f"registered {call('agent.register_tools', {'plugin': PLUGIN, 'tools': TOOLS})}")
            with request("agent.tool_calls+watch", {"plugin": PLUGIN}) as stream:
                for raw in stream:
                    line = raw.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    for tool_call in json.loads(line[6:]).get("result") or []:
                        output = answer(tool_call)
                        call("agent.tool_result", {"id": tool_call["id"], "output": output})
                        log(f"answered {tool_call['id']}: {output}")
            log("stream ended; reconnecting")
        except Exception as error:  # the agent is restarting: retry
            log(f"not connected ({error}); retrying")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
