#!/usr/bin/env python3
"""Drive a rig-pi session over BRP from another process.

    python3 examples/remote_client.py PORT [--session ID | --model REF] PROMPT...

Without --session it creates a remote session (on --model, if given). Each
prompt is sent in turn; the session's transcript streams to stdout until the
agent is idle again.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request
import uuid


def call(url, method, params=None):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    request = urllib.request.Request(url, data=json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=30) as reply:
        answer = json.load(reply)
    if "error" in answer:
        raise RuntimeError(answer["error"])
    return answer["result"]


def stream_entries(url, session, since):
    """Print the session's new transcript entries as the watch stream sends them."""
    watch = {"session": session, "stream": uuid.uuid4().hex, "since": since}
    body = {"jsonrpc": "2.0", "id": 1, "method": "rig_pi.session.watch+watch", "params": watch}
    with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode())) as stream:
        for line in stream:
            line = line.decode().strip()
            if line.startswith("data: "):
                for entry in json.loads(line[6:])["result"]["entries"]:
                    print(f"[{entry['kind']}] {entry['text']}", flush=True)


def run_prompt(url, session, prompt):
    since = call(url, "rig_pi.session.get", {"session": session})["next"]
    threading.Thread(target=stream_entries, args=(url, session, since), daemon=True).start()
    call(url, "rig_pi.session.prompt", {"session": session, "text": prompt})
    idle_seen = 0
    while idle_seen < 3:
        time.sleep(0.5)
        state = call(url, "rig_pi.session.get", {"session": session, "since": 10**9})
        idle_seen = idle_seen + 1 if state["state"] == "idle" and state["queued"] == 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("--session")
    parser.add_argument("--model")
    parser.add_argument("prompts", nargs="+")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    session = args.session
    if session is None:
        params = {"name": "remote-client"}
        if args.model:
            params["model"] = args.model
        created = call(url, "rig_pi.session.create", params)
        session = created["session"]
        print(f"created session {session} on {created['model']}", flush=True)
    for prompt in args.prompts:
        run_prompt(url, session, prompt)
    print(f"session {session} idle", flush=True)


if __name__ == "__main__":
    sys.exit(main())
