#!/usr/bin/env python3
"""Opt-in cheap live acceptance test; keys stay in the inherited environment."""
import fcntl
import http.server
import json
import os
from pathlib import Path
import pty
import select
import shutil
import struct
import subprocess
import termios
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "verification-artifacts"


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be exported")
    ARTIFACTS.mkdir(exist_ok=True)
    run = ARTIFACTS / str(time.time_ns())
    run.mkdir()
    private_bin = run / "bin"
    private_bin.mkdir()
    sysroot = subprocess.check_output(["rustc", "+1.98.1", "--print", "sysroot"], text=True).strip()
    (private_bin / "rustc").symlink_to(Path(sysroot) / "bin/rustc")
    (private_bin / "sh").symlink_to("/bin/sh")
    for tool in ["cc", "clang", "ld", "xcrun", "ar", "dsymutil", "codesign"]:
        path = shutil.which(tool)
        if path:
            (private_bin / tool).symlink_to(path)
    env = os.environ.copy()
    env["PATH"] = str(private_bin)
    env["TERM"] = "xterm-256color"
    assert shutil.which("dx", path=env["PATH"]) is None
    source = run / "native.rs"
    source.write_text((ROOT / "native.rs").read_text())
    (run / "fixture.txt").write_text("LIVE_SENTINEL_K7\n")
    endpoint_file = run / "endpoint.txt"
    snapshot_file = run / "frame.txt"
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 60, 160, 0, 0))
    process = subprocess.Popen([
        str(ROOT / "target/debug/bevy-coding-agent"), "--root", str(run),
        "--native", str(source), "--endpoint-file", str(endpoint_file),
        "--snapshot", str(snapshot_file),
    ], cwd=ROOT, env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    captured = bytearray()
    stop = threading.Event()

    def drain():
        while not stop.is_set():
            if select.select([master], [], [], 0.1)[0]:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                captured.extend(data)
                # Crossterm may query cursor position when initializing the backend.
                if b"\x1b[6n" in data:
                    os.write(master, b"\x1b[1;1R")

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    callback_calls = []

    class Callback(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            callback_calls.append(data)
            result = json.dumps({"words": len(data["arguments"]["text"].split()), "plugin": "external-process"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Callback)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    def wait(check, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"agent exited {process.returncode}; see {run}/terminal.ansi")
            try:
                result = check()
                if result:
                    return result
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise TimeoutError(f"acceptance deadline exceeded; see {run}")

    def rpc(method, params=None):
        request = urllib.request.Request(endpoint_file.read_text(), data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(), headers={"Content-Type": "application/json"})
        response = json.load(urllib.request.urlopen(request, timeout=3))
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def state():
        return rpc("agent/status")

    def prompt(text, via_brp=False):
        before = len(state()["lines"])
        if via_brp:
            assert rpc("agent/prompt", {"text": text})["accepted"]
        else:
            os.write(master, text.encode() + b"\r")
        wait(lambda: len(state()["lines"]) > before)
        result = wait(lambda: (s if not s["busy"] and any(line.startswith("Assistant:") for line in s["lines"][before:]) else None) if (s := state()) else None)
        lines = result["lines"][before:]
        assert not any(line.startswith("Error:") for line in lines), lines
        return result, lines

    def snapshot(name, expected):
        text = wait(lambda: (t if "| ready |" in t and all(s in t for s in expected) else None) if (t := snapshot_file.read_text()) else None)
        (run / name).write_text(text)

    try:
        wait(lambda: endpoint_file.exists())
        initial = wait(state)
        pid = initial["pid"]
        assert pid == process.pid
        first, lines = prompt("Use read_file to read fixture.txt. Reply with the exact marker from that file.")
        assert any(line.startswith("Tool: read_file") for line in lines), lines
        assert any(line.startswith("Assistant:") and "LIVE_SENTINEL_K7" in line for line in lines), lines
        snapshot("01-model-tool-answer.txt", ["Tool: read_file", "Result: LIVE_SENTINEL_K7", "Assistant:"])
        patched, lines = prompt("Read native.rs with read_file, use write_file to replace every native-v1 string with native-live-v2, preserving all signatures and leaving agent_plugin_version returning exactly 1 (it is the ABI, not the behavior version). Make no other edits. Then call patch_native and native with input status. Reply with the new status only after it is native-live-v2. Do not use shell.")
        assert any(line.startswith("Tool: write_file") for line in lines), lines
        assert any(line.startswith("Tool: patch_native") for line in lines), lines
        assert any(line.startswith("Tool: native") for line in lines), lines
        assert patched["native"] == "native-live-v2", patched
        assert patched["generation"] > initial["generation"]
        assert patched["pid"] == pid
        snapshot("02-self-hot-patch.txt", ["native-live-v2", "Assistant:"])
        registration = rpc("agent/extend", {
            "name": "external_word_count", "description": "Count words using an external Python process",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
            "url": f"http://127.0.0.1:{server.server_port}/",
        })
        assert registration["registered"] == "external_word_count"
        extended, lines = prompt("Use external_word_count to count words in 'bevy native runtime'. Tell me its word count and plugin field.", via_brp=True)
        assert callback_calls and callback_calls[-1]["arguments"]["text"] == "bevy native runtime"
        assert any(line.startswith("Tool: external_word_count") for line in lines), lines
        assert any(line.startswith("Assistant:") and "3" in line and "external-process" in line for line in lines), lines
        assert extended["pid"] == pid
        snapshot("03-brp-extension.txt", ["Tool: external_word_count", "external-process", "Assistant:"])
        assert rpc("agent/unextend", {"name": "external_word_count"})["removed"]
        assert "external_word_count" not in state()["extensions"]
        evidence = {"agent_pid": pid, "same_process": True, "dx_on_path": False, "model": env.get("BEVY_AGENT_MODEL", "gpt-4o-mini"), "initial_generation": initial["generation"], "patched_generation": patched["generation"], "native": patched["native"], "callback_calls": callback_calls, "conversation": extended["lines"]}
        (run / "evidence.json").write_text(json.dumps(evidence, indent=2))
        print(f"PASS: model tool+answer in TUI, model-authored native hot patch, external BRP tool; evidence {run}")
    finally:
        if process.poll() is None:
            os.write(master, b"\x03")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        stop.set()
        reader.join(timeout=2)
        (run / "terminal.ansi").write_bytes(captured)
        os.close(master)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
