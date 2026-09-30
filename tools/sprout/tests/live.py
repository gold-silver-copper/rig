#!/usr/bin/env python3
"""Opt-in real-model/PTTY/hotpatch/BRP test. No keys are copied into artifacts.
Install terminal decoder locally: python3 -m pip install --target .sprout/python pyte
Run from tools/sprout after `cargo build --workspace --locked`.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import shutil
import struct
import subprocess
import sys
import termios
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / ".sprout/python"))
import pyte

spec = importlib.util.spec_from_file_location("uppercase", ROOT / "examples/uppercase.py")
plugin_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin_module)
rpc = plugin_module.rpc


def wait(predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"agent exited {process.returncode}; see .sprout/live.ansi")
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError("condition timed out; see .sprout/live.ansi and native-build.log")


assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be exported"
assert shutil.which("dx") is None, "test requires PATH with no dx"
source = ROOT / "native/src/behavior.rs"
original = source.read_bytes()
assert b"native: seedling" in original
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
screen = pyte.Screen(140, 40)
stream = pyte.Stream(screen)
lock = threading.Lock()
stop = threading.Event()


def read_terminal():
    import codecs
    decode = codecs.getincrementaldecoder("utf-8")("replace")
    with open(".sprout/live.ansi", "wb") as output:
        while not stop.is_set():
            if not select.select([master], [], [], 0.1)[0]:
                continue
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            output.write(data)
            output.flush()
            with lock:
                stream.feed(decode.decode(data))


def snapshot(name):
    with lock:
        text = "\n".join(screen.display)
    Path(f".sprout/{name}.txt").write_text(text)
    return text


process = subprocess.Popen([str(ROOT / "target/debug/sprout")], stdin=slave, stdout=slave, stderr=slave,
                           cwd=ROOT, env={**os.environ, "TERM": "xterm-256color"}, start_new_session=True)
os.close(slave)
reader = threading.Thread(target=read_terminal, daemon=True)
reader.start()
plugin = None
report = {"model": os.environ.get("SPROUT_MODEL", "gpt-4.1-mini"), "dx_on_path": False, "pid": process.pid}
try:
    def session_ready():
        try:
            session = json.loads(Path(".sprout/session.json").read_text())
            return session if session["pid"] == process.pid else None
        except (OSError, ValueError):
            return None
    session = wait(session_ready)
    url = session["url"]
    report["brp_url"] = url
    def status():
        return rpc(url, "sprout.status")
    initial = wait(lambda: (s if (s := status())["ticks"] > 2 else None))

    def prompt(text, final_marker):
        before = len(status()["transcript"])
        os.write(master, text.encode() + b"\r")
        return wait(lambda: (s if not (s := status())["busy"] and any(
            line.startswith("assistant:") and final_marker in line for line in s["transcript"][before:]) else None))

    first = prompt("Use shell to run printf SPROUT_TOOL_OK exactly once. Then answer exactly SPROUT_TOOL_OK.", "SPROUT_TOOL_OK")
    assert any(line.startswith("tool shell:") for line in first["transcript"])
    wait(lambda: "assistant: SPROUT_TOOL_OK" in snapshot("tui-tool-final"))
    report["prompt_tool_final"] = True

    prompt("Use shell to edit only native/src/behavior.rs: replace native: seedling with native: flourishing. Do not edit any other file, build manually, or restart anything. Then answer exactly PATCH_REQUESTED.", "PATCH_REQUESTED")
    patched = wait(lambda: (s if (s := status())["generation"] == 1 and s["native"] == "native: flourishing" else None))
    assert patched["pid"] == initial["pid"] and patched["ticks"] > initial["ticks"]
    wait(lambda: "native: flourishing" in snapshot("tui-native-patched"))
    report["native_patch"] = {k: patched[k] for k in ("pid", "ticks", "generation", "native")}

    plugin_log = open(".sprout/external-plugin.log", "w")
    plugin = subprocess.Popen([sys.executable, "examples/uppercase.py", url], stdout=plugin_log, stderr=plugin_log)
    wait(lambda: "registered uppercase" in Path(".sprout/external-plugin.log").read_text())
    external = prompt("Call the uppercase tool with text external_extension_ok. Do not use shell. Then answer exactly the returned text.", "EXTERNAL_EXTENSION_OK")
    assert any(line.startswith("tool uppercase:") for line in external["transcript"])
    assert "served call" in Path(".sprout/external-plugin.log").read_text()
    wait(lambda: "assistant: EXTERNAL_EXTENSION_OK" in snapshot("tui-brp-final"))
    report["external_process"] = {"pid": plugin.pid, "result": "EXTERNAL_EXTENSION_OK"}

    source.write_bytes(b"this is intentionally invalid Rust\n")
    wait(lambda: any("native patch rejected: cargo failed" in line for line in status()["transcript"]))
    failed = status()
    assert failed["generation"] == 1 and failed["native"] == "native: flourishing"
    report["compile_failure_preserves_old_plugin"] = True
    source.write_bytes(original)
    restored = wait(lambda: (s if (s := status())["generation"] == 2 and s["native"] == "native: seedling" else None))
    assert restored["pid"] == initial["pid"] and restored["ticks"] > patched["ticks"]
    report["second_patch"] = {k: restored[k] for k in ("pid", "ticks", "generation", "native")}
    report["transcript"] = restored["transcript"]
    Path(".sprout/live-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "transcript"}, indent=2))
finally:
    source.write_bytes(original)
    if plugin is not None:
        plugin.terminate()
        plugin.wait(timeout=10)
        plugin_log.close()
    if process.poll() is None:
        os.write(master, b"\x1b")
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
    stop.set()
    reader.join(timeout=2)
    os.close(master)
