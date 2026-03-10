"""Local HuggingFace Endpoint Simulator.

Runs the generated handler.py in a subprocess to catch import errors,
init failures, and runtime errors that ``exec()`` in the parent process
would miss (because it shares the parent's imports).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

_DRIVER_SCRIPT_STDIO = """\
import json, sys, importlib.util, traceback

repo_dir = sys.argv[1]
try:
    spec = importlib.util.spec_from_file_location("handler", f"{repo_dir}/handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    handler = mod.EndpointHandler(repo_dir)
except Exception:
    print(json.dumps({"status": "error", "error": traceback.format_exc()}), flush=True)
    sys.exit(1)

print(json.dumps({"status": "ready"}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        continue
    if msg.get("action") == "shutdown":
        break
    try:
        result = handler(msg.get("data", {}))
        print(json.dumps({"result": result}), flush=True)
    except Exception:
        print(json.dumps({"error": traceback.format_exc()}), flush=True)
"""

_DRIVER_SCRIPT_HTTP = """\
import json, sys, importlib.util, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

repo_dir = sys.argv[1]
port = int(sys.argv[2])

try:
    spec = importlib.util.spec_from_file_location("handler", f"{repo_dir}/handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    handler = mod.EndpointHandler(repo_dir)
except Exception:
    print(json.dumps({"status": "error", "error": traceback.format_exc()}), flush=True)
    sys.exit(1)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            result = handler(body)
            resp = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except Exception:
            resp = json.dumps({"error": traceback.format_exc()}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", port), Handler)
print(json.dumps({"status": "ready", "port": port}), flush=True)
server.serve_forever()
"""


class SimulatorError(Exception):
    """Raised when the simulator subprocess reports an error."""


class HFLocalSimulator:
    """Simulate HF Inference Endpoint locally in a subprocess."""

    def __init__(
        self,
        handler_code: str,
        artifacts: dict[str, bytes],
        requirements: str = "",
        isolate: bool = False,
        http: bool = False,
    ) -> None:
        self._handler_code = handler_code
        self._artifacts = artifacts
        self._requirements = requirements
        self._isolate = isolate
        self._http = http
        self._tmpdir: str | None = None
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None

    def start(self, timeout: float = 30) -> None:
        """Write files and launch the subprocess.

        Raises ``SimulatorError`` if the handler fails to load.
        """
        self._tmpdir = tempfile.mkdtemp(prefix="hf_sim_")
        tmpdir = Path(self._tmpdir)

        # Write handler.py
        (tmpdir / "handler.py").write_text(self._handler_code)

        # Write pickled artifacts
        art_dir = tmpdir / "artifacts"
        art_dir.mkdir()
        for name, data in self._artifacts.items():
            (art_dir / f"{name}.pkl").write_bytes(data)

        # Write requirements.txt
        if self._requirements:
            (tmpdir / "requirements.txt").write_text(self._requirements)

        # Write config.json (model repo structure like real HF)
        (tmpdir / "config.json").write_text(json.dumps({"framework": "custom"}))

        # Determine Python interpreter
        python = sys.executable

        if self._isolate:
            venv_dir = tmpdir / ".venv"
            # Create venv
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "venv", str(venv_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise SimulatorError(f"Failed to create venv:\n{result.stderr}")

            python = str(venv_dir / "bin" / "python")

            # Install requirements if present
            if self._requirements:
                result = subprocess.run(  # noqa: S603
                    [python, "-m", "pip", "install", "-r", str(tmpdir / "requirements.txt")],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise SimulatorError(f"pip install failed:\n{result.stderr}\n{result.stdout}")

        # Write driver script
        driver_path = tmpdir / "_driver.py"
        if self._http:
            driver_path.write_text(_DRIVER_SCRIPT_HTTP)
        else:
            driver_path.write_text(_DRIVER_SCRIPT_STDIO)

        # Launch subprocess with clean environment (no parent PYTHONPATH leak)
        env = os.environ.copy()
        env.pop("PYTHONDONTWRITEBYTECODE", None)

        cmd = [python, str(driver_path), self._tmpdir]
        if self._http:
            # Find a free port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            cmd.append(str(port))

        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Wait for ready or error
        assert self._proc.stdout is not None  # noqa: S101
        try:
            line = self._proc.stdout.readline()
        except Exception as exc:
            self.stop()
            raise SimulatorError(f"Failed to read from subprocess: {exc}") from exc

        if not line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            self.stop()
            raise SimulatorError(f"Subprocess exited without output.\nstderr: {stderr}")

        msg = json.loads(line)
        if msg.get("status") == "error":
            error_text = msg.get("error", "unknown error")
            self.stop()
            raise SimulatorError(error_text)

        if msg.get("status") != "ready":
            self.stop()
            raise SimulatorError(f"Unexpected startup message: {msg}")

        if self._http:
            self._port = msg.get("port")

    def call(self, data: dict) -> dict:
        """Send a request and return the response.

        Raises ``SimulatorError`` on handler errors.
        """
        if self._proc is None or self._proc.poll() is not None:
            raise SimulatorError("Simulator is not running")

        if self._http:
            return self._call_http(data)
        return self._call_stdio(data)

    def _call_stdio(self, data: dict) -> dict:
        assert self._proc is not None  # noqa: S101  # guaranteed by call()
        assert self._proc.stdin is not None  # noqa: S101
        assert self._proc.stdout is not None  # noqa: S101
        request = json.dumps({"action": "call", "data": data}) + "\n"
        self._proc.stdin.write(request)
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise SimulatorError("Subprocess closed stdout unexpectedly")

        msg = json.loads(line)
        if "error" in msg:
            raise SimulatorError(msg["error"])
        return dict(msg["result"])

    def _call_http(self, data: dict) -> dict[str, object]:
        url = f"http://127.0.0.1:{self._port}/"
        body = json.dumps(data).encode()
        req = urllib.request.Request(  # noqa: S310
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                result: dict[str, object] = json.loads(resp.read())
                return result
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode()
            try:
                err = json.loads(response_body)
                raise SimulatorError(err.get("error", response_body)) from exc
            except (json.JSONDecodeError, SimulatorError):
                raise
            except Exception:
                raise SimulatorError(response_body) from exc
        except urllib.error.URLError as exc:
            raise SimulatorError(f"HTTP request failed: {exc}") from exc

    def stop(self) -> None:
        """Shut down the subprocess and clean up the temp directory."""
        if self._proc is not None and self._proc.poll() is None:
            if self._http:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
                    self._proc.wait()
            elif self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
                    self._proc.wait()
        self._proc = None
        self._port = None

        if self._tmpdir is not None:
            import shutil

            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self) -> HFLocalSimulator:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
