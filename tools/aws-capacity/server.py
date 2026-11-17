"""istanceAws: local page with live AWS GPU capacity (spot automatic, on-demand on request)."""
import argparse
import json
import os
import signal
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app import ALL_REGIONS, DEFAULT_PROBE_REGIONS, DEFAULT_REGIONS, App, Busy
from awscli import CLI_TIMEOUT_S

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = int(os.environ.get("ISTANCEAWS_PORT", "8765"))


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal for what matters
            pass

        def _json(self, status: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _allowed(self) -> bool:
            """Only this page may drive the server: Host must be ours (no DNS rebinding) and, on a
            cross-origin request, Origin must be ours (no other site starting paid probes)."""
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in hosts or (origin and origin not in {f"http://{h}" for h in hosts}):
                self._json(403, {"error": "request not from the istanceAws page"})
                return False
            return True

        def do_GET(self):
            if not self._allowed():
                return
            if self.path in ("/", "/index.html"):
                data = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/ping":
                self._json(200, {"ok": True, "name": "istanceAws"})
            elif self.path == "/api/state":
                self._json(200, app.snapshot())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._allowed():
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._json(415, {"error": "send application/json"})
                return
            try:
                body = self._body()
                if self.path == "/api/spot/refresh":
                    app.request_spot_refresh()
                    self._json(202, {"ok": True})
                elif self.path == "/api/regions":
                    app.set_regions(list(body.get("regions") or []))
                    self._json(202, {"ok": True})
                elif self.path == "/api/probe":
                    plan = app.start_probe(list(body.get("types") or []), list(body.get("regions") or []))
                    self._json(202, plan)
                else:
                    self._json(404, {"error": "not found"})
            except Busy as e:
                self._json(409, {"error": str(e)})
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
SHUTDOWN_WAIT_S = CLI_TIMEOUT_S + 15  # an in-flight create must return before the sweep runs


def _stop(signum, frame):
    raise KeyboardInterrupt


def install_stop_signals() -> None:
    """Ctrl+C, kill and closing the terminal all go through the same sweep."""
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)


def ignore_stop_signals() -> None:
    """Once the sweep starts, a second Ctrl+C or a closed terminal must not cut it short."""
    for s in STOP_SIGNALS:
        signal.signal(s, signal.SIG_IGN)


def startup_sweep(app: App, regions: list[str] = ALL_REGIONS) -> None:
    """Cancel what a previous run left open (hard kill, crash) in every region it could probe."""
    for r in regions:
        try:
            left = app.r.sweep(r)
            if left:
                print(f"start-up sweep {r}: cancelled leftover reservations {', '.join(left)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"start-up sweep {r} failed: {e}", flush=True)


def format_shutdown(report: dict) -> str:
    lines = []
    for r, x in report.items():
        done = f"cancelled {', '.join(x['cancelled'])}" if x["cancelled"] else "nothing to cancel"
        lines.append(f"  {r}: {done}")
        if x["still_active"]:
            lines.append(f"  !! {r}: STILL ACTIVE {', '.join(x['still_active'])}: cancel them in the EC2 "
                         "console; each expires on its own within 1 h of its probe")
    return "\n".join(lines)


def finish(app: App, out=print) -> None:
    """Sweep first, talk after: with the terminal closed (SIGHUP) any write can raise, and the
    sweep must not depend on it."""
    def say(text: str) -> None:
        try:
            out(text, flush=True)
        except OSError:
            pass
    say(f"\nstopping: waiting up to {SHUTDOWN_WAIT_S} s for running probes, then sweeping. Ctrl+C is "
        f"ignored now; if it hangs, kill -9 {os.getpid()} (reservations expire within 1 h of their probe).")
    report = app.shutdown(SHUTDOWN_WAIT_S)
    say(format_shutdown(report))


def main(argv: list[str] | None = None, app_factory=None) -> int:
    ap = argparse.ArgumentParser(prog="istanceAws", description=__doc__)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    app = app_factory() if app_factory else App(DEFAULT_REGIONS, DEFAULT_PROBE_REGIONS)
    try:
        # bind first: the port is the single-instance lock, so a second server never sweeps
        # (and cancels) the reservation a running one is probing with
        srv = make_server(app, args.port)
    except OSError as e:
        print(f"istanceAws: port {args.port} is busy ({e}); is another istanceAws running?", file=sys.stderr)
        return 1
    startup_sweep(app)

    url = f"http://127.0.0.1:{args.port}/"
    app.start_background()
    print(f"istanceAws on {url}  (Ctrl+C stops it and cancels any reservation it created)", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()

    install_stop_signals()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ignore_stop_signals()
        finish(app)
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
