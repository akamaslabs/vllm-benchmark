import json
import signal
import socket
import threading
import unittest
import urllib.error
import urllib.request

import server
from appfakes import blocked_app, loaded_app


class ServerTest(unittest.TestCase):
    def start(self, app):
        srv = server.make_server(app, 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def get(self, url):
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())

    def post(self, url, body, headers=None):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_ping(self):
        a, _ = loaded_app()
        self.assertEqual(self.get(self.start(a) + "/api/ping"), (200, {"ok": True, "name": "istanceAws"}))

    def test_state_is_json(self):
        a, _ = loaded_app()
        status, body = self.get(self.start(a) + "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("groups", body)

    def test_index_served(self):
        a, _ = loaded_app()
        with urllib.request.urlopen(self.start(a) + "/", timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers["Content-Type"])

    def test_probe_accepted_is_202(self):
        a, _ = loaded_app()
        status, body = self.post(self.start(a) + "/api/probe", {"types": ["g6.xlarge"], "regions": ["us-east-2"]})
        a.wait_idle(5)
        self.assertEqual((status, body["planned"]), (202, 1))

    def test_probe_conflict_is_409(self):
        a, _, gate = blocked_app()
        base = self.start(a)
        try:
            self.post(base + "/api/probe", {"types": ["g7e.2xlarge"], "regions": ["us-east-2"]})
            status, body = self.post(base + "/api/probe", {"types": ["g6.xlarge"], "regions": ["us-east-2"]})
        finally:
            gate.set()
            a.wait_idle(5)
        self.assertEqual(status, 409)
        self.assertIn("us-east-2", body["error"])

    def test_probe_empty_is_400(self):
        a, _ = loaded_app()
        status, _ = self.post(self.start(a) + "/api/probe", {"types": [], "regions": ["us-east-2"]})
        self.assertEqual(status, 400)

    def test_post_text_plain_rejected(self):
        a, rec = loaded_app()
        status, _ = self.post(self.start(a) + "/api/probe", {"types": ["g6.xlarge"], "regions": ["us-east-2"]},
                              {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.assertEqual(rec.probes, [])

    def test_post_foreign_origin_rejected(self):
        a, rec = loaded_app()
        status, _ = self.post(self.start(a) + "/api/probe", {"types": ["g6.xlarge"], "regions": ["us-east-2"]},
                              {"Content-Type": "application/json", "Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(rec.probes, [])

    def test_post_foreign_host_rejected(self):
        a, rec = loaded_app()
        status, _ = self.post(self.start(a) + "/api/probe", {"types": ["g6.xlarge"], "regions": ["us-east-2"]},
                              {"Content-Type": "application/json", "Host": "evil.example:8765"})
        self.assertEqual(status, 403)
        self.assertEqual(rec.probes, [])

    def test_post_same_origin_accepted(self):
        a, _ = loaded_app()
        base = self.start(a)
        status, _ = self.post(base + "/api/probe", {"types": ["g6.xlarge"], "regions": ["us-east-2"]},
                              {"Content-Type": "application/json", "Origin": base})
        a.wait_idle(5)
        self.assertEqual(status, 202)


class MainTest(unittest.TestCase):
    def test_busy_port_exits_without_sweeping(self):
        a, rec = loaded_app()
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen()
        self.addCleanup(s.close)
        rc = server.main(["--port", str(s.getsockname()[1]), "--no-browser"], app_factory=lambda: a)
        self.assertNotEqual(rc, 0)
        self.assertEqual(rec.sweeps, [])

    def test_startup_sweep_covers_all_regions(self):
        a, rec = loaded_app()
        server.startup_sweep(a)
        self.assertEqual(sorted(rec.sweeps), sorted(server.ALL_REGIONS))

    def test_stop_signals_install_and_ignore(self):
        saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        self.addCleanup(lambda: [signal.signal(s, h) for s, h in saved.items()])
        server.install_stop_signals()
        self.assertIs(signal.getsignal(signal.SIGHUP), signal.getsignal(signal.SIGTERM))
        self.assertNotIn(signal.getsignal(signal.SIGHUP), (signal.SIG_DFL, signal.SIG_IGN))
        server.ignore_stop_signals()
        for s in saved:
            self.assertIs(signal.getsignal(s), signal.SIG_IGN)

    def test_finish_sweeps_even_if_terminal_is_gone(self):
        a, rec = loaded_app()
        def dead_tty(*args, **kw):
            raise OSError(5, "Input/output error")
        server.finish(a, out=dead_tty)
        self.assertIn("us-east-2", rec.sweeps)

    def test_shutdown_waits_longer_than_one_cli_call(self):
        import awscli
        self.assertGreater(server.SHUTDOWN_WAIT_S, awscli.CLI_TIMEOUT_S)

    def test_shutdown_report_shows_still_active(self):
        text = server.format_shutdown({"us-east-2": {"cancelled": [], "still_active": ["cr-x"]}})
        self.assertIn("STILL ACTIVE", text)
        self.assertIn("cr-x", text)


if __name__ == "__main__":
    unittest.main()
