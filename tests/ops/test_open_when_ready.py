import http.server
import json
import threading
import time

from terrascope.utils.open_when_ready import main, wait_health


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            b = json.dumps({"ready": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def test_waits_for_health_then_returns():
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Timer(1.5, lambda: threading.Thread(target=srv.serve_forever, daemon=True).start()).start()
    t0 = time.time()
    h = wait_health(f"http://127.0.0.1:{port}", timeout=20, interval=0.3)
    srv.shutdown()
    assert h == {"ready": True}
    assert time.time() - t0 >= 1.0  # it waited for the server, it did not answer early


def test_timeout_returns_1_without_browser():
    assert main(["http://127.0.0.1:9", "--timeout", "1", "--no-browser"]) == 1
