from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from embodied_claude_code.config import OwnedPortHTTPConnection, OwnedPortHTTPHandler


@contextmanager
def counting_server(*, hold_response=False):
    state = {"posts": 0, "bodies": [], "release": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            state["bodies"].append(self.rfile.read(int(self.headers["Content-Length"])))
            state["posts"] += 1
            if hold_response:
                state["release"].wait(5)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield server.server_address, state
    finally:
        state["release"].set()
        server.shutdown()
        server.server_close()
        worker.join(2)


class OwnedHTTPTransportTests(unittest.TestCase):
    def test_connect_timeouts_retry_before_single_post_and_restore_response_timeout(self):
        with counting_server() as (address, state):
            conn = OwnedPortHTTPConnection(*address, timeout=15)
            self.addCleanup(conn.close)
            real_connect = conn._create_connection
            attempts = []

            def flaky_connect(*args, **kwargs):
                attempts.append(args[1])
                if len(attempts) < 3:
                    raise TimeoutError("TCP establishment timed out")
                return real_connect(*args, **kwargs)

            with patch.object(conn, "_create_connection", side_effect=flaky_connect), \
                    patch.object(conn, "connect_backoff_s", 0):
                conn.request("POST", "/test-only", body=b"one movement")
                self.assertEqual(conn.timeout, 15)
                self.assertEqual(conn.sock.gettimeout(), 15)
                self.assertEqual(conn.getresponse().read(), b"ok")
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(0 < value <= 3 for value in attempts))
            self.assertEqual(state["posts"], 1)
            self.assertEqual(state["bodies"], [b"one movement"])

    def test_exhausted_connect_attempts_never_deliver_post(self):
        with counting_server() as (address, state):
            conn = OwnedPortHTTPConnection(*address, timeout=0.4)
            self.addCleanup(conn.close)
            with patch.object(conn, "_create_connection", side_effect=TimeoutError("connect")) as connect, \
                    patch.object(conn, "connect_backoff_s", 0):
                with self.assertRaises(TimeoutError):
                    conn.request("POST", "/test-only", body=b"movement")
            self.assertEqual(connect.call_count, 3)
            self.assertTrue(all(0 < call.args[1] <= 0.4 for call in connect.call_args_list))
            self.assertEqual(conn.timeout, 0.4)
            self.assertEqual(state["posts"], 0)

    def test_response_timeout_does_not_replay_delivered_post(self):
        with counting_server(hold_response=True) as (address, state):
            opener = build_opener(ProxyHandler({}), OwnedPortHTTPHandler())
            request = Request(f"http://{address[0]}:{address[1]}/test-only", data=b"one movement")
            original_connect = OwnedPortHTTPConnection.connect
            connections = []

            def connect(conn):
                connections.append(conn)
                return original_connect(conn)

            with patch.object(OwnedPortHTTPConnection, "connect", connect):
                with self.assertRaises((TimeoutError, URLError)):
                    opener.open(request, timeout=0.2)
            self.assertEqual(state["posts"], 1)
            self.assertEqual(len(connections), 1)

    def test_tunnel_path_uses_single_standard_connection_attempt(self):
        conn = OwnedPortHTTPConnection("127.0.0.1", 1, timeout=15)
        conn.set_tunnel("127.0.0.1", 2)
        with patch("http.client.HTTPConnection.connect", side_effect=TimeoutError("tunnel")) as connect:
            with self.assertRaises(TimeoutError):
                conn.connect()
        connect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
