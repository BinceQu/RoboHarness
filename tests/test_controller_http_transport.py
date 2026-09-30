from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

from roboharness.http_transport import LoopbackHTTPConnection
from roboharness.runner import request_json


@contextmanager
def counting_server(*, hold_response=False):
    state = {'bodies': [], 'release': threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            state['bodies'].append(self.rfile.read(int(self.headers['Content-Length'])))
            if hold_response:
                state['release'].wait(5)
            try:
                data = b'{"ok":true}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server.server_address[1], state
    finally:
        state['release'].set()
        server.shutdown()
        server.server_close()
        thread.join(2)


class ControllerHTTPTransportTests(unittest.TestCase):
    def test_connect_failures_retry_before_exactly_one_session_begin(self):
        real_connect = socket.create_connection
        attempts = []

        def flaky_connect(*args, **kwargs):
            attempts.append(args[1])
            if len(attempts) < 3:
                raise TimeoutError('TCP establishment timed out')
            return real_connect(*args, **kwargs)

        with counting_server() as (port, state):
            with patch('http.client.socket.create_connection', side_effect=flaky_connect), \
                    patch.object(LoopbackHTTPConnection, 'connect_backoff_s', 0):
                result = request_json(port, '/api/agent_monitor/session_begin', {'session_id': 'test-only'})
            self.assertEqual(result, {'ok': True})
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(0 < timeout <= 3 for timeout in attempts))
            self.assertEqual([json.loads(body) for body in state['bodies']], [{'session_id': 'test-only'}])

    def test_exhausted_connect_attempts_deliver_no_request(self):
        with counting_server() as (port, state):
            with patch('http.client.socket.create_connection', side_effect=TimeoutError('connect')) as connect, \
                    patch.object(LoopbackHTTPConnection, 'connect_backoff_s', 0):
                with self.assertRaises(URLError):
                    request_json(port, '/test-only', {'operation': 'once'}, timeout=0.4)
            self.assertEqual(connect.call_count, 3)
            self.assertTrue(all(0 < call.args[1] <= 0.4 for call in connect.call_args_list))
            self.assertEqual(state['bodies'], [])

    def test_response_timeout_never_replays_delivered_request(self):
        real_connect = socket.create_connection
        with counting_server(hold_response=True) as (port, state):
            with patch('http.client.socket.create_connection', wraps=real_connect) as connect:
                with self.assertRaises((TimeoutError, URLError)):
                    request_json(port, '/test-only', {'operation': 'once'}, timeout=0.15)
            self.assertEqual(connect.call_count, 1)
            self.assertEqual([json.loads(body) for body in state['bodies']], [{'operation': 'once'}])

    def test_success_restores_original_response_timeout(self):
        with counting_server() as (port, state):
            conn = LoopbackHTTPConnection('127.0.0.1', port, timeout=15)
            self.addCleanup(conn.close)
            conn.connect()
            self.assertEqual(conn.timeout, 15)
            self.assertEqual(conn.sock.gettimeout(), 15)

    def test_tunnels_and_non_loopback_hosts_are_not_retried(self):
        cases = [LoopbackHTTPConnection('example.invalid', 80, timeout=15),
                 LoopbackHTTPConnection('127.0.0.1', 1, timeout=15)]
        cases[1].set_tunnel('127.0.0.1', 2)
        for conn in cases:
            with self.subTest(host=conn.host, tunnel=conn._tunnel_host):
                with patch('http.client.HTTPConnection.connect', side_effect=TimeoutError('single')) as connect:
                    with self.assertRaises(TimeoutError):
                        conn.connect()
                connect.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
