"""Bounded loopback connection retries without replaying HTTP requests."""

from http.client import HTTPConnection
import socket
import time
from urllib.request import HTTPHandler


class LoopbackHTTPConnection(HTTPConnection):
    """Retry only TCP establishment, before HTTP headers or bodies are sent."""

    connect_attempts = 3
    connect_timeout_s = 3.0
    connect_budget_s = 10.0
    connect_backoff_s = 0.1

    def connect(self):
        # A tunnel can send HTTP bytes inside connect(); never replay it.
        if self._tunnel_host or self.host not in ('127.0.0.1', '::1'):
            return super().connect()
        original_timeout = self.timeout
        response_timeout = (socket.getdefaulttimeout()
                            if original_timeout is socket._GLOBAL_DEFAULT_TIMEOUT
                            else original_timeout)
        budget = self.connect_budget_s
        if response_timeout is not None:
            budget = min(budget, response_timeout)
        deadline = time.monotonic() + budget
        try:
            for attempt in range(self.connect_attempts):
                self.timeout = min(self.connect_timeout_s,
                                   max(0.001, deadline - time.monotonic()))
                try:
                    super().connect()
                except OSError:
                    # HTTPConnection.close() would reset the pending request.
                    if self.sock is not None:
                        self.sock.close()
                        self.sock = None
                    remaining = deadline - time.monotonic()
                    if attempt + 1 == self.connect_attempts or remaining <= 0:
                        raise
                    time.sleep(min(self.connect_backoff_s, remaining))
                else:
                    self.sock.settimeout(response_timeout)
                    return
        finally:
            self.timeout = original_timeout


class LoopbackHTTPHandler(HTTPHandler):
    def http_open(self, request):
        return self.do_open(LoopbackHTTPConnection, request)
