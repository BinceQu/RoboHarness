# Controller HTTP connection failures

An observation recorded at **2026-10-01 01:18:28 Asia/Shanghai** captured a
real, ten-second connection timeout while the task03 evaluator and agent
were still alive. The evidence is in
[the transport audit](../validation_results/http-transport-20261001/audit.json).

The exception originated in Python's `socket.create_connection`, before any
HTTP request was sent. At the same time, the diagnostic client's socket was
`127.0.0.1:34574 -> 127.0.0.1:15073` in `SYN-SENT`, while the reverse tuple at
the server was still in `LAST-ACK` with a retransmission timer. The listener
queue was `0/128`. This captured failure was TCP establishment, rather than
waiting for an HTTP response or for the model to finish.

The paired states are consistent with a new connection colliding with an
older closing connection. This observation does **not** establish why the
older FIN was not acknowledged, nor which kernel or network component caused
that condition. No global TCP settings were changed. Linux documents
loopback TIME-WAIT reuse in its
[IP sysctl reference](https://kernel.org/doc/html/v6.7/networking/ip-sysctl.html#tcp-tw-reuse-integer);
that setting alone does not prove the cause of this incident.

## Controller fix

`roboharness.http_transport.LoopbackHTTPConnection` adds the same bounded
TCP-establishment policy already present in the released Claude agent:

- At most three establishment attempts, each at most three seconds.
- A total establishment budget of at most ten seconds, further bounded by
  the caller's request timeout.
- Retry only before HTTP headers or a body are sent. Neither a response
  timeout nor a server error replays the request.
- Restore the caller's socket timeout after a connection succeeds.
- Do not retry HTTP tunnels or non-loopback destinations.

The controller uses this handler for its existing JSON requests, including
the session-begin request between instances. Its request payloads, endpoints,
budget calculations, prompts and score handling are unchanged. Existing r5
controllers continue running their launch-time implementation; this change
applies to subsequent controller launches.

## Verification

All **33 root regression tests pass**, including five new transport checks.
An actual local HTTP server verifies that two simulated establishment
failures followed by success deliver exactly one session-begin POST; failed
establishment delivers none; and a response timeout does not replay a POST
already received by the server. The other new checks cover restoring the
response timeout and bypassing retries for tunnels and non-loopback hosts.

Read-only GET requests through the new handler also succeeded against all
three live interfaces. Their active session IDs and budgets remained the
expected values. These are transport checks, **not completed task scores**.

The session-local diagnostic observer probes only the three verified live
run ports once every 90 seconds and records exception stacks and socket
states. It sends no control requests and exits when all runs are terminal.
