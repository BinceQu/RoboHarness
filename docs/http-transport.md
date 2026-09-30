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

An independent clone of commit `a57d196` also passed all 33 root tests using
its own Python environment. See the [clone audit](../validation_results/http-transport-20261001/independent-clone-audit.json)
and [test log](../validation_results/http-transport-20261001/independent-clone-tests.txt).

### Recovery from a captured live collision

At **2026-10-01 02:23:01 Asia/Shanghai**, the raw observer's task01 GET
timed out after 10.012 seconds. Its source port was `49504` and the reverse
server tuple on `15071` was in `LAST-ACK`. After that failed probe closed,
a separate verifier confirmed the closing server socket remained and no
client socket occupied that tuple.

The verifier then issued one read-only monitor GET through the new handler.
It bound only the first connection attempt to the captured source port.
That real connection timed out after 3.005 seconds; the handler retried with
normal ephemeral allocation, connected from source port `38676`, and
returned the expected active session and 10535-step budget in 3.115 seconds
overall. No timeout exception was simulated. The procedure and captured
records are in the [live retry audit](../validation_results/http-transport-20261001/live-retry-audit.json).

This verifies recovery from the captured connection conflict in a separate
client process. It does not establish why the older closing socket remained,
or prove task-score reproduction. The existing evaluators and controllers
were not restarted or modified by this check.

The session-local diagnostic observer probes only the three verified live
run ports once every 90 seconds and records exception stacks and socket
states. It sends no control requests and exits when all runs are terminal.

### A surfaced agent timeout

A read-only audit of the live case-301 transcripts found one surfaced
transport error in task03. The agent invoked `open_gripper` at **2026-10-01
00:05:55.516 Asia/Shanghai** and received `transport_error` at 00:06:04.868,
after 9.352 seconds. The agent's connection helper was already using bounded
establishment retries. A transport error nevertheless reached the agent;
bounded retries do not guarantee that every such failure is hidden.

The inspected server log window, 00:05:00 through 00:07:59, contains no
queued gripper job or matching POST access entry for the failed invocation.
Controller ticks remain 1258 through 00:06:27. The agent subsequently
invoked the tool again at 00:07:17.678. The server records one queued and
completed gripper job for that later invocation, one successful monitor
card, and ticks increasing to 1270. The log contains both plain and
logger-formatted access entries; action counts therefore use queued and
completed jobs and cards rather than access-line counts.

The initial duration is consistent with three three-second connection
attempts plus backoff. The live MCP request timeout is 1900 seconds and
the relevant client source matches its launch commit. However, no
per-attempt stack or socket snapshot was captured during the original
00:06 failure, so its exact transport phase and cause remain unproven.

The [agent tool audit](../validation_results/http-transport-20261001/agent-tool-audit.json)
preserves the invocation/result timestamps, server records, step counts,
source hashes and exact byte-prefix hashes of the live logs. It also
distinguishes task01's stale-image rejection and task08's out-of-range
pixel argument from this transport error. Three occurrences of `HTTP 504`
in task01 are skill instruction text, not failed requests.

The audit issues no actuator requests and does not restart or modify the
running evaluations. It establishes no completed official score.
